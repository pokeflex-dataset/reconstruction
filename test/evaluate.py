import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import argparse
import torch
from dataset.dataset_meshes import MeshDataset, collate_fn
from torch.utils.data import DataLoader, ConcatDataset
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import (
    chamfer_distance, point_mesh_face_distance
)
from pytorch3d.ops import sample_points_from_meshes
import trimesh
import matplotlib.pyplot as plt
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_models(path):
    decoder = torch.load(f"{path}/decoder.pth", map_location=device)
    try:
        image_encoder = torch.load(f"{path}/image_encoder.pth", map_location=device)
    except FileNotFoundError:
        image_encoder = None
    try:
        attention_model = torch.load(f"{path}/attention_model.pth", map_location=device)
    except FileNotFoundError:
        attention_model = None
    try:
        force_encoder = torch.load(f"{path}/force_encoder.pth", map_location=device)
    except FileNotFoundError:
        force_encoder = None
    try:
        pointcloud_encoder = torch.load(f"{path}/pointcloud_encoder.pth", map_location=device)
    except FileNotFoundError:
        pointcloud_encoder = None

    return (decoder, image_encoder, attention_model, force_encoder, pointcloud_encoder)


def load_data(data_paths, modality, config):
    datasets = []
    for data_path in data_paths:
        if modality == "robot_data":
            current_set = MeshDataset(data_path, False, False, config['cameras'], 
                                      config['pointcloud_source'], config['sequence_length'])
        elif modality in ["images", "img_robot_data"]:
            current_set = MeshDataset(data_path, True, False, config['cameras'], 
                                      config['pointcloud_source'], config['sequence_length'])
        else:
            current_set = MeshDataset(data_path, False, True, config['cameras'], 
                                      config['pointcloud_source'], config['sequence_length'])
        datasets.append(current_set)
    combined_dataset = ConcatDataset(datasets)

    dataloader = DataLoader(combined_dataset, batch_size=config['batch_size'], shuffle=True,
                            collate_fn=lambda b, device=device: collate_fn(b, device), drop_last=True)

    return dataloader


def get_roi_meshes(meshes, target_position, distance_threshold, transformations):
    face_indices_array = []
    for i, mesh in enumerate(meshes):
        verts = mesh.verts_packed()
        pos = target_position[i].unsqueeze(0).to(device)
        distances_squared = torch.sum((verts - pos) ** 2, dim=1)

        # get vertices
        inverse_transformation = transformations[i].inverse().to(device)
        transformed_verts = inverse_transformation.transform_points(verts)
        vertices_mask = distances_squared < distance_threshold ** 2
        bounding_box_mask = transformed_verts[:, 1] > (torch.min(transformed_verts[:, 1]) + 0.2)
        combined_mask = vertices_mask & bounding_box_mask
        vertices_indices = combined_mask.nonzero(as_tuple=True)[0]

        # get faces
        faces = mesh.faces_packed()
        faces_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=device)
        for index in vertices_indices:
            faces_mask |= (faces == index).any(dim=1)

        face_indices = faces_mask.nonzero(as_tuple=True)[0]
        face_indices_array.append([face_indices])

    selected_meshes = meshes.submeshes(face_indices_array)

    return selected_meshes


def get_features(models, input_data, forces, batch_size, data_modality, sequence_length):
    decoder, image_encoder, attention_model, force_encoder, pointcloud_encoder = models
    if data_modality == "images":
        features = image_encoder.forward(input_data)
        if sequence_length > 1:
            features = features.view(sequence_length, batch_size, features.shape[1])
            features = attention_model.forward(features)


    elif data_modality == "pointclouds":
        seq_length_pcd = int(input_data.shape[0] / batch_size)
        features = pointcloud_encoder.encoder.forward(input_data.permute(0, 2, 1))
        if seq_length_pcd > 1:
            features = features.view(seq_length_pcd, batch_size, features.shape[1])
            features = attention_model.forward(features)


    elif data_modality == "robot_data":
        features = force_encoder.forward(forces)
        if sequence_length > 1:
            features = features.view(sequence_length, batch_size, features.shape[1])
            features = attention_model.forward(features)


    elif data_modality == "img_robot_data":
        image_features = image_encoder.forward(input_data)
        force_features = force_encoder.forward(forces)
        image_features = image_features.view(sequence_length, batch_size, image_features.shape[1])
        force_features = force_features.view(sequence_length, batch_size, force_features.shape[1])
        features = attention_model.forward(image_features, force_features)

    elif data_modality == "pointcloud_forces":
        seq_length_pcd = int(input_data.shape[0] / batch_size)
        pointcloud_features = pointcloud_encoder.encoder.forward(input_data.permute(0, 2, 1))
        force_features = force_encoder.forward(forces)
        pointcloud_features = pointcloud_features.view(seq_length_pcd, batch_size,
                                                        pointcloud_features.shape[1])
        force_features = force_features.view(sequence_length, batch_size, force_features.shape[1])
        features = attention_model.forward(pointcloud_features, force_features)

    return features


def loss(decoder, template_vertices, template_faces, target_meshes, 
         features, forces, num_points, transformations,
         batch_size, sequence_length, scales, centers):
    # predict deformation
    coordinates = decoder.forward(features, template_vertices)
    coordinates = coordinates.reshape(batch_size, 10000, 3)

    # create new source mesh
    predicted_mesh_vertices = [coordinates[s][:num_points[s]] for s in range(batch_size)]
    predicted_meshes = Meshes(verts=predicted_mesh_vertices, faces=template_faces)
    # chamfer loss
    sampled_points = 10000
    predicted_sampled, normals_predicted = sample_points_from_meshes(predicted_meshes, sampled_points,
                                                                     return_normals=True)
    target_sampled, normals_target = sample_points_from_meshes(target_meshes, sampled_points,
                                                               return_normals=True)

    pcl = Pointclouds(predicted_sampled)
    point_face_loss = point_mesh_face_distance(target_meshes, pcl, min_triangle_area=1e-7)

    pcl2 = Pointclouds(target_sampled)
    template_mesh = Meshes(verts=[verts for verts in template_vertices], faces=template_faces)
    pfd_temp_gt = point_mesh_face_distance(template_mesh, pcl2, min_triangle_area=1e-7)

    try:
        j_pred_gt = jaccard_idx(predicted_meshes, target_meshes)
        dj_temp_gt = 1 - jaccard_idx(template_mesh, target_meshes)
    except ValueError:
        j_pred_gt = None
        dj_temp_gt = None


    # roi loss
    sampled_points_roi = 2500
    ee_pos = [force[3:] for force in forces.view(sequence_length, batch_size, 6)[-1]]
    predicted_roi_meshes = get_roi_meshes(predicted_meshes, ee_pos, 0.4, transformations)
    try:
        predicted_roi_sampled, predicted_roi_normals = sample_points_from_meshes(predicted_roi_meshes,
                                                                                 sampled_points_roi,
                                                                                 return_normals=True)
        chamfer_loss_roi, normals_loss_roi = chamfer_distance(predicted_roi_sampled, target_sampled,
                                                              x_normals=predicted_roi_normals,
                                                              y_normals=normals_target, 
                                                              single_directional=True)

    except ValueError:
        chamfer_loss_roi = torch.tensor(0.05, device=device)
        normals_loss_roi = torch.tensor(0.4, device=device)

    jaccard_dist = dj_temp_gt
    #y = j_pred_gt
    ratio = (point_face_loss/pfd_temp_gt).detach().cpu().numpy()

    return point_face_loss, chamfer_loss_roi, j_pred_gt, ratio.tolist(), dj_temp_gt


def jaccard_idx(mesh1, mesh2):
    mesh1_verts = mesh1[0].verts_packed().detach().cpu().numpy()
    mesh1_faces = mesh1[0].faces_packed().detach().cpu().numpy()
    mesh2_verts = mesh2[0].verts_packed().detach().cpu().numpy()
    mesh2_faces = mesh2[0].faces_packed().detach().cpu().numpy()

    predicted_mesh = trimesh.Trimesh(vertices=mesh1_verts, faces=mesh1_faces)
    target_mesh = trimesh.Trimesh(vertices=mesh2_verts, faces=mesh2_faces)

    union = predicted_mesh.union(target_mesh)
    intersection = predicted_mesh.intersection(target_mesh)

    return intersection.volume / union.volume



def eval(dataloader, models, data_modality, sequence_length, shots, save=False):
    point_face_losses = {}
    chamfer_losses_roi = {}
    ratios = {}
    jaccard_dist = {}
    jaccard_index = {}
    for shot in shots:
        point_face_losses[shot] = []
        chamfer_losses_roi[shot] = []
        ratios[shot] = []
        jaccard_dist[shot] = []
        jaccard_index[shot] = []
    total_point_face_loss, total_roi_chamfer_loss, total_j_pred_gt = 0, 0, 0
    total_ratio, total_dj_temp_gt, total_count = 0, 0, 0
    with torch.no_grad():
        for i, item in enumerate(dataloader):
            # get data
            (target_meshes, template_vertices, template_faces, input_data, forces, 
             centers, scales, frames, num_points, names, transformations) = item
            batch_size = len(frames)

            # get features
            features = get_features(models, input_data, forces, 
                                    batch_size, data_modality, sequence_length)

            # calculate loss
            point_face_loss, chamfer_loss_roi, j_pred_gt, ratio, dj_temp_gt = loss(models[0],
                        template_vertices,
                        template_faces,
                        target_meshes, features,
                        forces, num_points,
                        transformations,
                        batch_size,
                        sequence_length, scales,
                        centers)

            total_count += batch_size
            total_point_face_loss += chamfer_loss_roi.item() * batch_size
            total_roi_chamfer_loss += chamfer_loss_roi.item() * batch_size
            total_j_pred_gt += j_pred_gt * batch_size if j_pred_gt is not None else 0
            total_ratio += ratio * batch_size
            total_dj_temp_gt += dj_temp_gt * batch_size  if dj_temp_gt is not None else 0

            # losses for logging
            name = names[0].split("_")[0]
            for shot in shots:
                if shot == name:
                    point_face_losses[shot].append(point_face_loss.item())
                    chamfer_losses_roi[shot].append(chamfer_loss_roi.item())
                    ratios[shot].append(ratio)
                    jaccard_dist[shot].append(dj_temp_gt)
                    jaccard_index[shot].append(j_pred_gt)
    
    print(f"Average Point-face-distance loss: {total_point_face_loss / total_count:.4f}")
    print(f"Average Chamfer loss ROI: {total_roi_chamfer_loss / total_count:.4f}")
    print(f"Average Jaccard index: {total_j_pred_gt / total_count:.4f}")
    print(f"Average Jaccard distance: {total_dj_temp_gt / total_count:.4f}")
    print(f"Average Ratio: {total_ratio / total_count:.4f}")

    if save:
        log_save_path = f"./logs/{data_modality}"
        os.makedirs(log_save_path, exist_ok=True)
        with open(f"{log_save_path}/ratio_img.json", "w") as file:
            json.dump(ratios, file)
        with open(f"{log_save_path}/jaccard_dist_img.json", "w") as file:
            json.dump(jaccard_dist, file)
        with open(f"{log_save_path}/jaccard_index_img.json", "w") as file:
            json.dump(jaccard_index, file)


def plot_loss(x_loss, y_loss, shots):
    names = ["Foam Dice", "Sponge", "Plush Moon", "Plush Octopus", "Foam Half Sphere"]
    plt.style.use("../data/tamp.mplstyle")
    fig, ax = plt.subplots()

    # Plot scatter points
    colors = ['red', 'blue', 'green', 'black', "orange"]
    for i, shot in enumerate(shots):
        ax.scatter(x_loss[shot], y_loss[shot], color=colors[i], s=40,
                   label=names[i])  # size=8 in plotly corresponds to s=80 in matplotlib

    out = {"x": x_loss, "y": y_loss}
    with open("/home/pokingrobot/paper_data/ratio.json", "w") as file:
        json.dump(out, file)


    # Set title and labels
    x_title = r"$\mathrm{d_{J}(M_{T}, M_{GT}})$"
    y_title = "Relative Point-to-Surface Distance"
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_yscale('log')

    # Customize grid, ticks, and background
    ax.grid(True, color='lightgrey')
    ax.set_facecolor('white')


    # Customize legend
    ax.legend(loc='upper right',
              ncol=1, fancybox=True, shadow=True)

    # Show plot
    plt.savefig("../data/jaccard.pdf", format='pdf', dpi=600, bbox_inches='tight')
    plt.show()

def main(args, config):
    model_path = config['model_path']
    shots = ["MemoryFoam", "Volleyball", "HalfSphere", "Bunny", "Pyramid", 
             "Dice", "Moon", "Octopus", "PlushDice", "Turtle", "Fjadrar", 
             "Cylinder", "Beanbag", "Heart", "Foam", "ToiletPaper", "Sponge", "Pizza"]
    takes = [2, 4, 3, 1, 6, 3, 1, 6, 8, 3, 8, 7, 6, 14, 1, 1, 10, 13]
    paths = []
    for take, shot in zip(takes, shots):
        data_path = f"{config['data_path']}/{shot}/{shot}_T{take}"
        paths.append(data_path)

    models = load_models(model_path)
    dataloader = load_data(paths, args.modality, config)
    eval(dataloader, models, args.modality, config['sequence_length'], shots, save=args.save_loss)


if __name__ == "__main__":
    modalities = ["images", "pointclouds", "robot_data", "img_robot_data"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="img_robot_data", choices=modalities)
    parser.add_argument("--save_loss", action="store_true", help="Save loss values to file")
    args = parser.parse_args()

    # read testing config file
    config_file = f"config/testing.yml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(args, config)

