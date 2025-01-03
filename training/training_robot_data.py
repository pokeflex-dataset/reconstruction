import os
import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from models.nvp_cadex import NVP_v2_5_frame
from models.feature_extraction import SelfAttention, ForceFeatures
from pytorch3d.loss import (
    chamfer_distance, point_mesh_face_distance
)
from pytorch3d.io import save_obj
from dataset.dataset_robot_data import RobotDataDataset, collate_fn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class Training:
    def __init__(self, device, config):
        self.device = device

        self.attention_output_size = config['attention_feat_output_size']
        self.force_output_size = config['force_feat_output_size']
        self.sequence_length = config['sequence_length']

        # models
        self.force_encoder = ForceFeatures(self.force_output_size).to(self.device)
        if self.sequence_length > 1:
            self.attention_model = SelfAttention(self.force_output_size, self.attention_output_size,
                                                 self.device).to(self.device)
            decoder_input_size = self.attention_output_size
        else:
            self.attention_model = None
            decoder_input_size = self.force_output_size

        self.decoder = self.get_homeomorphism_model(decoder_input_size).to(self.device)

        # optimizer and weights
        params = []
        models = [self.decoder, self.attention_model, self.force_encoder]
        params += [param for model in models if model is not None for param in model.parameters()]

        self.optimizer = optim.Adam(params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10000, eta_min=1e-7)
        self.pfd_weight = config['pfd_weight']
        self.roi_weight = config['roi_weight']
        self.output_dir = config['output_path']




    def get_homeomorphism_model(self, input_size):
        n_layers = 6
        # dimension of the code

        hidden_size = [128, 64, 32, 32, 32]
        # the dimension of the coordinates to be projected onto
        proj_dims = 128
        code_proj_hidden_size = [128, 128, 128]
        proj_type = 'simple'
        block_normalize = True
        normalization = False

        homeomorphism_decoder = NVP_v2_5_frame(n_layers=n_layers, feature_dims=input_size,
                                               hidden_size=hidden_size,
                                               proj_dims=proj_dims, code_proj_hidden_size=code_proj_hidden_size,
                                               proj_type=proj_type,
                                               block_normalize=block_normalize, normalization=normalization)
        return homeomorphism_decoder

    def load_data(self, data_paths, config):
        train_datasets = []
        valid_datasets = []
        for data_path in data_paths:
            num_valid_paths = 1
            for i, directory in enumerate(os.listdir(data_path)):
                current_set = RobotDataDataset(os.path.join(data_path, directory), self.sequence_length)
                if i >= len(os.listdir(data_path)) - num_valid_paths:
                    valid_datasets.append(current_set)
                else:
                    train_datasets.append(current_set)

                print("Dataset ", i + 1, "loaded out of ", len(os.listdir(data_path)))

        total_training_sets = ConcatDataset(train_datasets)
        total_valid_sets = ConcatDataset(valid_datasets)

        # split the dataset into training and validation
        train_dataloader = DataLoader(total_training_sets, batch_size=config['batch_size'], shuffle=True,
                                      collate_fn=lambda b, device=self.device: collate_fn(b, device), drop_last=True)
        valid_dataloader = DataLoader(total_valid_sets, batch_size=config['batch_size'], shuffle=True,
                                      collate_fn=lambda b, device=self.device: collate_fn(b, device), drop_last=True)

        return train_dataloader, valid_dataloader

    def get_roi_meshes(self, meshes, target_position, distance_threshold):
        face_indices_array = []
        for i, mesh in enumerate(meshes):
            verts = mesh.verts_packed()
            pos = target_position[i].unsqueeze(0).to(self.device)
            distances_squared = torch.sum((verts - pos) ** 2, dim=1)

            # get vertices
            vertices_mask = distances_squared < distance_threshold ** 2
            bounding_box_mask = verts[:, 1] > (torch.min(verts[:, 1]) + 0.2)
            combined_mask = vertices_mask & bounding_box_mask
            vertices_indices = combined_mask.nonzero(as_tuple=True)[0]

            # get faces
            faces = mesh.faces_packed()
            faces_mask = torch.zeros(faces.shape[0], dtype=torch.bool, device=self.device)
            for index in vertices_indices:
                faces_mask |= (faces == index).any(dim=1)

            face_indices = faces_mask.nonzero(as_tuple=True)[0]
            face_indices_array.append([face_indices])

        selected_meshes = meshes.submeshes(face_indices_array)

        return selected_meshes

    def loss(self, template_vertices, template_faces, target_meshes, features, forces, num_points, batch_size, valid):
        # predict deformation
        coordinates = self.decoder.forward(features, template_vertices)
        coordinates = coordinates.reshape(batch_size, 11000, 3)

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

        # roi loss
        sampled_points_roi = 2500
        ee_pos = [force[3:] for force in forces.view(self.sequence_length, batch_size, 6)[-1]]
        predicted_roi_meshes = self.get_roi_meshes(predicted_meshes, ee_pos, 0.4)
        target_roi_meshes = self.get_roi_meshes(target_meshes, ee_pos, 0.4)
        try:
            predicted_roi_sampled, predicted_roi_normals = sample_points_from_meshes(predicted_roi_meshes,
                                                                                     sampled_points_roi,
                                                                                     return_normals=True)
            chamfer_loss_roi, _ = chamfer_distance(predicted_roi_sampled, target_sampled,
                                                   x_normals=predicted_roi_normals,
                                                   y_normals=normals_target, single_directional=True)

        except ValueError:
            chamfer_loss_roi = torch.tensor(0.05, device=self.device)

        if valid:
            return point_face_loss, chamfer_loss_roi, predicted_meshes, predicted_roi_meshes, target_roi_meshes, ee_pos
        else:
            return point_face_loss, chamfer_loss_roi

    def train(self, dataloader, epoch):
        models = [self.decoder, self.attention_model, self.force_encoder]
        for model in models:
            if model is not None:
                model.train()

        total_roi_chamfer_loss, total_loss_epoch, total_point_face_loss = 0, 0, 0
        for i, item in enumerate(dataloader):
            self.optimizer.zero_grad()
            # get data
            target_meshes, template_vertices, template_faces, forces, centers, scales, frames, num_points, names = item
            batch_size = len(frames)

            # get features
            features = self.force_encoder.forward(forces)
            if self.sequence_length > 1:
                features = features.view(self.sequence_length, batch_size, self.force_output_size)
                features = self.attention_model.forward(features)

            # calculate loss
            point_face_loss, chamfer_loss_roi = self.loss(template_vertices, template_faces, target_meshes, features,
                                                          forces, num_points, batch_size, False)

            # backward pass
            total_loss = point_face_loss * self.pfd_weight + chamfer_loss_roi * self.roi_weight
            total_loss.backward()

            # losses for logging
            total_point_face_loss += point_face_loss
            total_roi_chamfer_loss += chamfer_loss_roi
            total_loss_epoch += total_loss

            self.optimizer.step()
            self.scheduler.step()

    def validate(self, dataloader, epoch):
        models = [self.decoder, self.attention_model, self.force_encoder]
        for model in models:
            if model is not None:
                model.eval()

        total_point_face_loss, total_roi_chamfer_loss = 0, 0
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                # get data
                target_meshes, template_vertices, template_faces, forces, centers, scales, frames, num_points, names = item
                batch_size = len(frames)

                # get features
                features = self.force_encoder.forward(forces)
                if self.sequence_length > 1:
                    features = features.view(self.sequence_length, batch_size, self.force_output_size)
                    features = self.attention_model.forward(features)

                # calculate loss
                point_face_loss, chamfer_loss_roi, predicted_meshes, predicted_roi_meshes, target_roi_meshes, ee_pos = self.loss(
                    template_vertices, template_faces,
                    target_meshes, features, forces,
                    num_points, batch_size,
                    True)

                total_roi_chamfer_loss += chamfer_loss_roi
                total_point_face_loss += point_face_loss

                if epoch % 50 == 0:
                    transformed_meshes = self.transform_meshes(predicted_meshes, centers, scales)
                    self.save_meshes(transformed_meshes, names, frames, validation=True)
                    self.save_models()

    def transform_meshes(self, meshes, centers, scales):
        transformed_verts = []
        for i, mesh in enumerate(meshes):
            verts = mesh.verts_packed()
            verts = verts * scales[i].to(self.device)
            verts = verts + centers[i].to(self.device)
            transformed_verts.append(verts.float())

        transformed_meshes = Meshes(verts=transformed_verts, faces=meshes.faces_list())

        return transformed_meshes

    def save_meshes(self, meshes, names, frames, validation):
        with torch.no_grad():
            for i, mesh in enumerate(meshes):
                if validation:
                    data_path = f"{self.output_dir}/validation/{names[i]}"
                else:
                    data_path = f"{self.output_dir}/train/{names[i]}"
                os.makedirs(data_path, exist_ok=True)
                file = os.path.join(data_path, f"mesh-f{frames[i]}.obj")
                save_obj(file, mesh.verts_packed(), mesh.faces_packed())

    def save_models(self):
        path = f"{self.output_dir}/models"
        os.makedirs(path, exist_ok=True)
        models = [self.decoder, self.attention_model, self.force_encoder]
        names = ["decoder", "attention_model", "force_encoder"]
        for name, model in zip(names, models):
            if model is not None:
                torch.save(model, f"{path}/{name}.pth")


def robot_data_main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_paths = []
    for object in config['objects']:
        data_paths.append(f"{config['data_path']}/{object}")
    epochs = config['epochs']
    epoch_start = 1

    session = Training(device, config)
    # get data
    train_loader, valid_loader = session.load_data(data_paths, config)
    print("Training dataset size:", len(train_loader), "Validation set size:", len(valid_loader))

    for epoch in range(epoch_start, epochs):
        print("Epoch: ", epoch)
        session.train(train_loader, epoch)
        session.validate(valid_loader, epoch)


