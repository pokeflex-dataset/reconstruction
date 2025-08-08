from debugpy.launcher.debuggee import process
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import trimesh
from pytorch3d.structures import Meshes
import time
import json
import random
import pytorch3d.transforms as T

class MeshDataset(Dataset):
    def __init__(self, root_dir, images, pointclouds, cameras, pcd_type, sequence_length):
        self.data = {}
        self.name = root_dir.split("/")[-1]
        self.pcd_root = f"{root_dir}/pointclouds"
        self.sequence_length = sequence_length
        self.frames = []
        self.max_force = 100
        self.pcd_type = pcd_type
        self.image_root = f"{root_dir}/images"
        self.camera_names = [name for name in cameras]
        self.sequence_length = sequence_length
        self.pointclouds = pointclouds
        self.images = images
        self.transformation = transforms.Compose([
            transforms.Resize(256),  # Resize the image to 256x256
            transforms.CenterCrop(224),  # Crop a 224x224 patch from the center
            #transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Normalize with ImageNet's mean and std
        ])


        # template mesh
        template_dir = os.path.join(root_dir, "template_mesh")
        for filename in os.listdir(template_dir):
            if filename.startswith("mesh-f"):
                template = trimesh.load(os.path.join(root_dir, "template_mesh", filename), process = False)
                break
        self.data["template_verts"], self.data["template_faces"], self.center, self.scale = self.transform_template(template)

        # target meshes
        self.data["target_verts"], self.data["target_faces"] = [], []
        target_mesh_path = os.path.join(root_dir, "triangle_meshes")
        for mesh_file in sorted(os.listdir(target_mesh_path)):
            if mesh_file.endswith('.ply') or mesh_file.endswith('.obj'):
                target_mesh = trimesh.load(os.path.join(target_mesh_path, mesh_file), process = False)
                verts, faces = self.transform_target(target_mesh, self.center, self.scale)
                self.data["target_verts"].append(verts)
                self.data["target_faces"].append(faces)
                self.frames.append(mesh_file[6:11])


        # robot_data
        with open(os.path.join(root_dir, "robot_data.json"), 'r') as f:
            robot_data = json.load(f)

        # robot data and images
        if self.pointclouds:
            self.data["pointclouds"] = self.load_pointclouds()
        if self.images:
            self.data["images"] = self.load_images()
        self.data["forces"] = self.get_force(robot_data)


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        # forces
        forces = []
        for i in range(self.sequence_length, 0, -1):
            current_frame = int(frame) - i
            forces.append(self.data['forces'][current_frame])

        # pointclouds
        if self.pointclouds:
            pointclouds = []
            if self.pcd_type == "synthetic":
                pointclouds.append(self.data["pointclouds"][idx])
            else:
                for i in range(self.sequence_length, 0, -1):
                    current_frame = int(frame) - i
                    if current_frame < 0:
                        current_frame = 0
                    pointclouds.append(self.data["pointclouds"][current_frame])
        else:
            pointclouds = None

        if self.images:
            images = []
            camera = random.randint(0, len(self.camera_names) - 1)
            for i in range(self.sequence_length, 0, -1):
                current_frame = int(frame) - i
                if current_frame < 0:
                    current_frame = 0
                images.append(self.data["images"][camera][current_frame])
        else:
            images = None

        return {'target_verts': self.data["target_verts"][idx],
                'target_faces': self.data["target_faces"][idx],
                'template_verts': self.data["template_verts"],
                'template_faces': self.data["template_faces"],
                "images": images,
                "pointclouds": pointclouds,
                "forces": forces,
                'frame': self.frames[idx],
                'centers': self.center,
                'scales': self.scale,
                "name": self.name}

    def transform_template(self, mesh):
        # load
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.float32)

        # transform
        center = verts.mean(0)
        verts = verts - center
        scale = torch.sqrt((verts ** 2).sum(1)).max()
        verts = verts / scale

        return verts, faces, center, scale

    def transform_target(self, mesh, center, scale):
        # load
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.float32)

        # transform
        verts = verts - center
        verts = verts / scale

        return verts, faces

    def load_pointclouds(self):
        pointclouds = []
        if self.pcd_type == "kinect":
            pcd_folder = "2"
        else:
            pcd_folder = self.pcd_type
        for pcd_file in sorted(os.listdir(os.path.join(self.pcd_root, pcd_folder))):
            pcd_path = os.path.join(self.pcd_root, pcd_folder, pcd_file)
            pcd = trimesh.load(pcd_path)
            verts = torch.tensor(pcd.vertices, dtype=torch.float32)
            verts = (verts - self.center)/self.scale
            pointclouds.append(verts)

        return pointclouds

    def get_force(self, robot_data):
        forces = []
        for item in robot_data:
            current = torch.zeros(6, dtype=torch.float32)
            current[:3] = torch.tensor(item['forces'][:3], dtype=torch.float32)/self.max_force
            current[3:] = (torch.tensor(1000*np.array(item['T_WT'])[:3, 3], dtype=torch.float32) - self.center)/self.scale
            forces.append(current)

        return forces

    def load_images(self):
        offset = -20
        noise_std = 10
        images = []
        for camera in self.camera_names:
            images_current_cam = []
            for img_file in sorted(os.listdir(os.path.join(self.image_root, camera))):
                image_path = os.path.join(self.image_root, camera, img_file)
                image = Image.open(image_path)

                # image_np = np.array(image).astype(np.float32)
                # image_np += offset
                # noise = np.random.normal(0, noise_std, image_np.shape[:2])  # Same noise across all channels
                # noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                # image_np += noise
                # image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                # image = Image.fromarray(image_np)

                image = self.transformation(image)
                images_current_cam.append(image)
            images.append(images_current_cam)

        return images


def collate_fn(data, device):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    rotate = False
    # initialize tensors
    maxPointsMesh = 10000
    maxPointsPCD = 5000
    features_verts_src = torch.zeros((len(data), maxPointsMesh, 3), dtype=torch.float32)
    features_verts_trg = torch.zeros((len(data), maxPointsMesh, 3), dtype=torch.float32)
    sequence_length = len(data[0]['forces'])
    batch_size = len(data)
    if rotate:
        transformation = T.random_rotations(batch_size).to(device)
    else:
        transformation = torch.eye(3)
        transformation = transformation.unsqueeze(0).repeat(batch_size, 1, 1)
    transformation = T.Rotate(transformation).to(device)

    time1 = time.time()
    num_vertices_src = []
    num_vertices_trg = []
    # adjust size of tensors
    for i in range(batch_size):
        verts_src = data[i]['template_verts']
        verts_trg = data[i]['target_verts']
        num_vertices_src.append(verts_src.shape[0])
        num_vertices_trg.append(verts_trg.shape[0])
        features_verts_src[i] = torch.cat((verts_src, torch.zeros((maxPointsMesh - verts_src.shape[0], 3))), dim=0)
        features_verts_trg[i] = torch.cat((verts_trg, torch.zeros((maxPointsMesh - verts_trg.shape[0], 3))), dim=0)

    features_verts_src = transformation.transform_points(features_verts_src.to(device))
    features_verts_trg = transformation.transform_points(features_verts_trg.to(device))
    time2 = time.time()
    #print("Time to collate vertices: ", time2 - time1)


    # pointclouds and images
    if data[0]['images'] is not None:
        img_bool = True
        pcd_bool = False
        input_data = torch.zeros((sequence_length * batch_size, 3, 224, 224), dtype=torch.float32).to(device)

    elif data[0]['pointclouds'] is not None:
        pcd_bool = True
        img_bool = False
        pcd_length = len(data[0]['pointclouds'])
        input_data = torch.zeros((pcd_length*batch_size, maxPointsPCD, 3), dtype=torch.float32).to(device)
    else:
        pcd_bool = False
        img_bool = False
        input_data = None


    forces = torch.zeros((sequence_length*batch_size, 6), dtype=torch.float32).to(device)
    for i in range(sequence_length):
        for j in range(batch_size):
            forces[i*batch_size+j] = data[j]['forces'][i]
            forces[i*batch_size+j, :3] = transformation[j].transform_points(forces[i*batch_size+j, :3].unsqueeze(0)).squeeze(0)
            forces[i*batch_size+j, 3:] = transformation[j].transform_points(forces[i*batch_size+j, 3:].unsqueeze(0)).squeeze(0)
            if pcd_bool:
                if i < pcd_length:
                    points = data[j]['pointclouds'][i]
                    input_data[i*batch_size+j] = torch.cat((points, torch.zeros((maxPointsPCD - points.shape[0], 3))), dim=0)
                    input_data[i*batch_size+j] = transformation[j].transform_points(input_data[i*batch_size+j])

            if img_bool:
                input_data[i*batch_size+j] = data[j]['images'][i]

    time3 = time.time()
    #print("Time to collate pointclouds and images: ", time3 - time2)

    frames, centers, scales, names, target_faces, template_faces = [], [], [], [], [], []
    target_verts = []
    for j, item in enumerate(data):
        target_verts.append(features_verts_trg[j][:num_vertices_trg[j]])
        target_faces.append(item['target_faces'].to(device))
        template_faces.append(item['template_faces'].to(device))
        frames.append(item['frame'])
        centers.append(item['centers'])
        scales.append(item['scales'])
        names.append(item['name'])

    # target meshes
    target_meshes = Meshes(verts=target_verts, faces=target_faces)

    #print("Rest: ", time.time() - time3)

    return target_meshes, features_verts_src, template_faces, input_data, forces, centers, scales, frames, num_vertices_src, names, transformation







