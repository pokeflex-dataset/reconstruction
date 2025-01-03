from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import trimesh
from pytorch3d.structures import Meshes
import json
import random

class ImagesDataset(Dataset):
    def __init__(self, root_dir, cameras, sequence_length):
        self.data = {}
        self.name = root_dir.split("/")[-1]
        self.frames = []
        self.max_force = 100
        self.image_root = f"{root_dir}/images"
        self.camera_names = [name for name in cameras]
        self.sequence_length = sequence_length
        self.transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        # template mesh
        template = trimesh.load(os.path.join(root_dir, "template_mesh", "template_mesh.obj"), process = False)
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
        self.data["forces"] = self.get_force(robot_data)

        # images
        self.data["images"] = self.load_images()


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]

        # forces
        forces = []
        for i in range(self.sequence_length, 0, -1):
            current_frame = int(frame) - i
            forces.append(self.data['forces'][current_frame])


        # images
        images = []
        camera = random.randint(0, len(self.camera_names) - 1)
        for i in range(self.sequence_length, 0, -1):
            current_frame = int(frame) - i
            if current_frame < 0:
                current_frame = 0
            images.append(self.data["images"][camera][current_frame])

        return {'target_verts': self.data["target_verts"][idx],
                'target_faces': self.data["target_faces"][idx],
                'template_verts': self.data["template_verts"],
                'template_faces': self.data["template_faces"],
                "images": images,
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


    def get_force(self, robot_data):
        forces = []
        for item in robot_data:
            current = torch.zeros(6, dtype=torch.float32)
            current[:3] = torch.tensor(item['forces'][:3], dtype=torch.float32)/self.max_force
            current[3:] = (torch.tensor(1000*np.array(item['T_WT'])[:3, 3], dtype=torch.float32) - self.center)/self.scale
            forces.append(current)

        return forces

    def load_images(self):
        images = []
        for camera in self.camera_names:
            images_current_cam = []
            for img_file in sorted(os.listdir(os.path.join(self.image_root, camera))):
                image_path = os.path.join(self.image_root, camera, img_file)
                image = Image.open(image_path)
                image = self.transformation(image)
                images_current_cam.append(image)
            images.append(images_current_cam)

        return images


def collate_fn(data, device):
    # initialize tensors
    maxPointsMesh = 11000
    features_verts_src = torch.zeros((len(data), maxPointsMesh, 3), dtype=torch.float32).to(device)
    features_verts_trg = torch.zeros((len(data), maxPointsMesh, 3), dtype=torch.float32).to(device)
    sequence_length = len(data[0]['forces'])
    batch_size = len(data)

    num_vertices_src = []
    num_vertices_trg = []
    # fill tensors
    for i in range(batch_size):
        verts_src = data[i]['template_verts']
        verts_trg = data[i]['target_verts']
        num_vertices_src.append(verts_src.shape[0])
        num_vertices_trg.append(verts_trg.shape[0])
        features_verts_src[i] = torch.cat((verts_src, torch.zeros((maxPointsMesh - verts_src.shape[0], 3))), dim=0).to(device)
        features_verts_trg[i] = torch.cat((verts_trg, torch.zeros((maxPointsMesh - verts_trg.shape[0], 3))), dim=0).to(device)


    # images and forces
    image_data = torch.zeros((sequence_length * batch_size, 3, 224, 224), dtype=torch.float32).to(device)
    forces = torch.zeros((sequence_length*batch_size, 6), dtype=torch.float32).to(device)
    for i in range(sequence_length):
        for j in range(batch_size):
            forces[i*batch_size+j] = data[j]['forces'][i]
            image_data[i*batch_size+j] = data[j]['images'][i]

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

    return target_meshes, features_verts_src, template_faces, image_data, forces, centers, scales, frames, num_vertices_src, names







