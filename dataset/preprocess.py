import numpy as np
import json
import os
import shutil
import trimesh
import open3d as o3d
import cv2
import yaml
import pyvista as pv


def load_depth_image(file_path, camera_matrix, dist_coeffs):
    # Load the depth image
    depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return depth_image


def create_point_cloud_from_depth(depth_image, intrinsics, extrinsics, depth_trunc, depth_scale):
    # Convert depth image to Open3D image format
    depth_o3d = o3d.geometry.Image(depth_image)

    # Create intrinsic matrix for the camera (adjust fx, fy, cx, cy accordingly)
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width=depth_image.shape[1],
                        height=depth_image.shape[0],
                        fx=intrinsics[0, 0],
                        fy=intrinsics[1, 1],
                        cx=intrinsics[0, 2],
                        cy=intrinsics[1, 2])

    # Convert depth image to point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intr, extrinsic=extrinsics,
                                                          depth_scale=depth_scale,
                                                          depth_trunc=depth_trunc)

    return pcd


def crop_image(image, intrinsics, extrinsics, mesh_center, crop_size):
    new_width, new_height = crop_size
    rows, cols = image.shape[:2]

    projection_matrix = intrinsics @ extrinsics
    pixel_coordinates = projection_matrix @ np.array([mesh_center[0], mesh_center[1], mesh_center[2], 1])
    pixel_coordinates = pixel_coordinates[:2] / pixel_coordinates[2]

    # Calculate the top left corner of the cropping rectangle in the second image
    start_x = max(0, int(pixel_coordinates[0]) - new_width // 2)
    start_y = max(0, int(pixel_coordinates[1]) - new_height // 2)

    if start_x + new_width > cols:
        start_x = cols - new_width
    if start_y + new_height > rows:
        start_y = rows - new_height

    # Crop the image
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]

    return cropped_image


def get_deformation_frames(robot_data, threshold):
    frames = []
    for item in robot_data:
        if item['forces'][1] > threshold:
            frames.append(int(item['frame']))

    return frames


def get_volucam_images(take_path, output_path, mesh_center, image_size):
    camera_dir = f"{take_path}/volucam"
    for camera in os.listdir(camera_dir):
        os.makedirs(f"{output_path}/images/volucam{camera}", exist_ok=True)
        camera_parameters = json.load(open(f"{camera_dir}/{camera}/camera_parameters.json", "r"))
        intriniscs = np.array(camera_parameters['intrinsics'])
        extrinsics = np.array(camera_parameters['extrinsics'])[0:3, :]
        for filename in os.listdir(f"{camera_dir}/{camera}/color"):
            image = cv2.imread(f"{camera_dir}/{camera}/color/{filename}")
            cropped_image = crop_image(image, intriniscs, extrinsics, mesh_center, image_size)
            cv2.imwrite(f"{output_path}/images/volucam{camera}/{filename}", cropped_image)


def get_kinect_images(take_path, output_path, mesh_center, image_size):
    camera_dir = f"{take_path}/kinect"
    for camera in os.listdir(camera_dir):
        os.makedirs(f"{output_path}/images/kinect{camera}", exist_ok=True)
        camera_parameters = json.load(open(f"{camera_dir}/{camera}/camera_parameters.json", "r"))
        intriniscs = np.array(camera_parameters['color_intrinsics'])
        extrinsics = np.array(camera_parameters['color_extrinsics'])[0:3, :]
        for filename in os.listdir(f"{camera_dir}/{camera}/color"):
            image = cv2.imread(f"{camera_dir}/{camera}/color/{filename}")
            cropped_image = crop_image(image, intriniscs, extrinsics, mesh_center, image_size)
            cv2.imwrite(f"{output_path}/images/kinect{camera}/{filename}", cropped_image)


def get_realsense_images(source, dest):
    for camera in ["0", "1"]:
        camera_dir = f"{source}/realsense/{camera}/color"
        os.makedirs(f"{dest}/images/realsense{camera}", exist_ok=True)
        for filename in os.listdir(camera_dir):
            shutil.copy(f"{camera_dir}/{filename}", f"{dest}/images/realsense{camera}/{filename}")

def get_camera_parameters(source_path, output_path, mesh_center, volucam_crop_size, kinect_crop_size):
    out = {}
    for volucam in os.listdir(f"{source_path}/volucam"):
        out[f"volucam{volucam}"] = {}
        size_x, size_y = volucam_crop_size
        parameters = json.load(open(f"{source_path}/volucam/{volucam}/camera_parameters.json", "r"))
        camera_matrix = np.array(parameters['intrinsics'])
        extrinsics = np.array(parameters['extrinsics'])[0:3, :]
        projection_matrix = camera_matrix @ extrinsics
        pixel_coordinates = projection_matrix @ np.array([mesh_center[0], mesh_center[1], mesh_center[2], 1])
        pixel_coordinates = pixel_coordinates[:2] / pixel_coordinates[2]
        center_x, center_y = pixel_coordinates
        camera_matrix[0, 2] -= (center_x - int(size_x / 2))
        camera_matrix[1, 2] -= (center_y - int(size_y / 2))
        out[f"volucam{volucam}"]["intrinsics"] = camera_matrix.tolist()
        out[f"volucam{volucam}"]["extrinsics"] = np.vstack([extrinsics, np.array([0, 0, 0, 1])]).tolist()

    for kinect in os.listdir(f"{source_path}/kinect"):
        out[f"kinect{kinect}"] = {}
        size_x, size_y = volucam_crop_size
        parameters = json.load(open(f"{source_path}/kinect/{kinect}/camera_parameters.json", "r"))
        camera_matrix = np.array(parameters['color_intrinsics'])
        extrinsics = np.array(parameters['color_extrinsics'])[0:3, :]
        projection_matrix = camera_matrix @ extrinsics
        pixel_coordinates = projection_matrix @ np.array([mesh_center[0], mesh_center[1], mesh_center[2], 1])
        pixel_coordinates = pixel_coordinates[:2] / pixel_coordinates[2]
        center_x, center_y = pixel_coordinates
        camera_matrix[0, 2] -= (center_x - int(size_x / 2))
        camera_matrix[1, 2] -= (center_y - int(size_y / 2))
        out[f"kinect{kinect}"]["intrinsics"] = camera_matrix.tolist()
        out[f"kinect{kinect}"]["extrinsics"] = np.vstack([extrinsics, np.array([0, 0, 0, 1])]).tolist()

    for realsense in os.listdir(f"{source_path}/realsense"):
        out[f"realsense{realsense}"] = {}
        camera_parameters = json.load(open(f"{source_path}/realsense/{realsense}/camera_parameters.json", "r"))
        color_intrinsics = np.array(camera_parameters['color_intrinsics'])
        extrinsics = camera_parameters['extrinsics']
        out[f"realsense{realsense}"]["intrinsics"] = color_intrinsics.tolist()
        out[f"realsense{realsense}"]["extrinsics"] = extrinsics

    with open(f"{output_path}/images/camera_parameters.json", "w") as file:
        json.dump(out, file)



def get_kinect_point_clouds(final_dir, output_dir, mesh, num_points):
    # get bounding box
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mesh.vertices / 1000))
    bounding_box = target.get_axis_aligned_bounding_box()
    min_bounds_old = bounding_box.get_min_bound()
    bounding_box = bounding_box.scale(1.3, bounding_box.get_center())
    min_bounds = bounding_box.get_min_bound()
    min_bounds[1] = min_bounds_old[1] + 0.01
    max_bounds = bounding_box.get_max_bound()
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds, max_bound=max_bounds)

    verts = []
    for camera in ["0", "1"]:
        # camera parameters
        camera_parameters = json.load(open(f"{final_dir}/kinect/{camera}/camera_parameters.json", "r"))
        depth_intrinsics = np.array(camera_parameters['depth_intrinsics'])
        depth_dist_coeffs = np.array(camera_parameters['depth_dist_coeffs'])
        depth_extrinsics = np.array(camera_parameters['depth_extrinsics'])

        image_dir = f"{final_dir}/kinect/{camera}/depth"

        # get point clouds
        for filename in sorted(os.listdir(image_dir)):
            depth_image = load_depth_image(f"{image_dir}/{filename}", depth_intrinsics, depth_dist_coeffs)
            pcd = create_point_cloud_from_depth(depth_image, depth_intrinsics, depth_extrinsics, 2.5, 1000)
            pcd = pcd.crop(bounding_box)
            verts.append(np.asarray(pcd.points))

    num_frames = int(len(verts) / 2)
    for i in range(num_frames):
        fused = np.vstack((verts[i], verts[i + num_frames]))
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fused))

        # simplify point cloud
        pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd.remove_radius_outlier(nb_points=20, radius=0.05)
        voxel_size = 0.001
        while (np.asarray(pcd.points).shape[0] > num_points):
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            voxel_size += 0.001
        point_cloud_dir = f"{output_dir}/pointclouds/kinect"
        os.makedirs(point_cloud_dir, exist_ok=True)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000))
        o3d.io.write_point_cloud(f"{point_cloud_dir}/{i + 1:05d}.ply", pcd)

def get_synthetic_point_clouds(source_path, output_dir, num_points_dense, num_points_sparse):
    os.makedirs(f"{output_dir}/pointclouds/synthetic_dense", exist_ok=True)
    os.makedirs(f"{output_dir}/pointclouds/synthetic_sparse", exist_ok=True)
    for filename in os.listdir(f"{source_path}/meshes"):
        if not filename.endswith(".obj"):
            continue
        mesh = o3d.io.read_triangle_mesh(f"{source_path}/meshes/{filename}")
        pcd_dense = mesh.sample_points_uniformly(num_points_dense)
        pcd_sparse = mesh.sample_points_uniformly(num_points_sparse)
        o3d.io.write_point_cloud(f"{output_dir}/pointclouds/synthetic_dense/{filename[-9:-4]}.ply", pcd_dense)
        o3d.io.write_point_cloud(f"{output_dir}/pointclouds/synthetic_sparse/{filename[-9:-4]}.ply", pcd_sparse)


def get_meshes(source_path, output_path, frames, num_faces_coarse):
    os.makedirs(f"{output_path}/template_mesh", exist_ok=True)
    os.makedirs(f"{output_path}/triangle_meshes", exist_ok=True)
    if 1 in frames:
        last_frame = frames[0]
        template_frame = None
        for frame in sorted(frames[1:]):
            if frame - last_frame > 5:
                template_frame = int((frame + last_frame) / 2)
                print("template frame", template_frame)
                break
            last_frame = frame
        if template_frame is not None:
            template_mesh_path = f"{source_path}/mesh-f{template_frame:05d}.obj"
        else:
            raise ValueError("No template frame found")
    else:
        template_mesh_path = f"{source_path}/meshes/mesh-f00001.obj"

    template_mesh = pv.read(template_mesh_path)
    num_faces = template_mesh.n_cells
    factor = num_faces_coarse / num_faces
    if factor < 1:
        template_mesh.decimate_pro(
            reduction=1-factor,
            preserve_topology=True,
            inplace=True
        )
    # decimated_mesh.clean(inplace=True)
    pv.save_meshio(f"{output_path}/template_mesh/template_mesh.obj", template_mesh)

    for frame in frames:
        mesh_path = f"{source_path}/meshes/mesh-f{frame:05d}.obj"
        mesh = pv.read(mesh_path)
        num_faces = mesh.n_cells
        factor = num_faces_coarse / num_faces
        if factor < 1:
            mesh.decimate_pro(
                reduction=1-factor,
                preserve_topology=True,
                inplace=True
            )

        #decimated_mesh.clean(inplace=True)
        pv.save_meshio(f"{output_path}/triangle_meshes/mesh-f{frame:05d}.obj", mesh)



def main(config):
    data_path = config["data_path"]
    output_dir = config["output_path"]


    #
    for obj in config["objects"]:
        for take in os.listdir(f"{data_path}/{obj}"):
            output_path = f"{output_dir}/{obj}/{take}"
            source_path = f"{data_path}/{obj}/{take}"
            os.makedirs(output_path, exist_ok=True)

            # load robot data
            with open(f"{source_path}/robot_data.json", "r") as file:
                robot_data = json.load(file)

            # copy robot data
            shutil.copy(f"{source_path}/robot_data.json", f"{output_path}/robot_data.json")

            frames = get_deformation_frames(robot_data, config["force_threshold"])

            # extract mesh center coordinates
            mesh = trimesh.load(f"{source_path}/meshes/mesh-f00001.obj")
            vertices = mesh.vertices / 1000
            mesh_center = np.mean(vertices, axis=0)

            # process images
            get_volucam_images(source_path, output_path, mesh_center, config["volucam_crop_size"])
            get_kinect_images(source_path, output_path, mesh_center, config["kinect_crop_size"])
            get_realsense_images(source_path, output_path)

            # process point clouds
            get_kinect_point_clouds(source_path, output_path, mesh, config["points_kinect_pcd"])
            get_synthetic_point_clouds(source_path, output_path, config["points_dense_synthetic_pcd"], config["points_sparse_synthetic_pcd"])

            # process meshes
            get_meshes(source_path, output_path, frames, config["faces_mesh_simplified"])

            # get camera parameters
            get_camera_parameters(source_path, output_path, mesh_center, config["volucam_crop_size"], config["kinect_crop_size"])


if __name__ == "__main__":
    # read config file
    config_file = f"config/preprocessing.yml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
