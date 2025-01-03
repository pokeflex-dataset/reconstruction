import yaml
import argparse
from training.training_images import images_main
from training.training_pointclouds import pointclouds_main
from training.training_robot_data import robot_data_main
from training.training_img_robot_data import img_robot_data_main

if __name__ == "__main__":
    modalities = ["images", "pointclouds", "robot_data", "img_robot_data"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="img_robot_data", choices=modalities)
    args = parser.parse_args()

    # read training config file
    config_file = f"config/training_{args.modality}.yml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    if args.modality == "images":
        images_main(config)
    elif args.modality == "pointclouds":
        pointclouds_main(config)
    elif args.modality == "robot_data":
        robot_data_main(config)
    elif args.modality == "img_robot_data":
        img_robot_data_main(config)
    else:
        raise ValueError(f"Unknown modality: {args.modality}")


