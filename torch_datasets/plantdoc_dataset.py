"""
    Returns a dataloader for the plantDoc dataset
    Formatted with rich logging for better logs

"""
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from io import BytesIO
import torch
import cv2
import numpy as np
import os
import logging
from rich.logging import RichHandler


# Dataset sub-datasets and Data Loader

# Full dataset containing all images from PlantDoc (including non detected diseases)

# Dataset to filter (Doesnt load images as it would be too much storage)


class PlantDoc_valid_labels_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.ds_path: Path = Path(dataset_path)
        self.image_paths = self.get_all_paths()

    def __getitem__(self, idx):
        # Path chosen through indexing
        path_image: Path = self.image_paths[idx]
        # Creating the path for the labels
        path_parts = list(path_image.parts)
        path_parts[-2] = "labels"
        labels_path_wo_suffix = Path(*path_parts)
        path_labels = labels_path_wo_suffix.with_suffix(".txt")

        # Getting the labels from .txt file
        labels = []
        with open(path_labels, "r", encoding="utf-8") as file:
            for line in file:
                label, x1, y1, x2, y2 = map(str.strip, line.split(" "))
                labels.append({"label": label, "box": (x1, y1, x2, y2)})
        return (path_image, labels)

    def __len__(self):
        # Returns the length of the dataset
        return len(self.image_paths)

    def get_all_paths(self):
        # get all paths for the images in the dataset (labels paths are inferred later)
        image_paths = list(self.ds_path.rglob("**/*.jpg"))
        return image_paths

# Filtered Dataset with labels


class PlantDoc_Dataset_distilled(Dataset):
    def __init__(self, items, labels_dict):
        self.resize_transform = T.Resize(size=(256, 256), antialias=True)
        self.max_labels = 16  # cuz i need uniform tensor sizes rip
        self.max_plants = 1
        self.items = items
        self.labels_dict = labels_dict

    def __getitem__(self, idx):
        img_path, original_labels = self.items[idx]
        labels = [item.copy() for item in original_labels]
        plants = set()
        for label in labels:
            for plant, dicts in self.labels_dict["plants"].items():
                if label["label"] in dicts["idx"]:
                    plants.add(plant)

        with open(img_path, "rb") as f:
            img_buffer = BytesIO(f.read())
        raw_bytes = np.frombuffer(img_buffer.getbuffer(), dtype=np.uint8)
        img_array_np = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)[
            :, :, ::-1]  # reverts the colors into rgb
        img_tensor = torch.from_numpy(
            np.ascontiguousarray(img_array_np)).permute(2, 0, 1)
        img = self.resize_transform(img_tensor)

        plant_list = list(plants)
        plant_list.extend(["None",]*(self.max_plants-len(plant_list)))
        plant_list = plant_list[:self.max_plants]

        padding_dict = {"label": "None", "box": ("0", "0", "0", "0")}
        labels.extend([padding_dict] * (self.max_labels - len(labels)))
        labels = labels[:self.max_labels]

        return plant_list, img, labels

    def __len__(self):
        return len(self.items)


# Define which labels will be tested
plantDoc_labels = {"plants":
                   {
                       "Potato":
                       {
                           "labels": {"11": "EarlyBlight", "12": "LateBlight", "13": "Healthy"},
                           "idx": {"11", "12", "13"},
                       },
                       "Tomato":
                       {
                           "labels": {"19": "EarlyBlight", "20": "SeptoriaLeafSpot",
                                      "21": "BacterialSpot", "22": "TomatoLateBlight",
                                      "23": "MosaicVirus", "24": "YellowLeafCurlVirus",
                                      "25": "HealthyTomato", "26": "LeafMold", "27": "SpiderMites"},
                           # index
                           "idx": {"19", "20", "21", "22", "23", "24", "25", "26", "27"}
                       },
                   },
                   "used_idx": {"11", "12", "13", "19", "20", "21", "22", "23", "24", "25", "26", "27"}
                   }


# Checks if the label is valid
def is_valid(item, all_labels):
    _, item_labels = item
    for item_label in item_labels:
        if item_label["label"] not in all_labels["used_idx"]:
            return False
    if not item_labels:
        return False

    return True


def get_loader_plantdoc(batch_size=32, num_workers=4, prefetch_factor=3, persistent_workers=True):

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    log = logging.getLogger("rich")

    # Downloads the dataset from Kaggle if it does not currently exist on local fs
    if not os.path.exists("../datasets/plant_doc"):
        from install import download_plantdoc
        log.info("Downloading Dataset")
        download_plantdoc()

    log.info("loading labels ds...")
    # Creates the complete dataset (compatibility-agnostic)
    labels_dataset = PlantDoc_valid_labels_Dataset("../datasets/plant_doc")
    log.info("labels ds finished. filtering items...")
    # filters items to only use those that can be detected by the model
    filtered_items = [
        item for item in labels_dataset if is_valid(item, plantDoc_labels)]
    log.info("filtering complete. Creating new filtered dataset...")
    # Creating the filtered dataset
    filtered_plantdoc_dataset = PlantDoc_Dataset_distilled(
        filtered_items, plantDoc_labels)
    log.info("Creating data loader")
    # Creates the dataloader
    loader = DataLoader(filtered_plantdoc_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers,
                        pin_memory=True,
                        pin_memory_device="cuda")

    log.info("Loader created")

    return loader
