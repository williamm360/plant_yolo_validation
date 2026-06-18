from models import models, predictBoxes_batch
from torch_datasets.plantdoc_dataset import get_loader_plantdoc
import concurrent.futures
import multiprocessing

import rich
import logging


def inference(loader, plant_models):
    for batch in loader:
        # create batches for each model
        plants, images, labels = batch
        plant_names = plants[0]
        unique_plants = set(plant_names)
        subbatches = dict()

        # Sorts batches per model into subbatches
        for plant in unique_plants:
            filter_mask = [name == plant for name in plant_names]
            plant_images = images[filter_mask]
            plant_labels = [labels for labels, bool_flag in zip(
                labels, filter_mask) if bool_flag]
            subbatches[plant] = [plant_images, plant_labels]

            print(subbatches[plant], "\n")
        print("New batch! \n subbatches for plants : \n ",
              "\n  ".join(subbatches.keys()))

        # Concurrency through thread pool executor (or process pool executor) for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for plant, subbatch in subbatches.items():
                imgs, correct_labels = subbatch
                future = executor.submit(
                    predictBoxes_batch, imgs, models[plant])
                futures.append[future]
            concurrent.futures.wait(futures)
            for ftr in futures:
                print(ftr)


if __name__ == "__main__":
    loader = get_loader_plantdoc()
    ...
