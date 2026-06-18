from models import models, predictBoxes_batch
from torch_datasets.plantdoc_dataset import get_loader_plantdoc
import concurrent.futures
import multiprocessing

from rich.logging import RichHandler
import logging


def inference(loader, plant_models):

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )

    log = logging.getLogger("rich")

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
        log.info(f"found unique plants: {", ".join(unique_plants)}")

        log.info(
            f"New batch! \n subbatches for plants : \n {"\n  ".join(subbatches.keys())}")

        # Concurrency through thread pool executor (or process pool executor) for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for plant, subbatch in subbatches.items():
                imgs, correct_labels = subbatch
                future = executor.submit(
                    predictBoxes_batch, imgs, plant)
                futures.append(future)
            concurrent.futures.wait(futures)
            for ftr in futures:
                ...
                print(ftr.result())

    return None


if __name__ == "__main__":
    loader = get_loader_plantdoc(batch_size=128)
    plant_doc_inf = inference(loader, models)

    ...
