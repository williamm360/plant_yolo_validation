from models import predictBoxes_batch
from torch_datasets.plantdoc_dataset import get_loader_plantdoc, plantDoc_labels
import concurrent.futures
# from queue import Queue
from torch.utils.data import DataLoader
import numpy as np
from multiprocessing import Process, Queue

from rich.logging import RichHandler
import logging
from time import sleep


num_workers_data = 4
NUM_WORKERS_INFERENCE = 4
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


def inference(loader: DataLoader, futures_queue: Queue, correct_labels_queue: Queue):
    # Concurrency through thread pool executor (or process pool executor) for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS_INFERENCE) as executor:
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

            # Parallel processing of inference
            futures_list = []
            for plant, subbatch in subbatches.items():
                imgs, correct_labels = subbatch
                future = executor.submit(
                    predictBoxes_batch, imgs, plant)
                futures_list.append(future)
                # Pushes the correct labels into the answer queue (Used to estimate precision afterwards)
                correct_labels_queue.put(correct_labels)

            # Wait for all inference results to be completed (No error handling functions cuz im too lazy)
            _ = concurrent.futures.wait(
                futures_list, return_when=concurrent.futures.ALL_COMPLETED)

            # Get results and push them into the guesses queue (Used to estimate precision afterwards)
            for ftr in futures_list:
                res = ftr.result()
                futures_queue.put(res)

        # Sentinels to know when to terminate data handling process
        for worker in range(num_workers_data):
            futures_queue.put(None)
            correct_labels_queue.put(None)

    return None  # inference is a process that outputs to the queues, it doesnt have a return


def process_data(guesses_queue: Queue, answers_queue: Queue):
    running = True
    while running:
        # Wait if the queue is currently empty. To change since qsize is inconsistent
        if guesses_queue.qsize == 0 or answers_queue.qsize == 0:
            sleep(0.05)

        else:
            guess = guesses_queue.get()

            answer_preproc = answers_queue.get()
            # Sentinel condition to terminate process
            if guess is None or answer_preproc is None:
                if not (guess is None and answer_preproc is None):
                    log.critical(
                        "Non homogeneity between guesses and answers. Program terminated")
                running = False
                break
            answer = get_answ_from_batch(answer_preproc)

            print(len(guess))
            print(len(answer))

    return None  # No return because its a process


def get_answ_from_batch(batch_out):
    # Transformations to deconstruct the answer tensor (it got really jumbled up)
    labels = np.stack([dic["label"] for dic in batch_out], axis=1)
    boxes = np.asarray(
        [np.asarray(dic["box"]).T for dic in batch_out]).transpose(1, 0, 2)

    # list of formatted outputs for quick comparisons betwen answer and guess
    list_out = []

    # Defines the padding constants (this is hardcodded so you cant have more than 255 labels)
    # You can change the values in here and in plantdoc_dataset.py if need be.
    padding_label = np.int64(255)
    padding_box = np.array([0, 0, 0, 0])

    for label, box in zip(labels, boxes):
        label_per_plant = list(zip(label, box))
        # TO change for homogeneity bwtween datasets (or change in dataset creation idk)
        filtered_list = [item for item in label_per_plant if not (
            item[0] == padding_label and np.array_equal(item[1], padding_box))]
        list_out.append(filtered_list)

    return list_out


if __name__ == "__main__":
    loader = get_loader_plantdoc(batch_size=128)
    guess_queue = Queue()
    answer_queue = Queue()
    data_handling = Process(target=process_data,
                            args=(guess_queue, answer_queue))
    data_handling.start()
    inference(loader, guess_queue, answer_queue)
    data_handling.join()
    ...
