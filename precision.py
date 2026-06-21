import numpy as np


def get_IoU(boxes1: list[list], boxes2: list[list]):
    ...


def get_weighted_IoU(boxes1: list[list], boxes2: list[list]):
    ...


def get_mAP50():
    ...


def get_mAP50_95():
    ...


class precision():
    __slots__ = ("true_pos", "wrong_plant", "false_positive",
                 "true_negative", "False negative", "no detection")

    def __init__(self):
        self.true_pos = 0
