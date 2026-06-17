from ultralytics import YOLO
import os

potatoModel = YOLO("weights/potatoYOLOv11.pt")
tomatoModel = YOLO("weights/tomatoYOLOv11.pt")
onionModel = YOLO("weights/onionYOLOv11.pt")
lettuceModel = YOLO("weights/lettuceDisease.pt")
appleModel = YOLO("weights/appleDisease.pt")
strawberryModel = YOLO("weights/strawberryDisease.pt")


# Get all possible labels (Used to create homogeneity between different datasets)
def get_labels(*models):
  for model in models:
    labels = list(model.names.values())
    print(f"{model=} \n {"\n  ".join(labels)}")


get_labels(potatoModel, tomatoModel, onionModel,
           lettuceModel, appleModel, strawberryModel)


# Get data classes in the plant village Dataset
results = [r.split("/")[-1] for r, d,
           files in os.walk('/content/Datasets/PlantVillage/train')][1:]
for res in sorted(results):
  print(res)
