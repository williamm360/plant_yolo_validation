from ultralytics import YOLO
import torch
# adapted code for multiprocessing and batching
potatoModel = YOLO("weights/potatoYOLOv11.pt")
tomatoModel = YOLO("weights/tomatoYOLOv11.pt")
onionModel = YOLO("weights/onionYOLOv11.pt")
lettuceModel = YOLO("weights/lettuceDisease.pt")
appleModel = YOLO("weights/appleDisease.pt")
strawberryModel = YOLO("weights/strawberryDisease.pt")

models = {
    "Potato": potatoModel,
    "Tomato": tomatoModel,
    "Onion": onionModel,
    "Lettuce": lettuceModel,
    "Apple": appleModel,
    "Strawberry": strawberryModel,
}


def predictBoxes_batch(images: torch.Tensor, plant):
    model = models[plant]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    images.float().to(device=device, dtype=torch.float32)

    with torch.no_grad():
        results = model(images)
    output = []
    for res in results:
        boxes = res[0].boxes
        output_per_image = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            label = res[0].names[class_id]
            output_per_image.append({
                "label": label,
                "confidence": confidence,
                "box": [x_min, y_min, x_max, y_max]
            })
        output.append(output_per_image)

    return output
