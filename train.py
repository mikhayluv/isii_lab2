from ultralytics import YOLO
import torch
from clearml import Task



if __name__ == '__main__':
    task = Task.init(project_name="isii_lab2", task_name='train base model AdamW optimizer and custom box, cls')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO("yolo11m.pt")
    model = model.to(device)

    model.train(data="data.yaml", batch=8, epochs=50, box=9, cls=2)




