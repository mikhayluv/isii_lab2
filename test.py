from ultralytics import YOLO
from clearml import Task

if __name__ == '__main__':
    task = Task.init(project_name="isii_lab2", task_name='aug adamW model val', task_type='testing')

    model = YOLO("C:/Users/mikha/OneDrive/Desktop/ISII/runs/detect/train5/weights/best.pt")
    metrics = model.val(data="data.yaml", split="test")

    results_dict = metrics.results_dict

    for key, value in results_dict.items():
        task.get_logger().report_single_value(name=key, value=value)
