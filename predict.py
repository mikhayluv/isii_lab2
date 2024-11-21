from ultralytics import YOLO
# from clearml import Task

if __name__ == '__main__':
    # task = Task.init(project_name="isii_lab2", task_name='aug model predict', task_type= 'testing')

    model = YOLO('runs/detect/train5/weights/best.pt')
    dir_path = "data/test/images/"

    model.predict(source=dir_path, show=False, save=True, save_conf=True, conf=0.5, line_width=1)