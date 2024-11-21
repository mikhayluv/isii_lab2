import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


input_images_dir = "data/train/images/"  
input_labels_dir = "data/train/labels/"  
output_images_dir = "data/augmented/images/"  
output_labels_dir = "data/augmented/labels/"  

os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),                                                            # Горизонтальное отражение
    A.RandomBrightnessContrast(p=0.2),                                                  # Случайная яркость и контраст
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),     # Сдвиг, масштаб и поворот
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),                                           # Размытие
    A.HueSaturationValue(p=0.3),                                                        # Изменение оттенка и насыщенности
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Функция для чтения аннотаций YOLO
def read_yolo_annotations(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        label, x, y, w, h = map(float, line.strip().split())
        bboxes.append([x, y, w, h])
        class_labels.append(int(label))
    return bboxes, class_labels

# Функция для сохранения аннотаций YOLO
def save_yolo_annotations(output_path, bboxes, class_labels):
    with open(output_path, 'w') as file:
        for label, (x, y, w, h) in zip(class_labels, bboxes):
            file.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

# Проходим по всем изображениям
for img_file in os.listdir(input_images_dir):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):  # Проверяем формат
        img_path = os.path.join(input_images_dir, img_file)
        label_path = os.path.join(input_labels_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # Читаем изображение и аннотации
        image = cv2.imread(img_path)
        h, w, _ = image.shape  # Определяем размер изображения
        bboxes, class_labels = read_yolo_annotations(label_path)

        # Применяем аугментации
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = transformed['image']
        augmented_bboxes = transformed['bboxes']
        augmented_labels = transformed['class_labels']

        # Сохраняем аугментированные данные
        output_img_path = os.path.join(output_images_dir, f"aug_{img_file}")
        output_label_path = os.path.join(output_labels_dir, f"aug_{img_file.replace('.jpg', '.txt').replace('.png', '.txt')}")
        cv2.imwrite(output_img_path, augmented_image)
        save_yolo_annotations(output_label_path, augmented_bboxes, augmented_labels)

print("Аугментация завершена!")