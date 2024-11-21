import os
import shutil
from sklearn.model_selection import train_test_split

# Пути к исходным папкам
images_path = "urbanhack_data/images"  # Папка с изображениями
labels_path = "urbanhack_data/labels"  # Папка с аннотациями

# Папка для сохранения разделённых выборок
output_path = "data"
os.makedirs(os.path.join(output_path, "train/images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "train/labels"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val/images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val/labels"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test/images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "test/labels"), exist_ok=True)

# Получаем список всех изображений
images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

# Разделяем изображения на train, val, test
train_files, temp_files = train_test_split(images, test_size=0.3, random_state=1)
val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=1)

# Функция для копирования изображений и соответствующих аннотаций
def copy_files(file_list, dest_images, dest_labels):
    for file in file_list:
        # Копируем изображение
        shutil.copy(os.path.join(images_path, file), os.path.join(dest_images, file))
        # Копируем аннотацию
        annotation = file.rsplit('.', 1)[0] + '.txt'
        if os.path.exists(os.path.join(labels_path, annotation)):
            shutil.copy(os.path.join(labels_path, annotation), os.path.join(dest_labels, annotation))

# Копируем файлы в соответствующие папки
copy_files(train_files, os.path.join(output_path, "train/images"), os.path.join(output_path, "train/labels"))
copy_files(val_files, os.path.join(output_path, "val/images"), os.path.join(output_path, "val/labels"))
copy_files(test_files, os.path.join(output_path, "test/images"), os.path.join(output_path, "test/labels"))





