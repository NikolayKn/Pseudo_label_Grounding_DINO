import pandas as pd
import os
import cv2


COLORS = [
    [255, 0, 0],        # Красный
    [0, 255, 0],        # Зеленый
    [0, 0, 255],        # Синий
    [255, 255, 0],      # Желтый
    [255, 0, 255],      # Фиолетовый
    [0, 255, 255],      # Бирюзовый
    [128, 0, 0],        # Темно-красный
    [0, 128, 0],        # Темно-зеленый
    [0, 0, 128],        # Темно-синий
    [128, 128, 0],      # Темно-желтый
    [128, 0, 128],      # Темно-фиолетовый
    [0, 128, 128],      # Темно-бирюзовый
    [192, 192, 192],    # Серый
    [255, 165, 0],      # Оранжевый
    [255, 192, 203],    # Розовый
    [139, 69, 19],      # Коричневый
    [169, 169, 169],    # Темно-серый
    [144, 238, 144],    # Светло-зеленый
    [255, 20, 147],     # Гелиотроп
    [255, 99, 71]       # Темно-красный
] 

class AnnotationDF:
    def __init__(self, classes=None):
        self.classes = classes
        self.df = pd.DataFrame([], columns=['image_name','class', 'class_index','num_boxes', 'probability', 'bbox', 'box_area'])
        self.df_incorrect = pd.DataFrame([], columns=['image_name','class', 'class_index','num_boxes', 'probability', 'bbox', 'box_area'])

    def add_annotation(self, filename, detections):
        for box, logit, phrase in detections:
            try:
                class_index= self.classes.index(phrase)
            except ValueError:
                class_index= -1
            box_string = ' '.join([str(x) for x in box.tolist()])
            box_area = box[2] * box[3]
            new_row = pd.DataFrame([pd.Series({
                    'image_name':filename[0],
                    'class':phrase, 
                    'class_index':class_index, 
                    'num_boxes': len(detections),
                    'probability':logit.item(),
                    'bbox':box_string,
                    'box_area': box_area.item()
                    })])
            if phrase in self.classes:
                self.df = pd.concat([self.df, pd.DataFrame(new_row)], ignore_index=True)
            else:
                self.df_incorrect = pd.concat([self.df_incorrect, pd.DataFrame(new_row)], ignore_index=True)
    
    def save_annotations(self, save_dir):
        if self.df.shape[0] != 0:
            df_filename = os.path.join(save_dir, 'annotations.parquet')
            self.df.to_parquet(df_filename)
        if self.df_incorrect.shape[0] != 0:
            df_filename = os.path.join(save_dir, 'annotations_incorrect.parquet')
            self.df_incorrect.to_parquet(df_filename)

    def create_COCO_annotation(self):
        pass

    def load_annotation(self, load_dir):
        df_filename = os.path.join(load_dir, 'annotations.parquet')
        df_incorrect_filename = os.path.join(load_dir, 'annotations_incorrect.parquet')
        if os.path.isfile(df_filename):
            self.df = pd.read_parquet(df_filename, engine='pyarrow')
        if os.path.isfile(df_incorrect_filename):
            self.df_incorrect = pd.read_parquet(df_incorrect_filename, engine='pyarrow')


def draw_annotations(image, annotations, class_names):

    labels = []
    # Размеры изображения
    height, width, _ = image.shape

    # Отобразить аннотации на изображении
    for annotation in annotations:
        class_index, x_center, y_center, w, h = annotation
        class_index = int(class_index)
        labels.append(class_names[class_index])
        
        # Преобразование относительных координат в абсолютные
        x = int(x_center * width)
        y = int(y_center * height)
        w = int(w * width)
        h = int(h * height)

        # Вычислить координаты углов прямоугольника
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        x_max = int(x + w / 2)
        y_max = int(y + h / 2)

        # Отобразить прямоугольник на изображении
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=COLORS[class_index], thickness=2)

        # Добавить метку класса
        image = cv2.putText(image, class_names[class_index], (x_min, y_min - 2), 
                          fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=COLORS[class_index],thickness=1)
        
    return image, labels