import os
import zipfile
import wget
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

def main():
    # Download
    if os.path.isdir('datasets/VOC/VOCdevkit'):
        print('VOC dataset already exists. \nTo download the VOC dataset again remove  "datasets/VOC/VOCdevkit" directory ')
        return


    data_dir = 'datasets/VOC'  # dataset root dir
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.isfile(data_dir + '/VOCtest_06-Nov-2007.zip'):
        url_test = "https://github.com/ultralytics/yolov5/releases/download/v1.0/VOCtest_06-Nov-2007.zip"  # 438MB, 4953 images
        wget.download(url_test, data_dir)
    if not os.path.isfile(data_dir + '/VOCtrainval_06-Nov-2007.zip'):
        url_trainval = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/VOCtrainval_06-Nov-2007.zip' # 446MB, 5012 images
        wget.download(url_trainval, data_dir)



    with zipfile.ZipFile(data_dir + '/VOCtrainval_06-Nov-2007.zip', 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    with zipfile.ZipFile(data_dir + '/VOCtest_06-Nov-2007.zip', 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Convert
    path = data_dir + '/VOCdevkit'
    for year, image_set in ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
        imgs_path = data_dir + '/images/' +  f'{image_set}'
        lbs_path = data_dir + '/labels/' + f'{image_set}'
        imgs_path = Path(imgs_path)
        lbs_path = Path(lbs_path)
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)

        with open(path + f'/VOC{year}/ImageSets/Main/{image_set}.txt') as f:
            image_ids = f.read().strip().split()
        for id in tqdm(image_ids, desc=f'{image_set}{year}'): 
            f = path + f'/VOC{year}/JPEGImages/{id}.jpg'  # old img path 
            f = Path(f)
            f.rename(imgs_path / f.name)  # move image

if __name__ == "__main__":
    main()