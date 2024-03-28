# Pseudo labelling using Grounding DINO

The repository contains code for obtaining pseudo labelling using Grounding DINO. The pseudo labelling was evaluated using the mAP metric, the implementation of which was taken from the Ultralytics repository.


## Datasets

I use PASCAL VOC 2007 Dataset. The PASCAL VOC (Visual Object Classes) dataset is a well-known object detection, segmentation, and classification dataset. It is designed to encourage research on a wide variety of object categories and is commonly used for benchmarking computer vision models. It is an essential dataset for researchers and developers working on object detection, segmentation, and classification tasks.

## Instructions

**Installation:**

1.Clone the repository from GitHub.

```bash
git clone git@github.com:NikolayKn/Pseudo_label_Grounding_DINO.git
```

2. Run the ```setup.sh``` script to download the Grounding DINO with model weights and install all requrements.

```bash
sh setup.sh
```


## Credits
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Pascal VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/)
- [ mAP realization by Ultralitics](https://github.com/ultralytics/yolov5/blob/master/val.py)  
