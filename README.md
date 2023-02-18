# Blood-Cell-Object-Detection-using-YOLO-V8

# Introduction

Classifying different types of blood cells is a critical task in medical imaging, which can help in the diagnosis and treatment of various blood-related diseases. In recent years, computer vision techniques, particularly deep learning models, have been widely used to perform this task. Two popular models in object detection and image classification are **YOLO (You Only Look Once)** and **R-CNN (Regions with Convolutional Neural Networks)**.

This project aims to compare YOLO and R-CNN on a blood cell classification dataset and evaluate their performance in terms of accuracy and efficiency. The study provides valuable insights into the suitability of these models for blood cell classification tasks and helps in selecting the most appropriate model for similar applications.

# Data

Data can be downloaded from the following link.
[https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset/download?datasetVersionNumber=1](https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset/download?datasetVersionNumber=1)

This dataset contains 100 labeled images of White Blood Cells (WBC) and Red Blood Cells (RBC) combined. A separate file named `annotations.csv` is provided that contains labeling for each object in the given image.

![fig 1 Sample Image file (Unlabeled)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6df9e24d-2f98-4f6f-a9c6-7e44b144191d/66fe360f-776f-425a-aad7-2d71672bb231.png)

fig 1 Sample Image file (Unlabeled)

![fig 2 Sample Labeled image](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e3988ef2-1b4a-42c9-8291-d7d483bf2cbc/62eedc08-f3ee-4ee1-bdbb-cd3147f97611.png)

fig 2 Sample Labeled image

To train various deep learning models, this data has to be converted to the format that is expected by the model. This can be done by a python script or by using tools like robot flow. For demonstration purposes, both methods have been illustrated. The python script is used to generate data in YOLOv5 format. and RoboFlow is used to convert data to YOLOv8 and COCO format.

YOLOv8 format is used to train YOLOv8 model and the COCO format dataset is used to train the RCNN model

# YOLO

YOLO (You Only Look Once) is a real-time object detection model designed for real-time object detection. The key idea behind YOLO is that it **only performs one forward pass** through the network, making it much faster than other object detection systems that may require multiple forward passes.

For this project, YOLO version 8 was used which is a newer/revised version of YOLO. YOLO on its own is much faster compared to traditional methods such as R-CNN but it is also less accurate. Although YOLO has made a significant improvement and is closing the gap.

Transfer learning was used to retrain or fine-tune the model to the new dataset. The model was trained on 80% data (80 images out of 100) for 30 epochs (20 epochs would be efficient). As the model was trained and then finetuned on a specific dataset, not a lot of data is required.

There are various versions of YOLO available based on the number of parameters as seen below. For this demonstration, **YOLOv8m** was used which is a medium size model with 25.9 million parameters.

![fig 3 YOLO Models](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/259fd9c8-7e5d-4dfc-8fd1-230e284e1cb1/Untitled.png)

fig 3 YOLO Models

YOLO takes care of image transformations and augmentations internally, resulting in the application of various modifications such as rotation, scaling, and others to the images during the training process. This helps to enhance the robustness of the model by exposing it to a diverse range of variations.

Here are some results from the YOLO model

![fig 4](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4e72b4e1-7427-473e-955e-f012d2caba6b/val_batch0_pred.jpg)

fig 4

![fig 5](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/780a2262-bd36-465b-857e-bfb25692b653/val_batch1_pred.jpg)

fig 5

As seen, the model is accurately able to identify the labels in the images.

![fig 6 Trains and val metrics YOLO](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5877158-b9e8-46ee-8384-d027a6a81d6b/results.png)

fig 6 Trains and val metrics YOLO

The training log for the YOLO model is displayed in the above image (fig 6).
