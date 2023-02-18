# Blood-Cell-Object-Detection-using-YOLO-V8

# Introduction

Classifying different types of blood cells is a critical task in medical imaging, which can help in the diagnosis and treatment of various blood-related diseases. In recent years, computer vision techniques, particularly deep learning models, have been widely used to perform this task. Two popular models in object detection and image classification are **YOLO (You Only Look Once)** and **R-CNN (Regions with Convolutional Neural Networks)**.

This project aims to compare YOLO and R-CNN on a blood cell classification dataset and evaluate their performance in terms of accuracy and efficiency. The study provides valuable insights into the suitability of these models for blood cell classification tasks and helps in selecting the most appropriate model for similar applications.

# Data

Data can be downloaded from the following link.
[https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset/download?datasetVersionNumber=1](https://www.kaggle.com/datasets/draaslan/blood-cell-detection-dataset/download?datasetVersionNumber=1)

This dataset contains 100 labeled images of White Blood Cells (WBC) and Red Blood Cells (RBC) combined. A separate file named `annotations.csv` is provided that contains labeling for each object in the given image.

![image](https://user-images.githubusercontent.com/47318130/219865693-fa0fe647-a642-49a5-b806-d117eee1b152.png)

fig 1 Sample Image file (Unlabeled)

![image](https://user-images.githubusercontent.com/47318130/219865705-20b2a6cf-5db8-482c-9c95-832e849ae527.png)

fig 2 Sample Labeled image

To train various deep learning models, this data has to be converted to the format that is expected by the model. This can be done by a python script or by using tools like robot flow. For demonstration purposes, both methods have been illustrated. The python script is used to generate data in YOLOv5 format. and RoboFlow is used to convert data to YOLOv8 and COCO format.

YOLOv8 format is used to train YOLOv8 model and the COCO format dataset is used to train the RCNN model

# YOLO

YOLO (You Only Look Once) is a real-time object detection model designed for real-time object detection. The key idea behind YOLO is that it **only performs one forward pass** through the network, making it much faster than other object detection systems that may require multiple forward passes.

For this project, YOLO version 8 was used which is a newer/revised version of YOLO. YOLO on its own is much faster compared to traditional methods such as R-CNN but it is also less accurate. Although YOLO has made a significant improvement and is closing the gap.

Transfer learning was used to retrain or fine-tune the model to the new dataset. The model was trained on 80% data (80 images out of 100) for 30 epochs (20 epochs would be efficient). As the model was trained and then finetuned on a specific dataset, not a lot of data is required.

There are various versions of YOLO available based on the number of parameters as seen below. For this demonstration, **YOLOv8m** was used which is a medium size model with 25.9 million parameters.

![image](https://user-images.githubusercontent.com/47318130/219865713-5ce1b11b-953b-4185-bf2d-b12d5f4a91c4.png)

fig 3 YOLO Models

YOLO takes care of image transformations and augmentations internally, resulting in the application of various modifications such as rotation, scaling, and others to the images during the training process. This helps to enhance the robustness of the model by exposing it to a diverse range of variations.

Here are some results from the YOLO model

![image](https://user-images.githubusercontent.com/47318130/219865728-9744cb28-d1f6-48d7-a957-ca947b3f2645.png)

fig 4

![image](https://user-images.githubusercontent.com/47318130/219865733-97f98559-d915-4d44-9f01-a57ccd11efef.png)

fig 5

As seen, the model is accurately able to identify the labels in the images.

![image](https://user-images.githubusercontent.com/47318130/219865745-b3e6df9d-a39d-4123-9070-c27f29835424.png)

fig 6 Trains and val metrics YOLO

The training log for the YOLO model is displayed in the above image (fig 6).
