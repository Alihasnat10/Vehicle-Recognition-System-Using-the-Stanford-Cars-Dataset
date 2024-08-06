# Vehicle-Recognition-System-Using-the-Stanford-Cars-Dataset
This project aims to identify and recognise a car’s model given its image.

## Pytorch Project

### Table of Contents
1. [Introduction](#introduction)
    1. [Title](#title)
    2. [Description](#description)
    3. [Dataset](#dataset)
2. [Challenge in Preparing Dataset](#challenge-in-preparing-dataset)
3. [Preprocessing and Transformation](#preprocessing-and-transformation)
    1. [Results After Transformation](#results-after-transformation)
4. [Models and Training](#models-and-training)
    1. [GoogLeNet Transfer Learning](#googlenet-transfer-learning)
    2. [Simple CNN Classification Model](#simple-cnn-classification-model)
    3. [Complex CNN Classification Model](#complex-cnn-classification-model)
    4. [GoogLeNet Trained on the Entire Data](#googlenet-trained-on-the-entire-data)
5. [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction

### Title
Vehicle Recognition System Using the Stanford Cars Dataset

### Description
This project aims to identify and recognize a car’s model given its image. Using the Stanford Cars dataset, which contains images of various car classes with labels, I plan to train an image classification model that will learn to differentiate between different cars and be able to name them. This project can have many use cases such as traffic monitoring, vehicle identification, and car dealership inventory tracking.

### Dataset
The Stanford Cars dataset consists of images of 196 classes of cars. The dataset is divided into a training set and a testing set, with labels indicating the car class for each image.
<img width="539" alt="image" src="https://github.com/user-attachments/assets/b231c109-072a-41a7-b818-d0cd33ee3456">

## Challenge in Preparing Dataset
The Stanford Cars dataset's original link is broken, so it couldn't be imported directly into PyTorch. Hence, I found the dataset on Kaggle and downloaded it onto my machine. I then wrote a custom dataset class to load the data in the PyTorch format.

## Preprocessing and Transformation
I used GoogLeNet’s transformation to resize and augment the images. This helped in getting more diverse data.

### Results After Transformation
<img width="554" alt="image" src="https://github.com/user-attachments/assets/dac006fe-fbb5-4865-9776-87437af6443d">


## Models and Training

### GoogLeNet Transfer Learning
The first model I used is a pretrained GoogLeNet model for image classification. I changed the last layer to output the number of classes in the Stanford Cars dataset. I trained for 5 epochs on a Google Colab GPU.

#### Results
<img width="505" alt="image" src="https://github.com/user-attachments/assets/651a3661-8296-4e7e-a182-84f4c3738958">


### Simple CNN Classification Model
I trained a simple convolutional neural network with few layers for 10 epochs.

#### Results
<img width="553" alt="image" src="https://github.com/user-attachments/assets/fd9b6971-ad2f-4297-90a3-89e6e0e7b6ca">


### Conclusion
It can be seen that the prediction is always GMC Canyon, which is wrong. After exploring this problem, I found that the GMC class had the highest number of training images compared to any other class, hence the model was biased. To avoid this, undersampling should be applied and the model should be trained for a higher number of epochs.

### Complex CNN Classification Model
A more complex classification model with more layers and a greater number of neurons.

#### Results
<img width="562" alt="image" src="https://github.com/user-attachments/assets/d92e8a35-9512-40a5-b69a-3a7d033b8286">


The model has better accuracy than the previous simpler model but still is not accurate. It needs more training time.

### GoogLeNet Trained on the Entire Data
I trained GoogLeNet on the entire data instead of a subset. I noticed that the accuracy was increasing in every epoch and the loss was decreasing. This means that the model is converging and learning patterns.

<img width="574" alt="image" src="https://github.com/user-attachments/assets/29fecc8e-b7cf-44ae-9f84-4ab02ea2a953">


## Conclusion and Future Work
Based on the above analysis and model evaluation, the final conclusion is that models require more complexity and training time and power to improve accuracy. I augmented the training dataset to generate more data, but it took more time to train on the large data, hence I discarded the extra data. 

In future, more data can be added using augmentation techniques, a more complex model can be used, or the GoogLeNet can be trained for 100 or 200 epochs to get a higher accuracy. As mentioned previously, some classes had a higher number of images that caused bias in the model. This can be evaluated using the F1 score and a confusion matrix.
