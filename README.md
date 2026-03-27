# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
To design and implement a Convolutional Neural Network (CNN) model for image classification using the Fashion-MNIST dataset. The objective is to train the CNN to accurately classify grayscale images of fashion products into predefined categories such as T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle Boot, and to verify the model’s performance on unseen test images.

The Fashion-MNIST dataset is used for this project. It is a benchmark dataset consisting of 70,000 grayscale images of size 28×28 pixels, divided into:

Training Set: 60,000 images

Test Set: 10,000 images

Each image belongs to one of 10 fashion categories.
Include the Problem Statement and Dataset.


## DESIGN STEPS
STEP 1:
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

STEP 2:
Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

STEP 3:
Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

STEP 4:
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

STEP 5:
Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

STEP 6:
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.

## PROGRAM

### Name:SATHISH.B

### Register Number:212224040299

```python

   class CNNClassifier(nn.Module):
    def __init__(self):
       super(CNNClassifier, self).__init__()
       self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
       self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
       self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
       self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
       self.fc1=nn.Linear(128*3*3,128)
       self.fc2=nn.Linear(128,64)
       self.fc3=nn.Linear(64,10)
    def forward(self,x):
       x=self.pool(torch.relu(self.conv1(x)))
       x=self.pool(torch.relu(self.conv2(x)))
       x=self.pool(torch.relu(self.conv3(x)))
       x=x.view(x.size(0),-1)
       x=torch.relu(self.fc1(x))
       x=torch.relu(self.fc2(x))
       x=self.fc3(x)
       return x


```

### OUTPUT

## Training Loss per Epoch
<img width="450" height="492" alt="Screenshot 2026-02-23 094303" src="https://github.com/user-attachments/assets/96f7fd7d-6fc5-45c8-abf6-f566b1ce2344" />

Include the Training Loss per epoch

## Confusion Matrix

Include confusion matrix here
<img width="719" height="537" alt="Screenshot 2026-02-23 094504" src="https://github.com/user-attachments/assets/a2ff3362-7f52-4946-b5dd-e34ddf1337f3" />

## Classification Report
Include classification report here
<img width="494" height="317" alt="Screenshot 2026-02-23 094407" src="https://github.com/user-attachments/assets/9bc2cf3a-1f6b-4dac-b49d-a97fe5bf4dcd" />

### New Sample Data Prediction
Include your sample input and output here
<img width="589" height="521" alt="Screenshot 2026-02-23 094143" src="https://github.com/user-attachments/assets/f64412ef-e17c-4003-bfe2-cddbd087e6be" />

## RESULT
Include your result here
