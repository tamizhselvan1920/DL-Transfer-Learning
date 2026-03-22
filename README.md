# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset

An organization has a dataset of labeled images belonging to multiple categories, and accurate classification of these images is important for automation and decision-making. However, training a deep neural network from scratch requires a large amount of data and computational resources.

To address this, the organization plans to use transfer learning with the VGG19 architecture, which is a pre-trained convolutional neural network that has already learned rich feature representations from large-scale image datasets. By reusing this model, the system can efficiently extract important visual features from images.

The model will be fine-tuned using the given dataset so that it adapts to the specific classification task. This reduces training time and improves performance, especially when the dataset is limited.

After training, the model will be used to classify new, unseen images and evaluate its accuracy. The objective is to achieve high classification performance while minimizing computational cost and training effort.

## Neural Network Model

<img width="937" height="836" alt="image" src="https://github.com/user-attachments/assets/1a0b2a5b-5aad-4744-936f-11a4ee45c913" />


## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name: THAMIZH SELVAN R

### Register Number: 212222230158

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])

!unzip -qq ./chip_data.zip -d data

# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

# Show sample images from the training dataset
show_sample_images(train_dataset)

# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the training dataset
print(f"Total number of testing samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = test_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


## Step 2: Load Pretrained Model and Modify for Transfer Learning
# Load a pre-trained VGG19 model
# write your code here
model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
# Write your code here
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

summary(model, input_size=(3, 224, 224))

# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    # Write your code here
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        # Compute validation loss
        # Write your code here
        running_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for images, labels in test_loader:
              images, labels = images.to(device), labels.to(device)
              outputs=model(images)
              loss=criterion(outputs,labels.unsqueeze(1).float())
              val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: ABISHA RANI S")
    print("Register Number: 212224040012")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model
# Write your code here
train_model(model, train_loader,test_loader)


## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels=labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs=torch.sigmoid(outputs)
            predicted=(probs>0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Name: ABISHA RANI S")
    print("Register Number: 212224040012")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name: ABISHA RANI S")
    print("Register Number: 212224040012")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))


# Evaluate the model
# write your code here
test_model(model, test_loader)


## Step 5: Predict on a Single Image and Display It
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()


    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)


#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)


```

### OUTPUT

<img width="1325" height="145" alt="image" src="https://github.com/user-attachments/assets/37e1e5c6-38b7-48d4-a3db-a72d38f5ed31" />

<img width="1347" height="60" alt="image" src="https://github.com/user-attachments/assets/a31f1526-a40c-4d6f-8ec5-69b3279d9527" />

<img width="1253" height="52" alt="image" src="https://github.com/user-attachments/assets/72bae47d-1432-49d9-9928-10b7487f78f3" />

<img width="586" height="633" alt="image" src="https://github.com/user-attachments/assets/2d1f0b7a-8516-49a0-be38-13e22f40aba8" />

<img width="980" height="626" alt="image" src="https://github.com/user-attachments/assets/42ea135e-9d23-4330-9c4b-e5092a4d7a74" />


## Training Loss, Validation Loss Vs Iteration Plot

<img width="947" height="730" alt="image" src="https://github.com/user-attachments/assets/5501796c-0919-432a-9c60-0a951f3e6ada" />


## Confusion Matrix

<img width="778" height="605" alt="image" src="https://github.com/user-attachments/assets/130ead7b-55ac-4be1-bc04-163c193ae554" />


## Classification Report

<img width="663" height="212" alt="image" src="https://github.com/user-attachments/assets/0bf686db-3684-4ae1-9449-1b48c549d2a6" />


### New Sample Data Prediction

<img width="490" height="410" alt="image" src="https://github.com/user-attachments/assets/19014dbf-6ecb-4036-b9e8-76ce3e1d9fea" />

<img width="727" height="410" alt="image" src="https://github.com/user-attachments/assets/e1920d62-a119-4b25-84b1-aa54f827daa9" />


## RESULT

The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
