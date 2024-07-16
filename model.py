from torchvision.models import resnet18
net = resnet18(num_classes=10).cuda()
#imports libraries

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from torch import nn, optim
from PIL import Image

batch_size = 128
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Use GPU if available, otherwise use CPU

transform = transforms.Compose([ # defines transformations applied to TRAINING
    transforms.RandomHorizontalFlip(), # random horizontal reflection
    transforms.RandomCrop(32, padding=4), # random cropping
    transforms.RandomRotation(15), # random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # random brightness/saturation/contrast
    transforms.ToTensor(), # Convert image to a tensor
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)), # Normalize the data, so that the gradients are more consistent, thus making it easier to train
])

transformtest = transforms.Compose([ # defines transformations for TESTING
    transforms.ToTensor(), #converts to tensor
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # normalizes the data
])

# Load the CIFAR10 datasets

train_dataset = datasets.CIFAR10(
    root='train',
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.CIFAR10(
    root='test',
    train=False,
    transform=transformtest,
    download=True,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

net.load_state_dict(torch.load("/content/final_challenge_model_FINAL_BETTER_LAST.pth")) # Load the model

loss_fn = nn.CrossEntropyLoss() # Cross-entropy loss function
optimizer = optim.Adam(net.parameters(), lr = 0.001, weight_decay=1e-4) # Adam optimizer with weight decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Learning rate scheduler

from tqdm import tqdm

num_epochs = 60 # Number of epochs to train the model

for epoch in tqdm(range(0, num_epochs)):  # Iterate over the number of epochs, tqdm for progress bar
  net.train() # Set the model to training mode

  for i, data in enumerate(train_loader): # Iterate over the training data

    images, labels = data # Get the images and labels from the data
    images = images.to(device) # Move the images and labels to the device
    labels = labels.to(device) # Move the images and labels to the device

    pred = net(images) # Forward pass the images through the model
    loss = loss_fn(pred, labels) # Calculate the loss using the model's predictions and the true labels

    loss.backward() # Backward pass to calculate the gradients
    optimizer.step() # Update the model's parameters using the optimizer
    optimizer.zero_grad() # Zeros gradients

    if i % 100 == 99: # Print the loss every 100 iterations
      print(f"Epoch {epoch+1}, Step {i+1}, Loss = {loss.item()}")

  scheduler.step()  # Step the learning rate scheduler after each epoch

torch.save(net.state_dict(), "/content/final_challenge_model_FINAL_BETTER_LAST.pth") # Save the model


# testing

net.eval() # Set the model to evaluation mode
with torch.no_grad(): # Disable gradient calculation

    i = 0 # total cases
    total_accuracy = 0 # total correct cases
    for images, labels in test_loader: # Iterate over the test data

        images = images.to(device)  # Move the images to the device
        labels = labels.to(device)  # Move the labels to the device
        test_output = net(images)  # Forward pass the images through the model
        pred_y = torch.max(test_output, 1)[1]  # Get the predicted class labels
        total_accuracy += (pred_y == labels).sum().item() / float(labels.size(0))  # Calculate the accuracy
        i += 1 # add 1 to total cases

    FinalAccuracy = total_accuracy / i # Calculate the final accuracy

print(f'Test Accuracy of the model on the 10000 test images: {FinalAccuracy:.3f}') # Print the final accuracy

# Test Accuracy of the model on the 10000 test images: 0.817
