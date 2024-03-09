import cv2
import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms
from torchvision.models import resnet18, resnet101, resnet50, ResNet18_Weights
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F
import random

from models import ResNetClassifier, ResNetClassifier1

############################################################################

dataset = []
support_set = []
query_set = []

with h5py.File('./datasets/dataset-ACDC.h5', 'r') as h5_file:
    images_group = h5_file['images']
    labels_dataset = h5_file['labels']

    random_indices = []

    # query_set_len = 32
    query_set_len = 64

    query_set_len_label0 = query_set_len // 2
    random.seed(42)
    random_indices_label0 = random.sample(range(len(images_group) // 2), query_set_len_label0)
    random_indices.extend(random_indices_label0)

    query_set_len_label1 = query_set_len // 2
    random.seed(42)
    random_indices_label1 = random.sample(range(len(images_group) // 2, len(images_group)), query_set_len_label1)
    random_indices.extend(random_indices_label1)

    # print(random_indices)
    with tqdm(total=len(images_group), desc='Loading dataset') as pbar:
        for i in range(len(images_group)):
            # Load image from HDF5
            image = images_group[str(i)][()]

            # Load label from HDF5 dataset
            label = int(labels_dataset[i])

            if i in random_indices:
                query_set.append((image, label))
            else:
                support_set.append((image, -1))

            pbar.update(1)

for item in support_set:
    dataset.append(item)

print(len(query_set))

############################################################################

class MotionArtifactDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        image, label = data
        image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
        image = image * 255  # Normalize between 0 and 1
        image = cv2.resize(image, (224, 102))
        image = self.transform(image).float()
        image = np.array(image)
        return image, label


############################################################################

# Initialize the ResNet classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# D_Labeld
train_dataset = MotionArtifactDataset(query_set)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

###############################################################################

pre_trained_model = ResNetClassifier().to(device)
pre_trained_model.resnet = nn.Sequential(*list(pre_trained_model.resnet.children())[:-1])
pre_trained_model.load_state_dict(torch.load("student_model_64Q_DA1_FL1_40_1.pth"))
pre_trained_model = pre_trained_model.to(device)

# Create a new model with the pretrained ResNet backbone and the FC layer for classification
student_model = ResNetClassifier1().to(device)
student_model.resnet = pre_trained_model.resnet

optimizer_fc_fine_tuning = optim.Adam(student_model.parameters(), lr=0.0001)

criterion = nn.BCELoss()

# Define empty lists to store loss and accuracy values
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 20
best_train_loss = float('inf')  # Initialize with a high value

for epoch in range(num_epochs):
    print("---------Training phase----------")
    # Training phase
    student_model.train()
    train_loss = 0.0
    train_correct = 0

    train_predictions = []
    train_true_labels = []

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train", ncols=80) as pbar_train:
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            student_model.zero_grad()

            outputs = student_model(images)
            outputs = torch.squeeze(outputs)

            predicted = torch.round(outputs)
            train_correct += (predicted == labels).sum().item()
            train_predictions.extend(predicted.tolist())
            train_true_labels.extend(labels.tolist())

            loss = criterion(outputs, labels.float())

            loss.backward()
            optimizer_fc_fine_tuning.step()

            train_loss += loss.item() * images.size(0)
            pbar_train.set_postfix({'Loss': loss.item()})
            pbar_train.update()

    train_accuracy = train_correct / len(train_dataset)
    train_loss /= len(train_dataset)

    # Check if the current train loss is lower than the previous best train loss
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        torch.save(student_model.state_dict(), "fine_tuned_student_model_64Q_DA1_FL1_40_1_20F.pth")

    # Append accuracy and loss to the respective lists
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f}")
