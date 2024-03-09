import os
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms
from torchvision.models import resnet18, resnet101, resnet50, ResNet18_Weights
from sklearn.metrics import precision_score, recall_score
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from model import ResNetClassifier

datasets_path = [
    "../../Datasets/dataset-York.h5",
    "../../Datasets/dataset-laxRecon.h5",
    "../../Datasets/dataset-saxRecon.h5",
    "../../Datasets/dataset-ACDC.h5",
    "../../Datasets/dataset-BRAIN-MRI.h5",
    "../../Datasets/dataset-CMRxMotion.h5",
    "../../Datasets/dataset-CT.h5"
]

datasets = []

for dataset_path in datasets_path:

    # Load dataset
    dataset = []

    with h5py.File(dataset_path, 'r') as h5_file:
        images_group = h5_file['images']
        labels_dataset = h5_file['labels']

        with tqdm(total=len(images_group), desc=f'Loading {dataset_path}') as pbar:
            for i in range(len(images_group)):
                # Load image from HDF5
                image = images_group[str(i)][()]

                # Load label from HDF5 dataset
                label = int(labels_dataset[i])

                dataset.append((image, label))

                pbar.update(1)

    datasets.append(dataset)

# ################################################################

# Prepare the dataset
class MotionArtifactDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
        image = image * 255  # Scale to 0-255 range
        image = cv2.resize(image, (224, 102))
        image = self.transform(image).float()
        image = np.array(image)

        return image, label


################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################################

from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Split the dataset into training, validation, and test sets

#YORK_Teacher -> datasets[0]
trainval_dataset, test_dataset = train_test_split(datasets[0], test_size=0.2, random_state=42)

# Define the number of folds (k) for cross-validation
num_folds = 5

# Split the training and validation sets into k folds using KFold
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Create empty lists to store the overall loss and accuracy values for each fold
overall_train_losses = []
overall_val_losses = []
overall_test_losses = []
overall_train_accuracies = []
overall_val_accuracies = []
overall_test_accuracies = []

test_results = []

ACDC_BEST = 0

# Training loop with cross-validation
for fold, (train_indices, val_indices) in enumerate(kfold.split(trainval_dataset)):
    # print(f"--------- Fold {fold + 1}/{num_folds} ---------")

    # Split the training set into training and validation sets for the current fold
    train_dataset = [trainval_dataset[i] for i in train_indices]
    val_dataset = [trainval_dataset[i] for i in val_indices]

    # Create data loaders for the current fold
    batch_size = 8  # Adjust the batch size as needed
    train_loader = DataLoader(MotionArtifactDataset(train_dataset), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MotionArtifactDataset(val_dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MotionArtifactDataset(test_dataset), batch_size=batch_size, shuffle=True)

    # Initialize the model and other variables for the current fold
    model = ResNetClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    test_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    # Training loop for the current fold
    num_epochs = 20  # Adjust the number of epochs as needed
    for epoch in range(num_epochs):
        # print(f"--------- Training phase - Epoch {epoch + 1}/{num_epochs} ---------")
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_predictions = []
        train_true_labels = []

        with tqdm(train_loader, desc=f"Fold {fold + 1}/{num_folds} - Epoch {epoch + 1}/{num_epochs} - Train",
                  ncols=80) as pbar_train:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                predicted = torch.round(outputs)
                train_correct += (predicted == labels).sum().item()
                train_predictions.extend(predicted.tolist())
                train_true_labels.extend(labels.tolist())

                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pbar_train.set_postfix({'Loss': loss.item()})
                pbar_train.update()

        train_accuracy = train_correct / len(train_dataset)
        train_loss /= len(train_dataset)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            with tqdm(val_loader, desc=f"Fold {fold + 1}/{num_folds} - Epoch {epoch + 1}/{num_epochs} - Val",
                      ncols=80) as pbar_val:
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    predicted = torch.round(outputs)
                    val_correct += (predicted == labels).sum().item()
                    val_predictions.extend(predicted.tolist())
                    val_true_labels.extend(labels.tolist())

                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item() * images.size(0)
                    pbar_val.set_postfix({'Loss': loss.item()})
                    pbar_val.update()

        val_accuracy = val_correct / len(val_dataset)
        val_loss /= len(val_dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

    results = []

    # for dataset_path in datasets:
    for i, dataset in enumerate(datasets):

        # Split the dataset into training and test sets
        _, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


        # Create data loaders
        batch_size = 8
        test_loader = DataLoader(MotionArtifactDataset(test_dataset), batch_size=batch_size)

        # Test phase
        model.eval()
        test_correct = 0
        test_predictions = []
        test_true_labels = []

        with tqdm(test_loader, desc=f'Testing {datasets_path[i]}') as pbar_test:
            with torch.no_grad():
                for images, labels in pbar_test:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    predicted = torch.round(outputs)
                    test_correct += (predicted == labels).sum().item()
                    test_predictions.extend(predicted.tolist())
                    test_true_labels.extend(labels.tolist())

                    pbar_test.set_postfix({'Accuracy': test_correct / len(test_dataset)})
                    pbar_test.update()

        test_accuracy = test_correct / len(test_dataset)
        test_precision = precision_score(test_true_labels, test_predictions)
        test_recall = recall_score(test_true_labels, test_predictions)
        test_f1_score = f1_score(test_true_labels, test_predictions)
        test_auc = roc_auc_score(test_true_labels, test_predictions)
        test_confusion_matrix = confusion_matrix(test_true_labels, test_predictions)

        # Calculate Specificity and Sensitivity
        tn, fp, fn, tp = test_confusion_matrix.ravel()
        test_specificity = tn / (tn + fp)
        test_sensitivity = tp / (tp + fn)

        if i == 3:
            #ACDC Dataset
            if test_accuracy > ACDC_BEST:
                print("ACDC BEST!")
                torch.save(model.state_dict(), "YORK_model.pth")
                ACDC_BEST = test_accuracy

        results.append({
            'Dataset': datasets_path[i],
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1_score,
            'AUC': test_auc,
            'Specificity': test_specificity,
            'Sensitivity': test_sensitivity,
            'Confusion Matrix': test_confusion_matrix,
        })

        df = pd.DataFrame(results)

    # Create the results directory if it doesn't exist
    os.makedirs('./results/', exist_ok=True)
    df.to_csv(f'./results/resultFold{fold + 1}.csv', index=False)

    # Append the metrics for the current fold to the overall lists
    overall_train_losses.append(train_losses)
    overall_val_losses.append(val_losses)
    overall_train_accuracies.append(train_accuracies)
    overall_val_accuracies.append(val_accuracies)

# Compute the average metrics across all folds
avg_train_losses = np.mean(overall_train_losses, axis=0)
avg_val_losses = np.mean(overall_val_losses, axis=0)
avg_train_accuracies = np.mean(overall_train_accuracies, axis=0)
avg_val_accuracies = np.mean(overall_val_accuracies, axis=0)
