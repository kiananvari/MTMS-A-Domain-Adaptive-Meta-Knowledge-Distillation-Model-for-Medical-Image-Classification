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

from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import DA_Model

#####################################################################

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

for domain_label, dataset_path in enumerate(datasets_path):

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
                class_label = int(labels_dataset[i])

                dataset.append((image, class_label, domain_label))

                pbar.update(1)

    datasets.append(dataset)

#####################################################################

DA_dataset = []

# Containing York, laxRecoon, saxRecon datasets
for dataset in datasets[:3]:
    for item in dataset:
        DA_dataset.append(item)


print(len(DA_dataset))


#####################################################################

class MotionArtifactDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, class_label, domain_label = self.data[index]
        image = (image - image.min()) / (image.max() - image.min())  # Normalize between 0 and 1
        image = image * 255  # Scale to 0-255 range
        image = cv2.resize(image, (224, 102))
        image = self.transform(image).float()
        image = np.array(image)

        return image, class_label, domain_label


################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
################################################################

trainval_dataset, test_dataset = train_test_split(DA_dataset, test_size=0.2, random_state=42)

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
    batch_size = 8

    train_loader = DataLoader(MotionArtifactDataset(train_dataset), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MotionArtifactDataset(val_dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MotionArtifactDataset(test_dataset), batch_size=batch_size, shuffle=True)

    # Initialize the model and other variables for the current fold
    model = DA_Model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    # criterion = nn.BCELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    def calculate_accuracy(output, target):
        _, predicted = torch.max(output, dim=1)
        correct = (predicted == target).sum().item()
        accuracy = correct / target.size(0)
        return accuracy


    # Initialize empty lists to store the losses and accuracies
    train_losses_class = []
    train_accs_class = []
    val_losses_class = []
    val_accs_class = []

    num_epochs = 4  # Adjust the number of epochs as needed

    for epoch in range(num_epochs):

        model.train()
        train_loss_class = 0.0
        train_loss_domain = 0.0
        train_acc_class = 0.0

        with tqdm(total=len(train_loader), desc=f"Fold {fold+1}/{num_folds}, Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (images, class_label, domain_label) in enumerate(train_loader):
                p = float(i + epoch * len(train_loader)) / (num_epochs * len(train_loader))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                images = images.to(device)
                class_label = class_label.to(device)
                domain_label = domain_label.to(device)

                optimizer.zero_grad()

                # print(f"Class label: {class_label}")
                # print(f"Domain label: {domain_label}")
                class_output, domain_output = model(images, alpha=alpha)
                # print(f"Class output: {class_output}")
                # print(f"Domain output: {domain_output}")

                err_s_label = loss_class(class_output, class_label)
                err_s_domain = loss_domain(domain_output, domain_label)

                err = err_s_label + err_s_domain
                # err = err_s_label
                err.backward()
                optimizer.step()

                train_loss_class += err_s_label.item()
                train_loss_domain += err_s_domain.item()
                train_acc_class += calculate_accuracy(class_output, class_label)

                pbar.update(1)
                pbar.set_postfix({
                    "class_loss": train_loss_class / (i + 1),
                    "domain_loss": train_loss_domain / (i + 1),
                    "class_acc": train_acc_class / (i + 1)
                })

            train_loss_class /= len(train_loader)
            train_loss_domain /= len(train_loader)
            train_acc_class /= len(train_loader)

            train_losses_class.append(train_loss_class)
            train_accs_class.append(train_acc_class)

            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"Training - Class Loss: {train_loss_class:.4f}, Class Accuracy: {train_acc_class:.4f}")

        model.eval()

        val_loss_class = 0.0
        val_loss_domain = 0.0
        val_acc_class = 0.0

        alpha = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Fold {fold+1}/{num_folds}, Epoch {epoch + 1}/{num_epochs}") as pbar:
                for i, (images, class_label, domain_label) in enumerate(val_loader):
                    images = images.to(device)
                    class_label = class_label.to(device)
                    domain_label = domain_label.to(device)

                    class_output, domain_output = model(images, alpha=alpha)

                    err_s_label = loss_class(class_output, class_label)
                    err_s_domain = loss_domain(domain_output, domain_label)

                    val_loss_class += err_s_label.item()
                    val_loss_domain += err_s_domain.item()
                    val_acc_class += calculate_accuracy(class_output, class_label)

                    pbar.update(1)
                    pbar.set_postfix({
                        "valid_class_loss": val_loss_class / (i + 1),
                        "valid_domain_loss": val_loss_domain / (i + 1),
                        "valid_class_acc": val_acc_class / (i + 1)
                    })

            val_loss_class /= len(val_loader)
            val_loss_domain /= len(val_loader)
            val_acc_class /= len(val_loader)

            val_losses_class.append(val_loss_class)
            val_accs_class.append(val_acc_class)

            print(f"Validation - Class Loss: {val_loss_class:.4f}, Class Accuracy: {val_acc_class:.4f}")


    #  Evaluate the model on the test set for the current fold

    results = []

    # for dataset_path in datasets:
    for i, dataset in enumerate(datasets):

        # Split the dataset into training and test sets
        train_dataset, final_test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

        # Split the test set into validation and final test sets
        #val_dataset, final_test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

        # Create data loaders
        batch_size = 8
        train_loader = DataLoader(MotionArtifactDataset(train_dataset), batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(MotionArtifactDataset(val_dataset), batch_size=batch_size)
        test_loader = DataLoader(MotionArtifactDataset(final_test_dataset), batch_size=batch_size)

        # Test phase
        model.eval()
        test_correct = 0
        test_predictions = []
        test_true_labels = []

        with tqdm(test_loader, desc=f'Testing {datasets_path[i]}') as pbar_test:
            with torch.no_grad():
                for (images, class_label, domain_label) in pbar_test:
                    images = images.to(device)
                    class_label = class_label.to(device)
                    domain_label = domain_label.to(device)

                    class_output, _ = model(images, alpha=alpha)

                    _, predicted = torch.max(class_output, dim=1)
                    test_correct += (predicted == class_label).sum().item()
                    test_predictions.extend(predicted.tolist())
                    test_true_labels.extend(class_label.tolist())

                    pbar_test.set_postfix({'Accuracy': test_correct / len(final_test_dataset)})
                    pbar_test.update()

        test_accuracy = test_correct / len(final_test_dataset)
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
                torch.save(model.state_dict(), "DA_model.pth")
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

