import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms
from torchvision.models import resnet18, resnet101, resnet50, ResNet18_Weights
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from models import ResNetClassifier, ResNetClassifier1

################################################################

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


################################################################


datasets = [
    "./datasets/dataset-ACDC.h5",
    "./datasets/dataset-York.h5",
    "./datasets/dataset-laxRecon.h5",
    "./datasets/dataset-saxRecon.h5"
]


# Function to split the test dataset into 5 parts
def split_test_dataset(test_dataset):
    test_parts = []
    test_size = len(test_dataset) // 5

    for i in range(4):
        start_index = i * test_size
        end_index = (i + 1) * test_size
        test_part = test_dataset[start_index:end_index]
        test_parts.append(test_part)

    # Add the remaining samples to the last part
    test_part = test_dataset[(4 * test_size):]
    test_parts.append(test_part)

    return test_parts

results = []

for dataset_path in datasets:
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

    # Split the dataset into training and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Split the test set into 5 parts
    test_parts = split_test_dataset(test_dataset)

    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(MotionArtifactDataset(train_dataset), batch_size=batch_size, shuffle=True)

    # Initialize the ResNet classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetClassifier1().to(device)
    model.resnet = nn.Sequential(*list(model.resnet.children())[:-1])
    model.load_state_dict(torch.load("fine_tuned_student_model_64Q_DA1_FL1_40_1_20F.pth"))

    # Calculate metrics for each part of the test dataset
    test_metrics_parts = []

    for test_part in test_parts:
        test_loader = DataLoader(MotionArtifactDataset(test_part), batch_size=batch_size)
        model.eval()
        test_correct = 0
        test_predictions = []
        test_true_labels = []

        with tqdm(test_loader, desc=f'Testing {dataset_path}') as pbar_test:
            with torch.no_grad():
                for images, labels in pbar_test:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    outputs = torch.squeeze(outputs)

                    predicted = torch.round(outputs)
                    test_correct += (predicted == labels).sum().item()
                    test_predictions.extend(predicted.tolist())
                    test_true_labels.extend(labels.tolist())

                    pbar_test.set_postfix({'Accuracy': test_correct / len(test_part)})
                    pbar_test.update()

        test_accuracy = test_correct / len(test_part)
        test_precision = precision_score(test_true_labels, test_predictions)
        test_recall = recall_score(test_true_labels, test_predictions)
        test_f1_score = f1_score(test_true_labels, test_predictions)
        test_auc = roc_auc_score(test_true_labels, test_predictions)
        test_confusion_matrix = confusion_matrix(test_true_labels, test_predictions)

        # Calculate Specificity and Sensitivity
        tn, fp, fn, tp = test_confusion_matrix.ravel()
        test_specificity = tn / (tn + fp)
        test_sensitivity = tp / (tp + fn)

        test_metrics_parts.append({
            'Accuracy': test_accuracy,
            'Precision': test_precision,
            'Recall': test_recall,
            'F1 Score': test_f1_score,
            'AUC': test_auc,
            'Specificity': test_specificity,
            'Sensitivity': test_sensitivity,
            'Confusion Matrix': test_confusion_matrix,
        })

    # Calculate mean and standard deviation of metrics for each dataset
    test_accuracy_mean = np.mean([metrics['Accuracy'] for metrics in test_metrics_parts])
    test_accuracy_std = np.std([metrics['Accuracy'] for metrics in test_metrics_parts])
    test_precision_mean = np.mean([metrics['Precision'] for metrics in test_metrics_parts])
    test_precision_std = np.std([metrics['Precision'] for metrics in test_metrics_parts])
    test_recall_mean = np.mean([metrics['Recall'] for metrics in test_metrics_parts])
    test_recall_std = np.std([metrics['Recall'] for metrics in test_metrics_parts])
    test_f1_score_mean = np.mean([metrics['F1 Score'] for metrics in test_metrics_parts])
    test_f1_score_std = np.std([metrics['F1 Score'] for metrics in test_metrics_parts])
    test_auc_mean = np.mean([metrics['AUC'] for metrics in test_metrics_parts])
    test_auc_std = np.std([metrics['AUC'] for metrics in test_metrics_parts])
    test_specificity_mean = np.mean([metrics['Specificity'] for metrics in test_metrics_parts])
    test_specificity_std = np.std([metrics['Specificity'] for metrics in test_metrics_parts])
    test_sensitivity_mean = np.mean([metrics['Sensitivity'] for metrics in test_metrics_parts])
    test_sensitivity_std = np.std([metrics['Sensitivity'] for metrics in test_metrics_parts])

    results.append({
        'Dataset': dataset_path,
        'Accuracy': f'{test_accuracy_mean:.5f}({test_accuracy_std:.5f})',
        'Precision': f'{test_precision_mean:.5f}({test_precision_std:.5f})',
        'Recall': f'{test_recall_mean:.5f}({test_recall_std:.5f})',
        'F1 Score': f'{test_f1_score_mean:.5f}({test_f1_score_std:.5f})',
        'AUC': f'{test_auc_mean:.5f}({test_auc_std:.5f})',
        'Specificity': f'{test_specificity_mean:.5f}({test_specificity_std:.5f})',
        'Sensitivity': f'{test_sensitivity_mean:.5f}({test_sensitivity_std:.5f})',
        'Confusion Matrix': test_metrics_parts[0]['Confusion Matrix'],  # Use the first part's confusion matrix
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
df.to_csv('64Q_DA1_FL1_40_1_20F.csv', index=False)

# Display the results
print(df)
