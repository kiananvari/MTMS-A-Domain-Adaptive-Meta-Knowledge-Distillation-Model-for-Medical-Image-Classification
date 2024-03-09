import os

import cv2
import h5py
import numpy as np
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
import random

from models import ResNetClassifier, ResNetClassifier1, DA_Model, fa_selector, fa_selector1

# Load dataset in two query_set, support_set
dataset = []
support_set = []
query_set = []

with h5py.File('./Datasets/dataset-ACDC.h5', 'r') as h5_file:
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


##################################################################

class MotionArtifactDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]

        image, label = data
        image = (image - image.min()) / (image.max() - image.min()) * 255  # Normalize between 0 and 1
        image = cv2.resize(image, (224, 102))
        image = self.transform(image).float()
        image = np.array(image)
        return image, label


##################################################################

expert_models = [
    "./models/lax_model.pth",
    "./models/sax_model.pth",
    "./models/York_model.pth",
    "./models/DA_model.pth",
]


def get_features(image_set, image_labels=None, query_set_input=False):
    feature_list = []

    # INPUT: D_Unlabeld
    # Without FILTER 
    if query_set_input == False:

        for expert_model in expert_models:

            # DA_Teacher
            if expert_model == expert_models[3]:

                model = DA_Model().to(device)
                model.load_state_dict(torch.load(expert_model))

                with torch.no_grad():
                    # Remove the last fully connected (fc) layer
                    _, features_vector = model(image_set)
                    features_vector = torch.squeeze(features_vector)

                feature_list.append(features_vector)

            # ORDINARY_Teacher
            else:
                model = ResNetClassifier().to(device)
                model.load_state_dict(torch.load(expert_model))

                with torch.no_grad():
                    # Remove the last fully connected (fc) layer
                    model.resnet = nn.Sequential(*list(model.resnet.children())[:-1])
                    features_vector = model(image_set)
                    features_vector = torch.squeeze(features_vector)

                feature_list.append(features_vector)

    # INPUT: D_Labeld
    # With FILTER 
    else:
        for expert_model in expert_models:

            # DA_Teacher
            if expert_model == expert_models[3]:

                model = DA_Model().to(device)
                model.load_state_dict(torch.load(expert_model))

                with torch.no_grad():

                    class_output, features_vector = model(image_set)
                    features_vector = torch.squeeze(features_vector)

                    if len(features_vector.shape) == 1:
                        features_vector = features_vector.unsqueeze(0)



                    #FILTER
                    # Calculate predictions
                    _, predicted_labels = torch.max(class_output, 1)
                    # Iterate over each batch
                    for i in range(len(image_labels)):
                        if predicted_labels[i] == image_labels[i]:
                            features_vector[i] = features_vector[i]
                        else:
                            #FILTER MASK
                            features_vector[i] = torch.zeros(512)


                feature_list.append(features_vector)

            # ORDINARY_Teacher
            else:
                model = ResNetClassifier().to(device)
                model.load_state_dict(torch.load(expert_model))

                with torch.no_grad():

                    class_output = model(image_set)
                    # Remove the last fully connected (fc) layer
                    model.resnet = nn.Sequential(*list(model.resnet.children())[:-1])
                    features_vector = model(image_set)
                    features_vector = torch.squeeze(features_vector)

                    if len(features_vector.shape) == 1:
                        features_vector = features_vector.unsqueeze(0)

                    #FILTER
                    # Calculate predictions
                    predicted_labels = torch.round(class_output)
                    # Iterate over each batch
                    for i in range(len(image_labels)):
                        if predicted_labels[i] == image_labels[i]:
                            features_vector[i] = features_vector[i]
                        else:
                            #FILTER MASK
                            features_vector[i] = torch.zeros(512)

                feature_list.append(features_vector)

    return feature_list

##################################################################

# Split the dataset into training and test sets
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Add Query set to train_dataset
train_dataset.extend(query_set)
random.shuffle(train_dataset)

# Split the test set into validation and final test sets
val_dataset, final_test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

# Initialize the ResNet classifier
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MotionArtifactDataset(train_dataset)
val_dataset = MotionArtifactDataset(val_dataset)
final_test_dataset = MotionArtifactDataset(final_test_dataset)

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(final_test_dataset, batch_size=batch_size, shuffle=True)

################## MODELS ####################

aggregator_model = fa_selector(dim=512, depth=1, heads=16, mlp_dim=1024).to(device)

student_model = ResNetClassifier().to(device)
student_model.resnet = nn.Sequential(*list(student_model.resnet.children())[:-1])

# Define the loss function and optimizer

#D_Unlabeld 
support_set_loss = nn.MSELoss()
#D_labeld 
query_set_loss = nn.MSELoss()

optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

query = 0
num_epochs = 40


for epoch in range(num_epochs):

    student_model.train()

    optimizer_student.zero_grad()

    epoch_support_loss = 0
    epoch_query_loss = 0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):

        # Split Sets
        support_set_images = []
        support_set_labels = []
        query_set_images = []
        query_set_labels = []
        for image, label in zip(images, labels):
            if label == -1:
                support_set_images.append(image)
                support_set_labels.append(label)
            else:
                query_set_images.append(image)
                query_set_labels.append(label)

        ###### Support Set ##### D_Unlabeld
        support_set_images = torch.stack(support_set_images).to(device)

        features = get_features(support_set_images)

        features = torch.stack(features, dim=-1)
        features = features.permute((0, 2, 1))

        aggregator_out = aggregator_model(features)

        student_output = student_model(support_set_images)
        student_output = torch.squeeze(student_output)

        support_cost = support_set_loss(student_output, aggregator_out)

        ###### Query Set Loss ##### D_labeld
        if len(query_set_images) != 0:
            query_set_images = torch.stack(query_set_images).to(device)
            query_set_labels = torch.stack(query_set_labels).to(device)

            features = get_features(query_set_images, image_labels=query_set_labels, query_set_input=True)
            features = torch.stack(features, dim=-1)

            features = features.permute((0, 2, 1))

            aggregator_out = aggregator_model(features)

            student_output = student_model(query_set_images)
            student_output = torch.squeeze(student_output)

            if len(student_output.shape) == 1:
                student_output = student_output.unsqueeze(0)

            query_cost = query_set_loss(student_output, aggregator_out)

            epoch_query_loss += query_cost.item()

        else:
            query_cost = 0
            epoch_query_loss += 0

        support_cost.backward()
        if query_cost != 0:
            query_cost.backward()

        error = support_cost + query_cost

        optimizer_student.step()

        # Print support_error and query_error as prefixes
        tqdm.write(f"Batch {i + 1}/{len(train_loader)} - support_error: {support_cost:.4f}, query_error: {query_cost:.4f}, error:{error:.4f}")

    # End of epoch
    student_tuned_model = ResNetClassifier1().to(device)

    student_tuned_model.resnet = student_model.resnet

    if os.path.exists('./student_tuned_model.pth'):
        student_tuned_model.load_state_dict(torch.load("student_tuned_model.pth"))

    student_tuned_model.resnet = student_model.resnet
    optimizer_student_tuned_model = optim.Adam(student_tuned_model.parameters(), lr=0.001)

    aggregator_tuned_model = fa_selector1(dim=512, depth=1, heads=16, mlp_dim=1024).to(device)
    aggregator_tuned_model.transformer = aggregator_model.transformer
    aggregator_tuned_model.mlp = aggregator_model.mlp
    optimizer_aggregator_tuned_model = optim.Adam(aggregator_tuned_model.parameters(), lr=0.001)

    tune_batch_size = 8
    tune_dataset = MotionArtifactDataset(query_set)
    tune_loader = DataLoader(tune_dataset, batch_size=tune_batch_size, shuffle=True)

    agg_criterion = nn.BCELoss()
    criterion = nn.BCELoss()

    tune_epochs = 1
    best_train_loss = float('inf')  # Initialize with a high value

    for tune_epoch in range(tune_epochs):
        # Training phase

        train_loss = 0.0
        train_correct = 0

        train_predictions = []
        train_true_labels = []

        with tqdm(tune_loader, desc=f"Epoch {tune_epoch + 1}/{tune_epochs} - Updating Agg-Student",
                  ncols=80) as pbar_train:
            for images, labels in tune_loader:

                student_tuned_model.train()
                aggregator_tuned_model.train()

                query_set_images = []
                query_set_labels = []

                for image, label in zip(images, labels):
                    if label == -1:
                        support_set_images.append(image)
                        support_set_labels.append(label)
                    else:
                        query_set_images.append(image)
                        query_set_labels.append(label)

                images = images.to(device)
                labels = labels.to(device)
                query_set_images = torch.stack(query_set_images).to(device)
                query_set_labels = torch.stack(query_set_labels).to(device)

                features = get_features(query_set_images, image_labels=query_set_labels, query_set_input=True)
                features = torch.stack(features, dim=-1)

                features = features.permute((0, 2, 1))

                student_tuned_model.zero_grad()
                aggregator_tuned_model.zero_grad()

                agg_outputs = aggregator_tuned_model(features)
                agg_outputs = torch.squeeze(agg_outputs)

                outputs = student_tuned_model(images)
                outputs = torch.squeeze(outputs)

                predicted = torch.round(outputs)
                train_correct += (predicted == labels).sum().item()
                train_predictions.extend(predicted.tolist())
                train_true_labels.extend(labels.tolist())

                agg_loss = agg_criterion(agg_outputs, labels.float())
                loss = criterion(outputs, labels.float())

                agg_loss.backward()
                loss.backward()

                optimizer_aggregator_tuned_model.step()
                optimizer_student_tuned_model.step()

                train_loss += loss.item() * images.size(0)
                pbar_train.set_postfix({'Loss': loss.item()})
                pbar_train.update()

        train_accuracy = train_correct / len(tune_dataset)
        train_loss /= len(tune_dataset)

        # Check if the current train loss is lower than the previous best train loss
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(student_tuned_model.state_dict(), "student_tuned_model.pth")

        # Append accuracy and loss to the respective lists
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(f"Epoch {tune_epoch + 1}/{tune_epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f}")

    aggregator_model.transformer = aggregator_tuned_model.transformer
    aggregator_model.mlp = aggregator_tuned_model.mlp

    student_tuned_model.load_state_dict(torch.load("student_tuned_model.pth"))
    student_model.resnet = student_tuned_model.resnet

torch.save(student_model.state_dict(), "student_model_64Q_DA1_FL1_40_1.pth")
