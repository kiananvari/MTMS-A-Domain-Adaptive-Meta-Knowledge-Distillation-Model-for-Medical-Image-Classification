import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sns as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, transforms
from torchvision.models import resnet18
import pandas as pd
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split


class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.resnet = resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = torch.squeeze(x, dim=1)
        x = self.sigmoid(x)
        return x

################################################################
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
        image = image.astype(np.float32)  # Convert to float

        image = cv2.resize(image, (224, 102))
        image = self.transform(image).float()
        image = np.array(image)

        return image, label

################################################################

datasets = [
    "./Datasets/dataset-ACDC.h5",
    # "./Datasets/dataset-York.h5",
    # "./Datasets/dataset-laxRecon.h5",
    "./Datasets/dataset-saxRecon.h5",
    # "./Datasets/dataset-BRAIN-MRI.h5",
    # "./Datasets/dataset-CMRxMotion.h5",
    # "./Datasets/dataset-CT.h5",

]

results = []
all_features = []

for dataset_index, dataset_path in enumerate(datasets):
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


    # # Split the dataset into training and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # Split the test set into validation and final test sets
    val_dataset, final_test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(MotionArtifactDataset(dataset), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MotionArtifactDataset(val_dataset), batch_size=batch_size)
    test_loader = DataLoader(MotionArtifactDataset(final_test_dataset), batch_size=batch_size)

    # Initialize the ResNet classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    # model = ResNetClassifier(num_classes).to(device)

    model = ResNetClassifier().to(device)
    model.load_state_dict(torch.load("./models/saxRecon_model.pth"))
    model.resnet = nn.Sequential(*list(model.resnet.children())[:-1])

    # Test phase
    model.eval()
    test_features = []

    from sklearn.decomposition import PCA

    with tqdm(test_loader, desc=f'Testing {dataset_path}') as pbar_test:
        with torch.no_grad():
            for images, labels in pbar_test:
                images = images.to(device)
                labels = labels.to(device)

                features = model(images)  # Extract features instead of making predictions
                test_features.append(features.detach().cpu().numpy())
                pbar_test.update()

            test_features = np.concatenate(test_features)  # Concatenate features from all batches

    test_features = (test_features, dataset_index)
    all_features.append(test_features)

    
# Save all_features to a local file
np.save('all_features_A_SAX.npy', all_features)

# Load all_features from the local file
all_features = np.load('all_features_A_SAX.npy', allow_pickle=True)

features0 = all_features[0][0]
features0 = features0.reshape(features0.shape[0], -1)  # Reshape to (2000, 512)

features1 = all_features[1][0]
features1 = features1.reshape(features1.shape[0], -1)  # Reshape to (2000, 512)


tsne = TSNE(n_components=2, random_state=42)
reduced_features0 = tsne.fit_transform(features0)
reduced_features1 = tsne.fit_transform(features1)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_features0[:, 0], reduced_features0[:, 1], c = 'red', label='ACDC')
# plt.scatter(reduced_features1[:, 0], reduced_features1[:, 1], c = 'blue', label='YU')
# plt.scatter(reduced_features1[:, 0], reduced_features1[:, 1], c = 'green', label='CMRxRecon(LAX)')
plt.scatter(reduced_features1[:, 0], reduced_features1[:, 1], c = 'black', label='CMRxRecon(SAX)')


plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("t-SNE Visualization of Feature Vectors")
plt.show()