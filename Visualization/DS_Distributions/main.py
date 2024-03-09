import cv2
import h5py
import numpy as np
import sns as sns
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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
    "./Datasets/dataset-York.h5",
    "./Datasets/dataset-laxRecon.h5",
    "./Datasets/dataset-saxRecon.h5",
    "./Datasets/dataset-BRAIN-MRI.h5",
    "./Datasets/dataset-CMRxMotion.h5",
    "./Datasets/dataset-CT.h5",

]

results = []
all_features = []

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

    ds_features = []


    # Create the PCA object
    pca = PCA(n_components=1)  # Set the number of components as desired

    with tqdm(test_loader, desc=f'Testing {dataset_path}') as pbar_test:
        with torch.no_grad():
            for images, labels in pbar_test:
                images = images.to(device)
                labels = labels.to(device)

                # Flatten the images
                flattened_images = images.view(images.size(0), -1)  # Flatten each image

                # Apply PCA on flattened images
                features = pca.fit_transform(flattened_images.cpu().numpy())

                ds_features.append(features)
                pbar_test.update()

            test_features = np.concatenate(ds_features)  # Concatenate features from all batches

    all_features.append(ds_features)

print(len(all_features))
print(all_features[0].shape)

# Save all_features to a local file
np.save('all_features.npy', all_features)

