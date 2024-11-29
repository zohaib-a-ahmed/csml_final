import os
import imagehash
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Counter, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import random

# Define the label names based on the subdirectory names
LABEL_NAMES = ["notumor", "glioma", "meningioma", "pituitary"] # 0, 1, 2, 3

class TumorDataset(Dataset):
    ''' Brain Tumor dataset for classification '''

    def __init__(
        self,
        data: List[Tuple[Image.Image, str]],
        transform_pipeline: str = "default",
        image_size: Tuple[int, int] = (224, 224)  # Default size for resizing
    ):
        self.transform = self.get_transform(transform_pipeline, image_size)
        self.data = []
        
        # Process the gathered data
        for img, label in data:
            label_id = LABEL_NAMES.index(label)  # Get the label ID based on the order in LABEL_NAMES
            self.data.append((img, label_id))

    def get_transform(self, transform_pipeline: str = "default", image_size: Tuple[int, int] = (224, 224)):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.Compose([
                transforms.Resize(image_size),  # Resize the image
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        elif transform_pipeline == "aug":
            xform = transforms.Compose([
                transforms.Resize(image_size),  # Resize the image
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Pairs of images and labels (int) for classification
        """
        img, label_id = self.data[idx]
        data = (self.transform(img), label_id)

        return data

def gather_training_data(dataset_paths: List[str]) -> Tuple[List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]]]:
    """
    Gathers training image data from the specified dataset directories and loads images using PIL.
    Returns the training and validation sets.

    Args:
        dataset_paths (List[str]): A list of paths to dataset folders.

    Returns:
        Tuple[List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]]]:
            A tuple containing two lists:
            - Training data: List of tuples (image, label)
            - Validation data: List of tuples (image, label)
    """
    combined_data = []  # Initialize list to hold all unique image-label pairs
    image_hashes = {}   # Dictionary to track image hashes and their paths
    num_duplicates = 0

    for path in dataset_paths:
        split_path = os.path.join(path, 'Training')  # Only use 'Training' directory

        # Check if split path exists
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist. Skipping.")
            continue
        
        # Iterate through each class folder (glioma, meningioma, etc.)
        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            
            # Check if class path is a directory
            if not os.path.isdir(class_path):
                continue
            
            # Gather all jpg images in the class folder
            for image_file in os.listdir(class_path):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(class_path, image_file)
                    try:
                        with Image.open(image_path) as img:
                            image_hash = imagehash.phash(img)

                        # Check for duplicates
                        if image_hash not in image_hashes:
                            # Load the image using PIL
                            image = Image.open(image_path)
                            image_hashes[image_hash] = (image, class_folder, image_path)  # Store image, label, and path
                            combined_data.append((image, class_folder))  # Append (image, label) to combined data
                        else:
                            num_duplicates += 1

                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    print(f"Loaded {len(combined_data)} training images. Removed {num_duplicates} duplicates.")

    # Split the combined data into training and validation sets (80/20 split)
    train_data, val_data = train_test_split(combined_data, test_size=0.2, random_state=42)

    return train_data, val_data

def gather_testing_data(dataset_paths: List[str]) -> List[Tuple[Image.Image, str]]:
    """
    Gathers testing image data from the specified dataset directories and loads images using PIL.

    Args:
        dataset_paths (List[str]): A list of paths to dataset folders.

    Returns:
        List[Tuple[Image.Image, str]]:
            A list of tuples containing test data (image, label).
    """
    combined_data = []  # Initialize list to hold all unique image-label pairs
    image_hashes = {}   # Dictionary to track image hashes and their paths
    num_duplicates = 0

    for path in dataset_paths:
        split_path = os.path.join(path, 'Testing')  # Only use 'Testing' directory

        # Check if split path exists
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist. Skipping.")
            continue
        
        # Iterate through each class folder (glioma, meningioma, etc.)
        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)
            
            # Check if class path is a directory
            if not os.path.isdir(class_path):
                continue
            
            # Gather all jpg images in the class folder
            for image_file in os.listdir(class_path):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(class_path, image_file)
                    try:
                        with Image.open(image_path) as img:
                            image_hash = imagehash.phash(img)

                        # Check for duplicates
                        if image_hash not in image_hashes:
                            # Load the image using PIL
                            image = Image.open(image_path)
                            image_hashes[image_hash] = (image, class_folder, image_path)  # Store image, label, and path
                            combined_data.append((image, class_folder))  # Append (image, label) to combined data
                        else:
                            num_duplicates += 1

                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    print(f"Loaded {len(combined_data)} testing images. Removed {num_duplicates} duplicates.")
    return combined_data

def load_training_data(
    dataset_paths: List[str],
    transform_pipeline: str = "default",
    num_workers: int = 4,
    batch_size: int = 128,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Loads training and validation data loaders.

    Args:
        dataset_paths (List[str]): List of paths to the datasets.
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    train_data, val_data = gather_training_data(dataset_paths)

    train_dataset = TumorDataset(train_data, transform_pipeline=transform_pipeline)
    val_dataset = TumorDataset(val_data, transform_pipeline=transform_pipeline)

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

def load_testing_data(
    dataset_paths: List[str],
    transform_pipeline: str = "default",
    num_workers: int = 4,
    batch_size: int = 128,
) -> DataLoader:
    """
    Loads testing data loader.

    Args:
        dataset_paths (List[str]): List of paths to the datasets.
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size

    Returns:
        DataLoader: Testing data loader.
    """
    test_data = gather_testing_data(dataset_paths)
    
    if not test_data:
        raise ValueError("No test data found. Please check the dataset paths.")

    test_dataset = TumorDataset(test_data, transform_pipeline=transform_pipeline)
    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, drop_last=True)

    return test_loader

def print_class_distribution(train_set, val_set, test_set):
    """
    Prints the class distribution for training, validation, and testing datasets.

    Args:
        train_set (List[Tuple[Image.Image, str]]): Training dataset.
        val_set (List[Tuple[Image.Image, str]]): Validation dataset.
        test_set (List[Tuple[Image.Image, str]]): Testing dataset.
    """
    # Count the occurrences of each class in the datasets
    train_labels = [label for _, label in train_set]
    val_labels = [label for _, label in val_set]
    test_labels = [label for _, label in test_set]

    train_distribution = Counter(train_labels)
    val_distribution = Counter(val_labels)
    test_distribution = Counter(test_labels)

    # Print the distributions
    print("Training Set Class Distribution:")
    for label, count in train_distribution.items():
        print(f"{label}: {count}")

    print("\nValidation Set Class Distribution:")
    for label, count in val_distribution.items():
        print(f"{label}: {count}")

    print("\nTesting Set Class Distribution:")
    for label, count in test_distribution.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    dataset_paths = ['data/dataset1', 'data/dataset2']
    # gathered_data = gather_data(dataset_paths)
    # training, val, test = gathered_data
    
    # Example usage after calling gather_data
    train, val = gather_training_data(dataset_paths)
    print(f"{len(train)=} {len(val)=}")
    t, v = load_training_data(dataset_paths)

    test = gather_testing_data(dataset_paths)
    print(f"{len(test)=}")
    
