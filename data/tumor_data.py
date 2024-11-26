import os
import imagehash
from pathlib import Path
from typing import List, Tuple, Union
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
                transforms.ToTensor()
            ])
        elif transform_pipeline == "aug":
            xform = transforms.Compose([
                transforms.Resize(image_size),  # Resize the image
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ColorJitter(0.9, 0.9, 0.9, 0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
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

def gather_data(dataset_paths: List[str]) -> Tuple[List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]]]:
    """
    Gathers image data from the specified dataset directories and loads images using PIL.
    Splits the combined data into training, validation, and testing sets.

    Args:
        dataset_paths (List[str]): A list of paths to dataset folders.

    Returns:
        Tuple[List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]], List[Tuple[Image.Image, str]]]:
            A tuple containing three lists:
            - Training data: List of tuples (image, label)
            - Validation data: List of tuples (image, label)
            - Testing data: List of tuples (image, label)
    """
    combined_data = []  # Initialize list to hold all unique image-label pairs
    image_hashes = {}   # Dictionary to track image hashes and their paths
    num_duplicates = 0

    for path in dataset_paths:
        for split in ['Training', 'Testing']:
            split_path = os.path.join(path, split)

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

    
    print(f"Loaded {len(combined_data)} images from {len(dataset_paths)} sources.")
    print(f"Duplicates found: {num_duplicates}")
    
    # Shuffle the combined data to ensure random split
    random.shuffle(combined_data)

    # Split the combined data into training, validation, and testing sets (80/10/10 split)
    total_data = len(combined_data)
    train_size = int(0.8 * total_data)  # 80% for training
    val_size = int(0.1 * total_data)    # 10% for validation
    train_set = combined_data[:train_size]  # Training data
    val_set = combined_data[train_size:train_size + val_size]  # Validation data
    test_set = combined_data[train_size + val_size:]  # Test data

    return train_set, val_set, test_set  # Return the three lists

def load_data(
    dataset_paths: List[str],
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 4,
    batch_size: int = 128,
    shuffle: bool = False,
) -> Union[DataLoader, TumorDataset]:
    """
    Constructs the dataset/dataloader.
    
    Args:
        dataset_paths (List[str]): List of paths to the datasets.
        transform_pipeline (str): 'default', 'aug', or other custom transformation pipelines
        return_dataloader (bool): returns either DataLoader or Dataset
        num_workers (int): data workers, set to 0 for VSCode debugging
        batch_size (int): batch size
        shuffle (bool): should be true for train and false for val

    Returns:
        DataLoader or TumorDataset
    """
    dataset_paths = ['data/dataset1', 'data/dataset2']

    data = gather_data(dataset_paths)
    
    # Check if data is empty or None
    if not data:
        raise ValueError("No data found. Please check the dataset paths.")
    
    train, val, test = data

    train_dataset = TumorDataset(train, transform_pipeline=transform_pipeline)
    val_dataset = TumorDataset(val, transform_pipeline=transform_pipeline)
    test_dataset = TumorDataset(test, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return train_dataset, val_dataset, test_dataset

    train_loader = DataLoader(train_dataset,num_workers=num_workers,batch_size=batch_size,shuffle=shuffle,drop_last=True,)
    val_loader = DataLoader(val_dataset,num_workers=num_workers,batch_size=batch_size,shuffle=shuffle,drop_last=True,)
    test_loader = DataLoader(test_dataset,num_workers=num_workers,batch_size=batch_size,shuffle=shuffle,drop_last=True,)

    return train_loader, val_loader, test

if __name__ == "__main__":
    dataset_paths = ['data/dataset1', 'data/dataset2']
    gathered_data = gather_data(dataset_paths)
    training, val, test = gathered_data
    print(f"{len(training)=}, {len(val)=}, {len(test)=}")
    print(f"TOTAL:{len(training) + len(val) + len(test)}")
