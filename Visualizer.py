import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random

class ImageAugmentationVisualizer:
    def __init__(self, transformations=None):
        """
        Initialize the ImageAugmentationVisualizer with a predefined set of transformations.
        
        Args:
            transformations (list): A list of tuples containing transformation names and their corresponding transforms.
        """
        self.transformations = transformations or [
            ("Original", transforms.Compose([])),
            ("Random Crop", transforms.RandomCrop(180)),
            ("Random Affine", transforms.RandomAffine(5, shear=5)),
            ("Random Sharpness", transforms.RandomAdjustSharpness(2)),
            ("Random Autocontrast", transforms.RandomAutocontrast(0.33)),
            ("Resize", transforms.Resize((224, 224)))
        ]

    def visualize(self, image=None, image_path=None):
        """
        Visualize the transformations applied to an image step-by-step.

        Args:
            image (PIL.Image.Image or torch.Tensor): The input image to transform.
            image_path (str): Path to the image file (optional if `image` is provided).
        """
        if image is None and image_path is None:
            raise ValueError("Either `image` or `image_path` must be provided.")

        # Load the image if a path is provided
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")

        # Convert torch.Tensor to PIL.Image if necessary
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Apply each transformation and store the results
        transformed_images = []
        for name, transform in self.transformations:
            transformed_image = transform(image)
            transformed_images.append((name, transformed_image))

        # Plot the results
        plt.figure(figsize=(15, 5))
        for i, (name, img) in enumerate(transformed_images):
            plt.subplot(1, len(transformed_images), i + 1)
            plt.imshow(np.asarray(img))
            plt.title(name)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualizeRandom(self, dataset, label):
        """
        Visualize transformations on a randomly selected image from the dataset based on the given label.

        Args:
            dataset (torchvision.datasets): The dataset to select the image from.
            label (str): The label of the image to randomly select from the dataset.
        """
        # Get the class index for the label
        class_index = dataset.class_to_idx.get(label)
        if class_index is None:
            raise ValueError(f"Label '{label}' not found in the dataset.")
        
        # Filter images with the given label
        label_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == class_index]
        if not label_indices:
            raise ValueError(f"No images found for label '{label}'.")
        
        # Randomly select an image index
        random_index = random.choice(label_indices)
        image, _ = dataset[random_index]
        image = transforms.ToPILImage()(image)  # Convert to PIL.Image

        # Visualize the transformations
        self.visualize(image=image)
