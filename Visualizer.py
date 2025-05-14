import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random

# Add this near the top of your script, after imports
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageAugmentationVisualizer:
    def __init__(self, transformations=None):
        """
        Initialize the ImageAugmentationVisualizer with a predefined set of transformations.
        
        Args:
            transformations (list): A list of tuples containing transformation names and their corresponding transforms.
        """
        self.transformations = transformations or [
            ("Original", transforms.Compose([])),
            ("Resize", transforms.Resize((224, 224))),
            ("Random Crop", transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop(180)
            ])),
            ("Random Affine", transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(5, shear=5)
            ])),
            ("Random Sharpness", transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAdjustSharpness(2)
            ])),
            ("Random Autocontrast", transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAutocontrast(0.33)
            ])),
        ]

    @staticmethod
    def split_dataset_by_category(dataset):
        """
        Splits the dataset into separate lists, one for each category.
        Returns a dict: {class_name: [(image, label), ...], ...}
        """
        class_names = dataset.classes
        category_dict = {name: [] for name in class_names}
        for img, lbl in dataset:
            class_name = class_names[lbl]
            category_dict[class_name].append((img, lbl))
        return category_dict

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

    def visualize_grid(self, image):
        """
        Show the original image and 8 augmented versions in a 3x3 grid.
        Args:
            image (PIL.Image.Image or torch.Tensor): The input image to transform.
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Define the 8 augmentations (each includes resizing)
        augmentations = [
            ("Original", transforms.Compose([transforms.Resize((224, 224))])),
            ("Random Crop", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomCrop(180), transforms.Resize((224, 224))])),
            ("Random Affine", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomAffine(5, shear=5)])),
            ("Random Sharpness", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomAdjustSharpness(2)])),
            ("Random Autocontrast", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomAutocontrast(0.33)])),
            ("Horizontal Flip", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0)])),
            ("Vertical Flip", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomVerticalFlip(p=1.0)])),
            ("Color Jitter", transforms.Compose([transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)])),
            ("Grayscale", transforms.Compose([transforms.Resize((224, 224)), transforms.RandomGrayscale(p=1.0)])),
        ]

        # Apply each augmentation
        images = []
        for name, aug in augmentations:
            img_aug = aug(image)
            images.append((name, img_aug))

        # Plot in a 3x3 grid
        plt.figure(figsize=(10, 10))
        for i, (name, img) in enumerate(images):
            plt.subplot(3, 3, i + 1)
            plt.imshow(np.asarray(img))
            plt.title(name)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def visualize_all_effects(self, image, label=None):
        """
        Apply all augmentations in sequence to the same image and display the result.
        Args:
            image (PIL.Image.Image or torch.Tensor): The input image to transform.
            label (str, optional): The category label to display.
        """
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        # Compose all augmentations in sequence
        all_effects = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(180),
            transforms.Resize((224, 224)),
            transforms.RandomAffine(5, shear=5),
            transforms.RandomAdjustSharpness(2),
            transforms.RandomAutocontrast(0.33),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomGrayscale(p=1.0),
        ])

        effected_img = all_effects(image)

        # Show original and fully-augmented image side by side, with label
        plt.figure(figsize=(8, 4))
        if label:
            plt.suptitle(f"Category: {label}", fontsize=16)
        plt.subplot(1, 2, 1)
        plt.imshow(np.asarray(image))
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(np.asarray(effected_img))
        plt.title("All Effects")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def augment_data(data, device='cpu'):
        augmented_data = torch.empty((data.size()[0], 3 * 9, 224, 224), device=device)
        for i in range(0, data.size()[0]):
            for j in range(0, 8):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),  
                    transforms.RandomCrop(180),
                    transforms.RandomAffine(5, shear=5),
                    transforms.RandomAdjustSharpness(2),
                    transforms.RandomAutocontrast(0.33),
                    transforms.Resize((224, 224))
                ])
                transformed = transform(data[i])
                augmented_data[i, j*3] = transformed[0]
                augmented_data[i, j*3 + 1] = transformed[1]
                augmented_data[i, j*3 + 2] = transformed[2]
        return augmented_data

    def visualize_resnet_augment(self, image, label=None, device='cpu'):
        """
        Visualize the effect of the ResNet augment_data function on a single image.
        Shows the original and all 8 augmented versions in a grid.
        """
        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        image = image.unsqueeze(0).to(device)
        augmented = self.augment_data(image, device=device)
        imgs = torch.chunk(augmented[0], 9, dim=0)  # 9 images total

        plt.figure(figsize=(10, 10))
        if label:
            plt.suptitle(f"Category: {label}", fontsize=16)
        # Show original
        plt.subplot(3, 3, 1)
        plt.imshow(transforms.ToPILImage()(image.squeeze(0).cpu()))
        plt.title("Original")
        plt.axis("off")
        # Show 8 augmentations
        for i, img in enumerate(imgs[:8]):  # Only first 8 augmentations
            plt.subplot(3, 3, i + 2)
            plt.imshow(transforms.ToPILImage()(img.cpu()))
            plt.title(f"Aug {i+1}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


def visualize_grid_for_all_categories(visualizer, dataset):
    """
    For each category in the dataset, pick a random image and show the original and 8 augmentations in a 3x3 grid.
    """
    class_names = dataset.classes
    for label in class_names:
        print(f"Visualizing grid for label: {label}")
        class_index = dataset.class_to_idx[label]
        label_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == class_index]
        random_index = random.choice(label_indices)
        image, _ = dataset[random_index]
        image = transforms.ToPILImage()(image)
        visualizer.visualize_grid(image)


from Visualizer import ImageAugmentationVisualizer
from torchvision import datasets, transforms
import random

# Load your dataset (no transform needed for visualization)
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Initialize the visualizer
visualizer = ImageAugmentationVisualizer()

# Get all class names
class_names = dataset.classes

# Splits into dictionaries by categories
category_dict = visualizer.split_dataset_by_category(dataset)

# for class_name, img_list in category_dict.items():
#     if img_list:  # Make sure the list is not empty
#         image, _ = random.choice(img_list)
#         visualizer.visualize_all_effects(image, label=class_name)

# # Example usage after creating your visualizer and category_dict:
# for class_name, img_list in category_dict.items():
#     if img_list:
#         image, _ = random.choice(img_list)
#         visualizer.visualize_resnet_augment(image, label=class_name, device=device)