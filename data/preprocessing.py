import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ImagePreprocessor:
    """
    Class for preprocessing images for vision-language models.
    """
    
    def __init__(self, image_size=224, normalize=True):
        """
        Initialize the image preprocessor.
        
        Args:
            image_size (int): Size to resize images to
            normalize (bool): Whether to normalize images with ImageNet stats
        """
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, image):
        """
        Preprocess an image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        return self.transform(image)


class TextPreprocessor:
    """
    Class for preprocessing text for vision-language models.
    """
    
    def __init__(self, max_length=77, remove_stopwords=False, lowercase=True):
        """
        Initialize the text preprocessor.
        
        Args:
            max_length (int): Maximum sequence length
            remove_stopwords (bool): Whether to remove stopwords
            lowercase (bool): Whether to convert text to lowercase
        """
        self.max_length = max_length
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        if remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
    
    def __call__(self, text):
        """
        Preprocess text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase if needed
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if needed
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Truncate to max length
        tokens = tokens[:self.max_length]
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        return processed_text


def create_data_transforms(config):
    """
    Create data transforms based on configuration.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        dict: Dictionary of transforms
    """
    image_size = config.get('image_size', 224)
    normalize = config.get('normalize', True)
    max_text_length = config.get('max_text_length', 77)
    remove_stopwords = config.get('remove_stopwords', False)
    
    image_transform = ImagePreprocessor(
        image_size=image_size,
        normalize=normalize
    )
    
    text_transform = TextPreprocessor(
        max_length=max_text_length,
        remove_stopwords=remove_stopwords
    )
    
    return {
        'image': image_transform,
        'text': text_transform
    }


def augment_dataset(dataset, augmentation_config):
    """
    Apply data augmentation to a dataset.
    
    Args:
        dataset: Dataset to augment
        augmentation_config (dict): Augmentation configuration
        
    Returns:
        Augmented dataset
    """
    # Example augmentations
    if augmentation_config.get('random_crop', False):
        dataset.transform.transforms.insert(
            0, transforms.RandomCrop(
                augmentation_config.get('crop_size', 224)
            )
        )
    
    if augmentation_config.get('random_flip', False):
        dataset.transform.transforms.insert(
            0, transforms.RandomHorizontalFlip(
                augmentation_config.get('flip_prob', 0.5)
            )
        )
    
    if augmentation_config.get('color_jitter', False):
        dataset.transform.transforms.insert(
            0, transforms.ColorJitter(
                brightness=augmentation_config.get('brightness', 0.1),
                contrast=augmentation_config.get('contrast', 0.1),
                saturation=augmentation_config.get('saturation', 0.1),
                hue=augmentation_config.get('hue', 0.1)
            )
        )
    
    return dataset

