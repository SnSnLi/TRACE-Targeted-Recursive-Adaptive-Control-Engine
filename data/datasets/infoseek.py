import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class InfoseekDataset(Dataset):
    """
    Dataset class for Infoseek dataset, which contains image-text pairs with
    information seeking questions and answers.
    """
    
    def __init__(self, data_path, transform=None, split='train'):
        """
        Initialize the Infoseek dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            transform (callable, optional): Transform to apply to images
            split (str): Dataset split ('train', 'val', or 'test')
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        
        # Load annotations
        self.annotations_file = os.path.join(data_path, f'infoseek_{split}.json')
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Image directory
        self.image_dir = os.path.join(data_path, 'images')
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing:
                - 'image': The image tensor
                - 'question': The question text
                - 'answer': The answer text
                - 'image_id': The image ID
        """
        ann = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, ann['image_id'] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get question and answer
        question = ann['question']
        answer = ann['answer']
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'image_id': ann['image_id']
        }
    
    def get_eval_metrics(self, predictions):
        """
        Calculate evaluation metrics for the predictions.
        
        Args:
            predictions (list): List of prediction dictionaries
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Example metrics calculation
        accuracy = 0
        for pred, gt in zip(predictions, self.annotations):
            if pred['answer'] == gt['answer']:
                accuracy += 1
        
        accuracy /= len(predictions)
        
        return {
            'accuracy': accuracy
        }
