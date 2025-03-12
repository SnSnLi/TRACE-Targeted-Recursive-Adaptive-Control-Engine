import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Flickr30kDataset(Dataset):
    """
    Dataset class for Flickr30k dataset, which contains images with multiple text captions.
    """
    
    def __init__(self, data_path, transform=None, split='train', max_captions=5):
        """
        Initialize the Flickr30k dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            transform (callable, optional): Transform to apply to images
            split (str): Dataset split ('train', 'val', or 'test')
            max_captions (int): Maximum number of captions per image
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        self.max_captions = max_captions
        
        # Load annotations
        self.annotations_file = os.path.join(data_path, f'flickr30k_{split}.json')
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Image directory
        self.image_dir = os.path.join(data_path, 'flickr30k-images')
        
        # Create image-caption pairs
        self.pairs = []
        for ann in self.annotations:
            image_id = ann['image_id']
            for i, caption in enumerate(ann['captions']):
                if i >= self.max_captions:
                    break
                self.pairs.append({
                    'image_id': image_id,
                    'caption': caption
                })
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing:
                - 'image': The image tensor
                - 'caption': The caption text
                - 'image_id': The image ID
        """
        pair = self.pairs[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, pair['image_id'])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'caption': pair['caption'],
            'image_id': pair['image_id']
        }
    
    def get_eval_metrics(self, predictions):
        """
        Calculate evaluation metrics for the predictions.
        
        Args:
            predictions (list): List of prediction dictionaries
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Example metrics calculation for image-text retrieval
        # Assuming predictions contain image_id and retrieved caption rankings
        
        r1, r5, r10 = 0, 0, 0
        for pred in predictions:
            gt_captions = [ann['captions'] for ann in self.annotations if ann['image_id'] == pred['image_id']][0]
            
            # Check if any ground truth caption is in top-k retrieved captions
            for k, retrieved_caption in enumerate(pred['retrieved_captions']):
                if any(self._caption_match(retrieved_caption, gt) for gt in gt_captions):
                    if k < 1:
                        r1 += 1
                    if k < 5:
                        r5 += 1
                    if k < 10:
                        r10 += 1
                    break
        
        total = len(predictions)
        return {
            'R@1': r1 / total,
            'R@5': r5 / total,
            'R@10': r10 / total
        }
    
    def _caption_match(self, pred_caption, gt_caption):
        """Simple exact match between captions"""
        return pred_caption.lower() == gt_caption.lower()
