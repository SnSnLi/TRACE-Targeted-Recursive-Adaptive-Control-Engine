import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import Counter

class VQADataset(Dataset):
    """
    Dataset class for Visual Question Answering (VQA) dataset.
    """
    
    def __init__(self, data_path, transform=None, split='train', version='v2'):
        """
        Initialize the VQA dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            transform (callable, optional): Transform to apply to images
            split (str): Dataset split ('train', 'val', or 'test')
            version (str): VQA version ('v1' or 'v2')
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        self.version = version
        
        # Load questions
        questions_file = os.path.join(data_path, f'v{version}_{split}_questions.json')
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
            self.questions = questions_data['questions']
        
        # Load annotations (answers) if not test split
        if split != 'test':
            annotations_file = os.path.join(data_path, f'v{version}_{split}_annotations.json')
            with open(annotations_file, 'r') as f:
                annotations_data = json.load(f)
                self.annotations = annotations_data['annotations']
        else:
            self.annotations = None
        
        # Image directory
        self.image_dir = os.path.join(data_path, f'{split}2014')
        
        # Create answer to id mapping
        if split != 'test':
            self._create_answer_mapping()
    
    def _create_answer_mapping(self):
        """Create mapping from answers to ids for most common answers"""
        answer_counts = Counter()
        for ann in self.annotations:
            for answer in ann['answers']:
                answer_counts[answer['answer']] += 1
        
        # Keep top 3000 answers
        self.top_answers = [a[0] for a in answer_counts.most_common(3000)]
        self.answer_to_id = {a: i for i, a in enumerate(self.top_answers)}
        self.id_to_answer = {i: a for i, a in enumerate(self.top_answers)}
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: A dictionary containing:
                - 'image': The image tensor
                - 'question': The question text
                - 'question_id': The question ID
                - 'answers': List of answer texts (if available)
                - 'answer_scores': One-hot encoded answer scores (if available)
        """
        question_data = self.questions[idx]
        
        # Load image
        image_id = question_data['image_id']
        image_path = os.path.join(self.image_dir, f'COCO_{self.split}2014_{image_id:012d}.jpg')
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get question
        question = question_data['question']
        question_id = question_data['question_id']
        
        result = {
            'image': image,
            'question': question,
            'question_id': question_id,
            'image_id': image_id
        }
        
        # Get answers if available
        if self.annotations is not None:
            # Find corresponding annotation
            ann = [a for a in self.annotations if a['question_id'] == question_id][0]
            
            # Get all answers
            answers = [a['answer'] for a in ann['answers']]
            result['answers'] = answers
            
            # Create answer scores (for training)
            if hasattr(self, 'answer_to_id'):
                answer_counter = Counter(answers)
                answer_scores = torch.zeros(len(self.top_answers))
                for answer, count in answer_counter.items():
                    if answer in self.answer_to_id:
                        answer_scores[self.answer_to_id[answer]] = count / len(answers)
                result['answer_scores'] = answer_scores
        
        return result
    
    def get_eval_metrics(self, predictions):
        """
        Calculate evaluation metrics for the predictions.
        
        Args:
            predictions (list): List of prediction dictionaries
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.annotations is None:
            return {'error': 'Cannot evaluate on test set without annotations'}
        
        # VQA accuracy
        accuracy = 0
        for pred in predictions:
            question_id = pred['question_id']
            pred_answer = pred['answer']
            
            # Find ground truth answers
            gt_answers = [a for a in self.annotations if a['question_id'] == question_id][0]['answers']
            gt_answers = [a['answer'] for a in gt_answers]
            
            # Calculate VQA accuracy
            n = len(gt_answers)
            acc = min(sum([1 for a in gt_answers if a.lower() == pred_answer.lower()]) / 3, 1)
            accuracy += acc
        
        accuracy /= len(predictions)
        
        return {
            'accuracy': accuracy
        }
