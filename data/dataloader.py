import torch
from torch.utils.data import DataLoader, Dataset
from .processors import FlickrProcessor, InfoseekProcessor, VqaProcessor

class RetrievalDataset(Dataset):
    def __init__(self, dataset_name, split, config):
        self.config = config
        
        if dataset_name == "flickr30k":
            self.processor = FlickrProcessor(split, config)
        elif dataset_name == "infoseek":
            self.processor = InfoseekProcessor(split, config)
        elif dataset_name == "vqa":
            self.processor = VqaProcessor(split, config)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
        self.data = self.processor.load_data()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.processor.process_item(self.data[idx])
        
def get_dataloader(dataset_name, split, config, batch_size=32, shuffle=True):
    dataset = RetrievalDataset(dataset_name, split, config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)