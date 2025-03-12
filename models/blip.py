
import torch
from transformers import BlipProcessor, BlipForImageTextRetrieval

class BLIPRetrieval:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cuda"):
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(device)
        
    def encode_image(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features
        
    def encode_text(self, text):
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features
        
    def compute_similarity(self, image_features, text_features):
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity scores
        similarity = (image_features @ text_features.T)
        return similarity