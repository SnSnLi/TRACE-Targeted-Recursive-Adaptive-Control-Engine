
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

    def retrieve(self, query, candidates):
        # 根据query类型编码
        if isinstance(query, str):
            query_feat = self.encode_text(query)
        elif isinstance(query, Image.Image) or isinstance(query, np.ndarray):
            query_feat = self.encode_image(query)
        elif isinstance(query, tuple) or isinstance(query, dict):
            # 同时包含文本和图像
            q_text, q_image = query if isinstance(query, tuple) else (query['text'], query['image'])
            text_feat = self.encode_text(q_text)
            image_feat = self.encode_image(q_image)
            # 融合嵌入（取平均或拼接+线性层；这里采用平均保证维度不变）
            query_feat = (text_feat + image_feat) / 2
        else:
            raise TypeError("Unsupported query type")
        # 编码候选
        candidate_feats = []
        for cand in candidates:
            feat = self.encode_text(cand) if isinstance(cand, str) else self.encode_image(cand)
            candidate_feats.append(feat)
        # 计算相似度分数
        scores = [float((query_feat @ feat.T).item()) for feat in candidate_feats]
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = [(candidates[i], scores[i]) for i in ranked_idx]
        return results
