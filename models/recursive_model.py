

import torch
from models import BLIPRetrieval
from .uncertainty import UncertaintyEstimator
from agents import LLMAgent

class RecursiveRetriever:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.base_retriever = BLIPRetrieval(config["base_model"], self.device)
        self.uncertainty_estimator = UncertaintyEstimator(config)
        self.llm_agent = LLMAgent(config)
        self.max_iterations = config["max_iterations"]
        
    def retrieve(self, query, image_db):
        """
        Perform recursive self-refining retrieval
        """
        current_query = query
        results_history = []
        uncertainties = []
        
        for i in range(self.max_iterations):
            # Perform retrieval with current query
            image_results, scores = self._single_retrieval(current_query, image_db)
            
            # Estimate uncertainty
            uncertainty = self.uncertainty_estimator.estimate(query, current_query, scores)
            uncertainties.append(uncertainty)
            
            # Save current results
            results_history.append({
                "iteration": i,
                "query": current_query,
                "results": image_results[:5],  # Top-5 results
                "uncertainty": uncertainty
            })
            
            # Check if we need to stop based on uncertainty
            if uncertainty < self.config["uncertainty_threshold"]:
                break
                
            # Use LLM agent to reflect and refine
            reflection = self.llm_agent.reflect_on_result(query, results_history[-1], i)
            current_query = self.llm_agent.refine_query(query, reflection, results_history)
            
        return results_history[-1]["results"], results_history
        
    def _single_retrieval(self, query, image_db):
        # Encode query
        query_emb = self.base_retriever.encode_text(query)
        
        # Compute similarities with all images in the database
        similarities = []
        for img_id, img_emb in image_db.items():
            sim = self.base_retriever.compute_similarity(img_emb, query_emb).item()
            similarities.append((img_id, sim))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results and all scores
        return [x[0] for x in similarities], [x[1] for x in similarities]