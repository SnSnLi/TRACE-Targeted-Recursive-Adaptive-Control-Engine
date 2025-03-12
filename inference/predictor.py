import torch
from models.base_retrieval.blip_model import BLIPRetriever
from models.uncertainty.estimator import UncertaintyEstimator
from models.refinement.recursive_model import RecursiveRefinementModel
from agents.meta_controller import MetaController
import matplotlib.pyplot as plt
import numpy as np

class RecursiveRetrievalPredictor:
    def __init__(self, config, model_path, agent_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.base_retriever = BLIPRetriever(config['model'])
        
        self.uncertainty_estimator = UncertaintyEstimator(
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['uncertainty_hidden_dim']
        ).to(self.device)
        
        self.recursive_model = RecursiveRefinementModel(
            base_retriever=self.base_retriever,
            uncertainty_estimator=self.uncertainty_estimator,
            embedding_dim=config['model']['embedding_dim']
        ).to(self.device)
        
        self.meta_controller = MetaController(
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['agent']['hidden_dim']
        ).to(self.device)
        
        # Load trained weights
        self.recursive_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.meta_controller.load_state_dict(torch.load(agent_path, map_location=self.device))
        
        # Set to evaluation mode
        self.recursive_model.eval()
        self.meta_controller.eval()
        
    def predict(self, query, candidates, max_iterations=10, visualize=False):
        """Perform recursive self-refinement retrieval."""
        with torch.no_grad():
            # Encode query
            query_emb = self.base_retriever.encode_text(query) if isinstance(query, str) else self.base_retriever.encode_image(query)
            
            # Initial retrieval
            results = self.base_retriever.retrieve(query, candidates)
            top_result, top_score = results[0]
            
            results_history = [results]
            uncertainties = []
            
            # Get embedding for top result
            top_result_emb = self.base_retriever.encode_text(top_result) if isinstance(top_result, str) else self.base_retriever.encode_image(top_result)
            
            # Estimate uncertainty
            uncertainty = self.uncertainty_estimator(query_emb, top_result_emb)
            uncertainties.append(uncertainty.item())
            
            # Recursive refinement
            iteration = 0
            done = False
            
            while not done and iteration < max_iterations:
                # Get action from meta-controller
                action = self.meta_controller.act(
                    query_emb.unsqueeze(0),
                    top_result_emb.unsqueeze(0),
                    torch.tensor([uncertainty.item()]).unsqueeze(0).unsqueeze(0),
                    deterministic=True
                )
                
                # If action is 0, stop refinement
                if action == 0:
                    done = True
                    continue
                
                # Otherwise, continue refinement
                iteration += 1
                
                # Refine query embedding
                refined_query_emb = self.recursive_model.refine_embedding(
                    query_emb.unsqueeze(0),
                    top_result_emb.unsqueeze(0),
                    uncertainty.unsqueeze(0)
                ).squeeze(0)
                
                # Retrieve with refined embedding
                new_results = self.recursive_model.retrieve_with_embedding(refined_query_emb, candidates)
                results_history.append(new_results)
                
                # Update top result
                top_result, top_score = new_results[0]
                top_result_emb = self.base_retriever.encode_text(top_result) if isinstance(top_result, str) else self.base_retriever.encode_image(top_result)
                
                # Update uncertainty
                uncertainty = self.uncertainty_estimator(refined_query_emb.unsqueeze(0), top_result_emb.unsqueeze(0))
                uncertainties.append(uncertainty.item())
                
                # Update query embedding
                query_emb = refined_query_emb
            
            # Visualize results if requested
            if visualize:
                self._visualize_results(query, results_history, uncertainties)
            
            return results_history, uncertainties
    
    def _visualize_results(self, query, results_history, uncertainties):
        """Visualize the refinement process."""
        iterations = len(results_history)
        
        # Plot top-1 and top-5 scores across iterations
        top1_scores = [results[0][1] for results in results_history]
        top5_scores = [np.mean([r[1] for r in results[:5]]) for results in results_history]
        
        plt.figure(figsize=(12, 8))
        
        # Plot top scores
        plt.subplot(2, 1, 1)
        plt.plot(range(iterations), top1_scores, 'b-', label='Top-1 Score')
        plt.plot(range(iterations), top5_scores, 'g-', label='Avg Top-5 Score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title(f'Refinement Progress for Query: {query[:30]}...' if len(query) > 30 else f'Refinement Progress for Query: {query}')
        plt.legend()
        plt.grid(True)
        
        # Plot uncertainties
        plt.subplot(2, 1, 2)
        plt.plot(range(iterations), uncertainties, 'r-', label='Uncertainty')
        plt.xlabel('Iteration')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty Estimates')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print top results for each iteration
        print("\nTop results at each iteration:")
        for i, results in enumerate(results_history):
            print(f"\nIteration {i}:")
            for j, (result, score) in enumerate(results[:5]):
                if isinstance(result, str):
                    result_display = result[:50] + '...' if len(result) > 50 else result
                else:
                    result_display = '[IMAGE]'
                print(f"  {j+1}. {result_display} (score: {score:.4f})")