import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.base_retrieval.blip_model import BLIPRetriever
from models.uncertainty.estimator import UncertaintyEstimator
from models.refinement.recursive_model import RecursiveRefinementModel
from agents.meta_controller import MetaController

class RecursiveRetrievalPredictor:
    def __init__(self, config, model_path, agent_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.base_retriever = BLIPRetriever(config['model'])  # 基础BLIP检索模型
        
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
        
    def predict(self, query, candidates, image=None, max_iterations=10, visualize=False):
        """执行递归自我完善检索。如果提供图像，则执行ViQuAE流程。"""
        with torch.no_grad():
            # 根据输入类型选择流程
            if isinstance(query, str) and isinstance(candidates, (list, tuple)) and len(candidates) > 0:
                # ViQuAE 模式：query为文本问题，image在外部提供的情况下
                query_text = query
                query_image = None
                # 如果候选提供的是 (文本候选列表, 图像) 二元组
                if isinstance(candidates, tuple) or (isinstance(candidates, list) and isinstance(candidates[0], tuple)):
                    # 若 candidates 是 (kb_texts, image)
                    kb_texts, query_image = candidates if isinstance(candidates, tuple) else candidates[0]
                else:
                    kb_texts = candidates
                    # 尝试从参数获取图像（如果用户通过参数 image 传入）
                    query_image = image
            else:
                query_text = query if isinstance(query, str) else None
                query_image = query if not isinstance(query, str) else image
                kb_texts = candidates

            # 1. 文本 -> 知识库初步检索（如果有文本查询和候选知识库文本）
            if query_text is not None and kb_texts:
                # 编码查询文本
                query_text_emb = self.base_retriever.encode_text(query_text)
                # 批量编码候选文本
                cand_text_embs = [self.base_retriever.encode_text(text) for text in kb_texts]
                # 计算相似度并排序候选 (文本 query vs 文本候选)
                sim_scores = [float((query_text_emb @ cand_emb.T).item()) for cand_emb in cand_text_embs]
                ranked_idx = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)
                # 选取 Top-K 候选用于下一步（例如5或10）
                K = min(10, len(ranked_idx))
                topk_idx = ranked_idx[:K]
                topk_texts = [kb_texts[i] for i in topk_idx]
                topk_text_embs = [cand_text_embs[i] for i in topk_idx]
            else:
                topk_texts = kb_texts  # 无文本检索步骤，直接使用提供的候选
                topk_text_embs = [self.base_retriever.encode_text(text) for text in topk_texts]

            # 2. 图像 -> 文本匹配检索
            if query_image is not None:
                query_img_emb = self.base_retriever.encode_image(query_image)
                # 计算图像与每个候选文本的相似度
                sim_scores_img = [(query_img_emb @ txt_emb.T).item() for txt_emb in topk_text_embs]
                best_idx = int(np.argmax(sim_scores_img))
            else:
                best_idx = 0

            top_result = topk_texts[best_idx]
            top_score = float(np.max(sim_scores_img)) if query_image is not None else (sim_scores[best_idx] if topk_texts else 0.0)
            results_history = [ [(text, 0.0) for text in topk_texts] ]
            uncertainties = []
            # 计算初始嵌入及不确定性
            query_emb = query_img_emb if query_image is not None else (query_text_emb if query_text is not None else None)
            top_result_emb = topk_text_embs[best_idx]
            uncertainty = self.uncertainty_estimator(query_emb.unsqueeze(0), top_result_emb.unsqueeze(0))
            uncertainties.append(uncertainty.item())
            
            # 3. 递归检索循环
            iteration = 0
            done = False
            
            while not done and iteration < max_iterations:
                # Get action from meta-controller
                action, _ = self.meta_controller.act(
                    query_emb.unsqueeze(0),
                    top_result_emb.unsqueeze(0),
                    torch.tensor([[uncertainty.item()]]),
                    quality=top_score if 'top_score' in locals() else None,
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
                new_results = self.recursive_model.retrieve_with_embedding(refined_query_emb, topk_texts)
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
