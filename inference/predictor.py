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
        
        # 基础 BLIP 检索模型
        self.base_retriever = BLIPRetriever(config['model'])
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['uncertainty_hidden_dim']
        ).to(self.device)
        
        # 递归自优化模型
        self.recursive_model = RecursiveRefinementModel(
            base_retriever=self.base_retriever,
            uncertainty_estimator=self.uncertainty_estimator,
            embedding_dim=config['model']['embedding_dim']
        ).to(self.device)
        
        # 元控制器（RL/停顿决策）
        self.meta_controller = MetaController(
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['agent']['hidden_dim']
        ).to(self.device)
        
        # 加载训练好的权重
        self.recursive_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.meta_controller.load_state_dict(torch.load(agent_path, map_location=self.device))
        
        # 设置评估模式
        self.recursive_model.eval()
        self.meta_controller.eval()
        
    def predict(self, query, candidates, image=None, max_iterations=10, visualize=False):
        """
        执行递归自我完善检索。如果提供图像，则执行 ViQuAE 模式：
          1) 文本 -> 候选知识检索
          2) 图像 -> 文本匹配
          3) 不确定性评估和查询自我优化 (meta_controller)
        """
        with torch.no_grad():
            # ----------- 1) 根据输入类型区分文本/图像模式 -----------
            if isinstance(query, str) and isinstance(candidates, (list, tuple)) and len(candidates) > 0:
                # ViQuAE 模式：query 为文本问题，可能有外部 image
                query_text = query
                query_image = None
                # 如果 candidates 是 (kb_texts, image) 这种二元组
                if isinstance(candidates, tuple) or (isinstance(candidates, list) and isinstance(candidates[0], tuple)):
                    kb_texts, query_image = candidates if isinstance(candidates, tuple) else candidates[0]
                else:
                    kb_texts = candidates
                    query_image = image
            else:
                # 如果 query 不是纯文本，可能 query 本身就是图像
                query_text = query if isinstance(query, str) else None
                query_image = query if not isinstance(query, str) else image
                kb_texts = candidates
            
            # ----------- 2) 文本 -> 候选知识初步检索 -----------
            if query_text is not None and kb_texts:
                # 编码文本查询
                query_text_emb = self.base_retriever.encode_text(query_text)
                # 编码候选文本并计算相似度
                cand_text_embs = [self.base_retriever.encode_text(text) for text in kb_texts]
                sim_scores = [float((query_text_emb @ cand_emb.T).item()) for cand_emb in cand_text_embs]
                # 排序并取 Top-K
                ranked_idx = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i], reverse=True)
                K = min(10, len(ranked_idx))
                topk_idx = ranked_idx[:K]
                topk_texts = [kb_texts[i] for i in topk_idx]
                topk_text_embs = [cand_text_embs[i] for i in topk_idx]
            else:
                # 无文本检索步骤，直接使用给定的候选
                topk_texts = kb_texts
                topk_text_embs = [self.base_retriever.encode_text(text) for text in topk_texts]

            # ----------- 3) 图像 -> 文本匹配检索 -----------
            if query_image is not None:
                query_img_emb = self.base_retriever.encode_image(query_image)
                # 计算图像与候选文本的相似度
                sim_scores_img = [(query_img_emb @ txt_emb.T).item() for txt_emb in topk_text_embs]
                best_idx = int(np.argmax(sim_scores_img))
                top_score = float(np.max(sim_scores_img))
            else:
                # 无图像时，默认取文本相似度最高
                best_idx = 0
                top_score = float(sim_scores[best_idx]) if topk_texts else 0.0
            
            top_result = topk_texts[best_idx]
            # 构造结果历史，初始时仅存储 topk_texts
            results_history = [[(text, 0.0) for text in topk_texts]]
            
            # ----------- 4) 计算初始不确定性 -----------
            query_emb = query_img_emb if query_image is not None else (query_text_emb if query_text is not None else None)
            top_result_emb = topk_text_embs[best_idx]
            uncertainty = self.uncertainty_estimator(query_emb.unsqueeze(0), top_result_emb.unsqueeze(0))
            uncertainties = [uncertainty.item()]
            
            # ----------- 5) 递归检索循环 -----------
            iteration = 0
            done = False
            
            while not done and iteration < max_iterations:
                # 调用元控制器决策
                action, _ = self.meta_controller.act(
                    query_emb.unsqueeze(0),
                    top_result_emb.unsqueeze(0),
                    torch.tensor([[uncertainty.item()]]),
                    quality=top_score,  # 若无 top_score 则可传 None
                    deterministic=True
                )
                
                # 若 action=0，停止
                if action == 0:
                    done = True
                    continue
                
                iteration += 1
                # 递归模型 refine_query
                refined_query_emb = self.recursive_model.refine_embedding(
                    query_emb.unsqueeze(0),
                    top_result_emb.unsqueeze(0),
                    uncertainty.unsqueeze(0)
                ).squeeze(0)
                
                # 使用 refine 后的嵌入再次检索
                new_results = self.recursive_model.retrieve_with_embedding(refined_query_emb, topk_texts)
                results_history.append(new_results)
                
                # 更新 top_result
                top_result, top_score = new_results[0]
                top_result_emb = (self.base_retriever.encode_text(top_result) if isinstance(top_result, str)
                                  else self.base_retriever.encode_image(top_result))
                
                # 更新不确定性
                uncertainty = self.uncertainty_estimator(refined_query_emb.unsqueeze(0), top_result_emb.unsqueeze(0))
                uncertainties.append(uncertainty.item())
                
                # 更新查询向量
                query_emb = refined_query_emb
            
            # 若需要可视化检索过程
            if visualize:
                self._visualize_results(query, results_history, uncertainties)
            
            return results_history, uncertainties
    
    def _visualize_results(self, query, results_history, uncertainties):
        """可视化每次迭代的得分和不确定性。"""
        import matplotlib.pyplot as plt
        
        iterations = len(results_history)
        
        # 统计 top-1 与 top-5 均值得分
        top1_scores = [results[0][1] for results in results_history]
        top5_scores = [np.mean([r[1] for r in results[:5]]) for results in results_history]
        
        plt.figure(figsize=(12, 8))
        
        # (1) 绘制 top-1 / top-5 得分曲线
        plt.subplot(2, 1, 1)
        plt.plot(range(iterations), top1_scores, 'b-', label='Top-1 Score')
        plt.plot(range(iterations), top5_scores, 'g-', label='Avg Top-5 Score')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        if len(query) > 30:
            plt.title(f'Refinement Progress for Query: {query[:30]}...')
        else:
            plt.title(f'Refinement Progress for Query: {query}')
        plt.legend()
        plt.grid(True)
        
        # (2) 绘制不确定性曲线
        plt.subplot(2, 1, 2)
        plt.plot(range(iterations), uncertainties, 'r-', label='Uncertainty')
        plt.xlabel('Iteration')
        plt.ylabel('Uncertainty')
        plt.title('Uncertainty Estimates')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 打印每次迭代的前5条结果
        print("\nTop results at each iteration:")
        for i, results in enumerate(results_history):
            print(f"\nIteration {i}:")
            for j, (result, score) in enumerate(results[:5]):
                if isinstance(result, str):
                    # 如果 result 太长，只截取前 50 字
                    result_display = result[:50] + '...' if len(result) > 50 else result
                else:
                    result_display = '[IMAGE]'
                print(f"  {j+1}. {result_display} (score: {score:.4f})")

