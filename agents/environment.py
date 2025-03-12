import numpy as np
import torch
from models.recursive_model import RecursiveRetrieval 

class RecursiveRetrievalEnvironment:
    def __init__(self, recursive_model, dataset, meta_controller, target_metric='hit@1'):
        self.recursive_model = recursive_model
        self.dataset = dataset
        self.target_metric = target_metric
        self.meta_controller = meta_controller  # 添加 MetaController
        self.current_query = None
        self.current_candidates = None
        self.current_target = None
        self.current_iteration = 0
        self.max_iterations = 10
        self.results_history = []
        self.uncertainties = []

    def reset(self, query_idx=None):
        if query_idx is None:
            query_idx = np.random.randint(0, len(self.dataset))
        query, candidates, target = self.dataset[query_idx]
        self.current_query = query
        self.current_candidates = candidates
        self.current_target = target
        self.current_iteration = 0
        query_emb = self.recursive_model.base_retriever.encode_text(query) if isinstance(query, str) else self.recursive_model.base_retriever.encode_image(query)
        initial_results = self.recursive_model.base_retriever.retrieve(query, candidates)
        self.results_history = [initial_results]
        self.uncertainties = []
        top_result, _ = initial_results[0]
        top_result_emb = self.recursive_model.base_retriever.encode_text(top_result) if isinstance(top_result, str) else self.recursive_model.base_retriever.encode_image(top_result)
        uncertainty = self.recursive_model.uncertainty_estimator(query_emb, top_result_emb)
        self.uncertainties.append(uncertainty.item())
        state = {
            'query_emb': query_emb,
            'result_emb': top_result_emb,
            'uncertainty': uncertainty,
            'iteration': self.current_iteration
        }
        return state

    def step(self, action=None):
        """Take a step in the environment based on the action."""
        # 如果 action 为空，首次调用时由 meta_controller 决定
        if action is None or self.current_iteration >= self.max_iterations:
            if action is None:
                quality = self._evaluate_result(self.results_history[-1])
                query_emb = self.recursive_model.base_retriever.encode_text(self.current_query) if isinstance(self.current_query, str) else self.recursive_model.base_retriever.encode_image(self.current_query)
                top_result, _ = self.results_history[-1][0]
                top_result_emb = self.recursive_model.base_retriever.encode_text(top_result) if isinstance(top_result, str) else self.recursive_model.base_retriever.encode_image(top_result)
                uncertainty = self.uncertainties[-1]
                action = self.meta_controller.act(query_emb, top_result_emb, uncertainty, quality, uncertainty_threshold=0.3, quality_threshold=0.8)
            if action == 0 or self.current_iteration >= self.max_iterations:
                done = True
                next_state = None
                reward = self._calculate_reward()
                return next_state, reward, done, {}

        # Continue refinement
        self.current_iteration += 1
        query_emb = self.recursive_model.base_retriever.encode_text(self.current_query) if isinstance(self.current_query, str) else self.recursive_model.base_retriever.encode_image(self.current_query)
        top_result, _ = self.results_history[-1][0]
        top_result_emb = self.recursive_model.base_retriever.encode_text(top_result) if isinstance(top_result, str) else self.recursive_model.base_retriever.encode_image(top_result)
        uncertainty = self.uncertainties[-1]
        uncertainty_tensor = torch.tensor([uncertainty]).unsqueeze(0)
        refined_query_emb = self.recursive_model.refine_embedding(query_emb, top_result_emb, uncertainty_tensor)
        new_results = self.recursive_model.retrieve_with_embedding(refined_query_emb, self.current_candidates)
        self.results_history.append(new_results)
        new_top_result, _ = new_results[0]
        new_top_result_emb = self.recursive_model.base_retriever.encode_text(new_top_result) if isinstance(new_top_result, str) else self.recursive_model.base_retriever.encode_image(new_top_result)
        new_uncertainty = self.recursive_model.uncertainty_estimator(refined_query_emb, new_top_result_emb)
        self.uncertainties.append(new_uncertainty.item())
        done = self.current_iteration >= self.max_iterations
        prev_result_quality = self._evaluate_result(self.results_history[-2])
        current_result_quality = self._evaluate_result(new_results)
        reward = current_result_quality - prev_result_quality
        quality = current_result_quality  
        
        new_action = self.meta_controller.act(refined_query_emb, new_top_result_emb, new_uncertainty, quality, uncertainty_threshold=0.3, quality_threshold=0.8)
        next_state = {
            'query_emb': refined_query_emb,
            'result_emb': new_top_result_emb,
            'uncertainty': new_uncertainty,
            'iteration': self.current_iteration
        }
        return next_state, reward, done, {'action': new_action}

    def _evaluate_result(self, results):
        top_k = 5
        top_items = [item for item, _ in results[:top_k]]
        if self.target_metric == 'hit@1':
            return 1.0 if self.current_target == top_items[0] else 0.0
        elif self.target_metric == 'hit@5':
            return 1.0 if self.current_target in top_items else 0.0
        elif self.target_metric == 'mrr':
            try:
                rank = top_items.index(self.current_target) + 1
                return 1.0 / rank
            except ValueError:
                return 0.0

    def _calculate_reward(self):
        final_quality = self._evaluate_result(self.results_history[-1])
        initial_quality = self._evaluate_result(self.results_history[0])
        improvement = final_quality - initial_quality
        efficiency_penalty = -0.05 * self.current_iteration
        return final_quality + improvement + efficiency_penalty