import argparse
import yaml
import torch
from predictor import RecursiveRetrievalPredictor
from PIL import Image
import matplotlib.pyplot as plt

def main(config_path, model_path, agent_path, query, dataset_path=None):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create predictor
    predictor = RecursiveRetrievalPredictor(config, model_path, agent_path)
    
    # Load dataset or sample candidates
    if dataset_path:
        # Load candidates from dataset
        if config['dataset']['name'] == 'infoseek':
            from data.datasets.infoseek import InfoseekDataset
            dataset = InfoseekDataset(dataset_path, split='test')
        elif config['dataset']['name'] == 'flickr30k':
            from data.datasets.flickr30k import Flickr30kDataset
            dataset = Flickr30kDataset(dataset_path, split='test')
        elif config['dataset']['name'] == 'vqa':
            from data.datasets.vqa import VQADataset
            dataset = VQADataset(dataset_path, split='test')
        
        # Get a sample query and candidates
        _, candidates, _ = dataset[0]
    else:
        # Sample candidates (for demo purposes)
        candidates = [
            "A dog running in the park",
            "A cat sitting on a windowsill",
            "Children playing with toys",
            "A sunset over the mountains",
            "A city skyline at night",
            "People walking on a beach",
            "A forest with tall trees",
            "A plate of food on a table",
            "A car driving on a road",
            "A bird flying in the sky"
        ]
    
    # Process the query
    if query.endswith(('.jpg', '.jpeg', '.png')):
        # It's an image query
        query_input = Image.open(query)
        plt.figure(figsize=(6, 6))
        plt.imshow(query_input)
        plt.title("Query Image")
        plt.axis('off')
        plt.show()
    else:
        # It's a text query
        query_input = query
        print(f"Query: {query_input}")
    
    # Perform recursive self-refinement retrieval
    results_history, uncertainties = predictor.predict(
        query=query_input,
        candidates=candidates,
        max_iterations=config['model']['max_refinement_iterations'],
        visualize=True
    )
    
    # Print final results
    print("\nFinal Results:")
    for i, (result, score) in enumerate(results_history[-1][:5]):
        if isinstance(result, str):
            result_display = result
        else:
            plt.figure(figsize=(4, 4))
            plt.imshow(result)
            plt.title(f"Result {i+1}")
            plt.axis('off')
            plt.show()
            result_display = "[IMAGE]"
        
        print(f"{i+1}. {result_display} (score: {score:.4f})")
    
    # Print refinement summary
    print(f"\nRefinement took {len(results_history) - 1} iterations")
    print(f"Final uncertainty: {uncertainties[-1]:.4f}")
    
    # Compare with baseline
    print("\nComparison with baseline (no refinement):")
    for i, ((baseline_result, baseline_score), (final_result, final_score)) in enumerate(
        zip(results_history[0][:3], results_history[-1][:3])):
        
        print(f"{i+1}. Baseline: {baseline_result[:50] + '...' if len(str(baseline_result)) > 50 else baseline_result} "
              f"(score: {baseline_score:.4f})")
        print(f"   Refined: {final_result[:50] + '...' if len(str(final_result)) > 50 else final_result} "
              f"(score: {final_score:.4f})")
        print(f"   Improvement: {(final_score - baseline_score) / baseline_score * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive Self-Refinement Retrieval Demo")
    parser.add_argument("--config", type=str, default="config/base_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                       help="Path to trained model weights")
    parser.add_argument("--agent", type=str, default="checkpoints/best_agent.pth",
                       help="Path to trained agent weights")
    parser.add_argument("--query", type=str, required=True,
                       help="Query text or path to query image")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to dataset for candidates (optional)")
    
    args = parser.parse_args()
    
    main(args.config, args.model, args.agent, args.query, args.dataset)