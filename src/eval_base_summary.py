#/bin/python /workspace/src/eval_base_summary.py --model gpt
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
import time
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from datasets import load_dataset
from evaluate_summary import calculate_rouge_scores, calculate_bleu_score
from base_model import BaselineModel
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HUGGINGFACE_TOKEN')

def evaluate_single_model(
    model: BaselineModel,
    test_dataset: Any,
    start_idx: int = 0,
    end_idx: int = None
) -> Dict[str, Any]:
    """단일 모델에 대한 평가를 수행합니다."""
    results = {
        "predictions": [],
        "times": [],
        "metrics": {}
    }
    
    if end_idx is None:
        end_idx = len(test_dataset)
    
    references = []
    print(f"\nGenerating summaries for {model.model_name} ({end_idx - start_idx} samples)...")
    
    for i in tqdm(range(start_idx, end_idx)):
        dialogue = test_dataset[i]['dialogue']
        reference = test_dataset[i]['summary']
        references.append(reference)
        
        try:
            start_time = time.time()
            prediction = model.generate_summary(dialogue)
            gen_time = time.time() - start_time
            
            results["predictions"].append(prediction)
            results["times"].append(gen_time)
        except Exception as e:
            print(f"\nError with model {model.model_name} on sample {i}: {str(e)}")
            continue
    
    if results["predictions"]:
        # Calculate metrics
        rouge_scores = calculate_rouge_scores(results["predictions"], references)
        bleu_scores = calculate_bleu_score(results["predictions"], references)
        avg_time = np.mean(results["times"])
        
        results["metrics"] = {
            "rouge": rouge_scores,
            "bleu": bleu_scores,
            "avg_generation_time": float(avg_time)
        }
        
        # Add example predictions
        results["examples"] = []
        for i in range(min(5, len(results["predictions"]))):
            results["examples"].append({
                "dialogue": test_dataset[start_idx + i]['dialogue'],
                "reference": references[i],
                "prediction": results["predictions"][i]
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate summary models')
    parser.add_argument('--model', type=str, choices=['bart', 'pegasus', 'gpt', 'all'], 
                      default='all', help='Model to evaluate')
    args = parser.parse_args()
    
    # Initialize models based on argument
    models = []
    if args.model == 'all' or args.model == 'bart':
        models.append(BaselineModel("facebook/bart-large-cnn", "summary", hf_token))
    if args.model == 'all' or args.model == 'pegasus':
        models.append(BaselineModel("google/pegasus-large", "summary", hf_token))
    if args.model == 'all' or args.model == 'gpt':
        models.append(BaselineModel("gpt-4o-mini", "summary", hf_token))
    
    # Load dataset
    print("Loading DialogSum dataset...")
    dialogsum_dataset = load_dataset('knkarthick/dialogsum')['test']
    
    # Evaluate models
    results = {}
    print("\nEvaluating dialogue summary models...")
    
    for model in models:
        model_results = evaluate_single_model(model, dialogsum_dataset)
        results[model.model_name] = model_results
        
        # Save intermediate results
        intermediate_results = {
            "summary_evaluation": {model.model_name: model_results},
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(f'summary_evaluation_{model.model_name.split("/")[-1]}.json', 'w') as f:
            json.dump(intermediate_results, f, indent=2)
            
        # Print intermediate results
        print(f"\n{model.model_name} Results:")
        print(f"ROUGE-1 F1: {model_results['metrics']['rouge']['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {model_results['metrics']['rouge']['rouge2_f']:.4f}")
        print(f"BLEU: {model_results['metrics']['bleu']['bleu']:.4f}")
        print(f"Avg Generation Time: {model_results['metrics']['avg_generation_time']:.3f}s")
    
    # Save final combined results
    if len(models) > 1:
        final_results = {
            "summary_evaluation": results,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('summary_evaluation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\nAll results saved to summary_evaluation_results.json")

if __name__ == "__main__":
    main()