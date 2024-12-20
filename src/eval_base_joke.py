#/bin/python /workspace/src/eval_base_joke.py --model gpt
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any
from evaluate_joke import JokeEvaluator
from evaluate_summary import calculate_rouge_scores, calculate_bleu_score
from base_model import BaselineModel
import argparse
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HUGGINGFACE_TOKEN')

def evaluate_single_model_jokes(
    model: BaselineModel,
    test_data: pd.DataFrame,
    num_samples: int = 800
) -> Dict[str, Any]:
    evaluator = JokeEvaluator()
    results = {
        "predictions": [],
        "times": [],
        "contexts": [],
        "last_utterances": [],
        "references": [],
        "metrics": {}
    }
    
    num_samples = min(num_samples, len(test_data))
    
    print(f"\nGenerating jokes for {model.model_name} ({num_samples} samples)...")
    for i in tqdm(range(num_samples)):
        context = test_data.iloc[i]['Context']
        last_utterance = test_data.iloc[i]['Last_Utterance']
        reference = test_data.iloc[i]['Joke']
        
        try:
            start_time = time.time()
            prediction = model.generate_joke(context, last_utterance)
            gen_time = time.time() - start_time
            
            if evaluator.is_valid_joke(prediction):
                results["predictions"].append(prediction)
                results["times"].append(gen_time)
                results["contexts"].append(context)
                results["last_utterances"].append(last_utterance)
                results["references"].append(reference)
        except Exception as e:
            print(f"\nError with model {model.model_name} on sample {i}: {str(e)}")
            continue

    if results["predictions"]:
        predictions = results["predictions"]
        references = results["references"]
        
        rouge_scores = calculate_rouge_scores(predictions, references)
        bleu_scores = calculate_bleu_score(predictions, references)
        
        perplexities = evaluator.calculate_perplexity(predictions)
        valid_perplexities = [p for p in perplexities if p != float('inf')]
        mean_perplexity = float(np.mean(valid_perplexities)) if valid_perplexities else float('inf')
        
        context_similarities = evaluator.calculate_semantic_similarity(
            results["contexts"],
            predictions
        )
        last_utterance_similarities = evaluator.calculate_semantic_similarity(
            results["last_utterances"],
            predictions
        )
        reference_similarities = evaluator.calculate_semantic_similarity(
            results["references"],
            predictions
        )
        
        avg_time = np.mean(results["times"])
        
        results["metrics"] = {
            "reference_based": {
                **bleu_scores,
                **rouge_scores
            },
            "perplexity": mean_perplexity,
            "context_similarity": float(np.mean(context_similarities)),
            "last_utterance_similarity": float(np.mean(last_utterance_similarities)),
            "reference_similarity": float(np.mean(reference_similarities))
        }
        results["avg_generation_time"] = float(avg_time)
        
        results["examples"] = []
        for i in range(min(5, len(predictions))):
            results["examples"].append({
                "context": results["contexts"][i],
                "last_utterance": results["last_utterances"][i],
                "reference": results["references"][i],
                "prediction": predictions[i],
                "perplexity": float(perplexities[i]),
                "context_similarity": float(context_similarities[i]),
                "last_utterance_similarity": float(last_utterance_similarities[i]),
                "reference_similarity": float(reference_similarities[i])
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate joke generation models')
    parser.add_argument('--model', type=str, 
                      choices=['qwen', 'gpt', 'all'], 
                      default='all', help='Model to evaluate')
    parser.add_argument('--num_samples', type=int, default=800,
                      help='Number of samples to evaluate')
    args = parser.parse_args()
    
    models = []
    if args.model == 'all' or args.model == 'gpt':
        models.append(BaselineModel("gpt-4o-mini", "joke", hf_token))
    if args.model == 'all' or args.model == 'qwen':
        models.append(BaselineModel("Qwen/Qwen2.5-1.5B-Instruct", "joke", hf_token))
    
    print("Loading joke dataset...")
    joke_dataset = pd.read_csv('data\ctx_joke_tuning_data\test_data.csv')
    
    results = {}
    print("\nEvaluating joke generation models...")
    
    for model in models:
        model_results = evaluate_single_model_jokes(model, joke_dataset, args.num_samples)
        results[model.model_name] = model_results
        
        intermediate_results = {
            "joke_evaluation": {model.model_name: model_results},
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        model_name_short = model.model_name.split("/")[-1].replace("-", "_")
        with open(f'src\joke_eval_results\joke_evaluation_{model_name_short}.json', 'w') as f:
            json.dump(intermediate_results, f, indent=2)
        
        print(f"\n{model.model_name} Results:")
        ref_metrics = model_results["metrics"]["reference_based"]
        print(f"BLEU: {ref_metrics['bleu']:.4f}")
        print(f"ROUGE-1 F1: {ref_metrics['rouge1_f']:.4f}")
        print(f"ROUGE-2 F1: {ref_metrics['rouge2_f']:.4f}")
        print(f"ROUGE-L F1: {ref_metrics['rougeL_f']:.4f}")
        print(f"Perplexity: {model_results['metrics']['perplexity']:.4f}")
        print(f"Context Similarity: {model_results['metrics']['context_similarity']:.4f}")
        print(f"Reference Similarity: {model_results['metrics']['reference_similarity']:.4f}")
        print(f"Avg Generation Time: {model_results['avg_generation_time']:.3f}s")
    
    if len(models) > 1:
        final_results = {
            "joke_evaluation": results,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('src\joke_eval_results\joke_evaluation_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\nAll results saved to joke_evaluation_results.json")

if __name__ == "__main__":
    main()