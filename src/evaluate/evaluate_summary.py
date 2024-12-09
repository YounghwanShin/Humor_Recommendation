import warnings
warnings.filterwarnings("ignore")

from src.utils.common_utils import JokeChatSystem
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer
import numpy as np
import json
from typing import Dict, List, Any
import time
import sacrebleu
from sacrebleu.metrics import BLEU

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {
        'rouge1_f': [],
        'rouge2_f': [],
        'rougeL_f': [],
        'rouge1_p': [],
        'rouge2_p': [],
        'rougeL_p': [],
        'rouge1_r': [],
        'rouge2_r': [],
        'rougeL_r': []
    }
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        
        scores['rouge1_f'].append(score['rouge1'].fmeasure)
        scores['rouge2_f'].append(score['rouge2'].fmeasure)
        scores['rougeL_f'].append(score['rougeL'].fmeasure)
        
        scores['rouge1_p'].append(score['rouge1'].precision)
        scores['rouge2_p'].append(score['rouge2'].precision)
        scores['rougeL_p'].append(score['rougeL'].precision)
        
        scores['rouge1_r'].append(score['rouge1'].recall)
        scores['rouge2_r'].append(score['rouge2'].recall)
        scores['rougeL_r'].append(score['rougeL'].recall)
    
    # Calculate means
    results = {}
    for key in scores:
        results[key] = float(np.mean(scores[key]))
    
    return results

def calculate_bleu_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    # BLEU expects a list of references for each prediction
    refs_list = [[ref] for ref in references]
    
    # Initialize BLEU scorer
    bleu = BLEU(tokenize='intl')  # Using international tokenization
    
    # Calculate BLEU score
    score = bleu.corpus_score(predictions, refs_list)
    
    return {
        'bleu': float(score.score),  # Overall BLEU score
        'bleu_1': float(score.precisions[0]),  # BLEU-1
        'bleu_2': float(score.precisions[1]),  # BLEU-2
        'bleu_3': float(score.precisions[2]),  # BLEU-3
        'bleu_4': float(score.precisions[3]),  # BLEU-4
        'brevity_penalty': float(score.bp),  # Brevity penalty
        'length_ratio': float(score.sys_len / score.ref_len)  # System/reference length ratio
    }

def test_model(system: JokeChatSystem, test_dataset: Any, num_samples: int = None) -> Dict[str, float]:
    predictions = []
    references = []
    times = []
    
    if num_samples is None:
        num_samples = len(test_dataset)
    else:
        num_samples = min(num_samples, len(test_dataset))
    
    print(f"\nTesting model on {num_samples} samples...")
    
    for i in tqdm(range(num_samples)):
        dialogue = test_dataset[i]['dialogue']
        reference = test_dataset[i]['summary']
        
        start_time = time.time()
        prediction = system.generate_summary(dialogue)
        end_time = time.time()
        
        predictions.append(prediction)
        references.append(reference)
        times.append(end_time - start_time)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(predictions, references)
    
    # Calculate BLEU scores
    bleu_scores = calculate_bleu_score(predictions, references)
    
    # Calculate average generation time
    avg_time = np.mean(times)
    
    # Prepare results
    results = {
        'metrics': {
            'rouge': rouge_scores,
            'bleu': bleu_scores,
        },
        'avg_generation_time': float(avg_time),
        'num_samples': num_samples,
        'example_predictions': []
    }
    
    # Add some example predictions
    num_examples = min(5, num_samples)
    for i in range(num_examples):
        results['example_predictions'].append({
            'dialogue': test_dataset[i]['dialogue'],
            'reference': references[i],
            'prediction': predictions[i]
        })
    
    return results

def save_results(results: Dict[str, Any], output_file: str) -> None:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_file}")

def print_metrics(results: Dict[str, Any]) -> None:
    print("\nROUGE Scores:")
    for metric, value in results['metrics']['rouge'].items():
        print(f"{metric}: {value:.4f}")
    
    print("\nBLEU Scores:")
    for metric, value in results['metrics']['bleu'].items():
        print(f"{metric}: {value:.4f}")

def main():
    print("Loading DialogSum dataset...")
    dialogsum_dataset = load_dataset('knkarthick/dialogsum')
    test_dataset = dialogsum_dataset['test']
    
    print("\nInitializing system...")
    system = JokeChatSystem()
    
    print("\nLoading best model...")
    system.load_models(epoch='best', model_type='summary')
    
    # Test the model
    results = test_model(system, test_dataset)
    
    # Print results
    print("\nTest Results:")
    print(f"Number of samples tested: {results['num_samples']}")
    print(f"Average generation time: {results['avg_generation_time']:.3f} seconds")
    
    # Print metrics
    print_metrics(results)
    
    # Save results
    save_results(results, 'test_results.json')
    
    # Print example predictions
    print("\nExample Predictions:")
    for i, example in enumerate(results['example_predictions']):
        print(f"\nExample {i+1}:")
        print("Dialogue:", example['dialogue'])
        print("Reference:", example['reference'])
        print("Prediction:", example['prediction'])

if __name__ == "__main__":
    main()