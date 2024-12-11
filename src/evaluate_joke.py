import warnings
warnings.filterwarnings("ignore")

from utils.common_utils import JokeChatSystem
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any
import time
import json
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoModel, AutoTokenizer
from torch.nn import functional as F

class JokeEvaluator:
   def __init__(self):
       self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
       self.bleu_scorer = BLEU(tokenize='intl')
       
       self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
       self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
       
       self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
       self.gpt2_model.eval()
       
       self.sbert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
       self.sbert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
       self.sbert_model.eval()
       
       if torch.cuda.is_available():
           self.gpt2_model = self.gpt2_model.cuda()
           self.sbert_model = self.sbert_model.cuda()
           
   def is_valid_joke(self, text: str) -> bool:
       invalid_patterns = [
           '```', 'import ', 'print(', 'setup =', 
           'Witty observation:', '__name__', 
           'def ', 'class ', 'return '
       ]
       
       if any(pattern in text for pattern in invalid_patterns):
           return False
           
       if len(text.strip()) < 10:
           return False
           
       return True

   def calculate_perplexity(self, texts: List[str]) -> List[float]:
       perplexities = []
       
       for text in texts:
           try:
               encoding = self.gpt2_tokenizer(
                   text, 
                   return_tensors='pt',
                   truncation=True,
                   max_length=256
               )
               
               if torch.cuda.is_available():
                   encoding = {key: value.cuda() for key, value in encoding.items()}
               
               with torch.no_grad():
                   outputs = self.gpt2_model(**encoding)
                   logits = outputs.logits
                   
                   labels = encoding['input_ids']
                   
                   shift_logits = logits[..., :-1, :].contiguous()
                   shift_labels = labels[..., 1:].contiguous()
                   
                   loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                   loss = loss_fct(
                       shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1)
                   )
                   
                   perplexity = torch.exp(loss).cpu().item()
                   perplexities.append(perplexity)
                   
           except Exception as e:
               print(f"Error in perplexity calculation: {str(e)}")
               perplexities.append(float('inf'))
               
       return perplexities

   def mean_pooling(self, model_output, attention_mask):
       token_embeddings = model_output[0]
       input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
       return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

   def calculate_semantic_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
       similarities = []
       
       for text1, text2 in zip(texts1, texts2):
           try:
               encodings1 = self.sbert_tokenizer([text1], padding=True, truncation=True, return_tensors='pt')
               encodings2 = self.sbert_tokenizer([text2], padding=True, truncation=True, return_tensors='pt')
               
               if torch.cuda.is_available():
                   encodings1 = {key: value.cuda() for key, value in encodings1.items()}
                   encodings2 = {key: value.cuda() for key, value in encodings2.items()}
               
               with torch.no_grad():
                   outputs1 = self.sbert_model(**encodings1)
                   outputs2 = self.sbert_model(**encodings2)
                   
                   embeddings1 = self.mean_pooling(outputs1, encodings1['attention_mask'])
                   embeddings2 = self.mean_pooling(outputs2, encodings2['attention_mask'])
                   
                   embeddings1 = F.normalize(embeddings1, p=2, dim=1)
                   embeddings2 = F.normalize(embeddings2, p=2, dim=1)
                   
                   similarity = torch.cosine_similarity(embeddings1, embeddings2, dim=1)
                   similarities.append(similarity.cpu().item())
                   
           except Exception as e:
               print(f"Error in similarity calculation: {str(e)}")
               similarities.append(0.0)
               
       return similarities

   def calculate_reference_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
       results = {}
       
       refs_list = [[ref] for ref in references]
       bleu_score = self.bleu_scorer.corpus_score(predictions, refs_list)
       
       results.update({
           'bleu': float(bleu_score.score),
           'bleu_1': float(bleu_score.precisions[0]),
           'bleu_2': float(bleu_score.precisions[1]),
           'bleu_3': float(bleu_score.precisions[2]),
           'bleu_4': float(bleu_score.precisions[3])
       })
       
       rouge_scores = {
           'rouge1_f': [], 'rouge2_f': [], 'rougeL_f': [],
           'rouge1_p': [], 'rouge2_p': [], 'rougeL_p': [],
           'rouge1_r': [], 'rouge2_r': [], 'rougeL_r': []
       }
       
       for pred, ref in zip(predictions, references):
           score = self.rouge_scorer.score(ref, pred)
           rouge_scores['rouge1_f'].append(score['rouge1'].fmeasure)
           rouge_scores['rouge2_f'].append(score['rouge2'].fmeasure)
           rouge_scores['rougeL_f'].append(score['rougeL'].fmeasure)
           rouge_scores['rouge1_p'].append(score['rouge1'].precision)
           rouge_scores['rouge2_p'].append(score['rouge2'].precision)
           rouge_scores['rougeL_p'].append(score['rougeL'].precision)
           rouge_scores['rouge1_r'].append(score['rouge1'].recall)
           rouge_scores['rouge2_r'].append(score['rouge2'].recall)
           rouge_scores['rougeL_r'].append(score['rougeL'].recall)
       
       for key in rouge_scores:
           results[key] = float(np.mean(rouge_scores[key]))
           
       return results

def evaluate_joke_model(system: JokeChatSystem, test_data: pd.DataFrame, num_samples: int = 1500) -> Dict[str, Any]:
   if num_samples is None:
       num_samples = len(test_data)
   else:
       num_samples = min(num_samples, len(test_data))

   evaluator = JokeEvaluator()
   predictions = []
   references = []
   contexts = []
   generation_times = []

   print(f"\nGenerating jokes for {num_samples} samples...")
   for i in tqdm(range(num_samples)):
       try:
           context = test_data.iloc[i]['Context']
           reference = test_data.iloc[i]['Joke']
           
           start_time = time.time()
           prediction = system.recommend_joke(context)
           generation_time = time.time() - start_time
           
           if not evaluator.is_valid_joke(prediction):
               print(f"\nInvalid joke format at sample {i}")
               continue
               
           predictions.append(prediction)
           references.append(reference)
           contexts.append(context)
           generation_times.append(generation_time)
           
       except Exception as e:
           print(f"\nError generating joke for sample {i}: {str(e)}")
           continue

   if len(predictions) == 0:
       print("No successful generations!")
       return None

   print("\nCalculating metrics...")
   perplexities = evaluator.calculate_perplexity(predictions)
   context_similarities = evaluator.calculate_semantic_similarity(contexts, predictions)
   reference_similarities = evaluator.calculate_semantic_similarity(references, predictions)
   reference_metrics = evaluator.calculate_reference_metrics(predictions, references)
   
   valid_perplexities = [p for p in perplexities if p != float('inf')]
   mean_perplexity = float(np.mean(valid_perplexities)) if valid_perplexities else float('inf')
   mean_context_similarity = float(np.mean(context_similarities))
   mean_reference_similarity = float(np.mean(reference_similarities))
   
   results = {
       'metrics': {
           'reference_based': reference_metrics,
           'perplexity': mean_perplexity,
           'context_similarity': mean_context_similarity,
           'reference_similarity': mean_reference_similarity
       },
       'avg_generation_time': float(np.mean(generation_times)),
       'num_samples': num_samples,
       'successful_generations': len(predictions),
       'example_generations': []
   }
   
   num_examples = min(5, len(predictions))
   for i in range(num_examples):
       results['example_generations'].append({
           'context': contexts[i],
           'reference': references[i],
           'prediction': predictions[i],
           'perplexity': float(perplexities[i]),
           'context_similarity': float(context_similarities[i]),
           'reference_similarity': float(reference_similarities[i])
       })
   
   return results

def main():
   test_data = pd.read_csv('/workspace/data/ctx_joke_tuning_data/test_data.csv')
   system = JokeChatSystem()
   system.load_models(epoch='best', model_type='joke')
   
   results = evaluate_joke_model(system, test_data)
   
   print("\nEvaluation Results:")
   print(f"Number of samples tested: {results['num_samples']}")
   print(f"Average generation time: {results['avg_generation_time']:.3f} seconds")
   
   print("\nReference-based Metrics:")
   for metric, value in results['metrics']['reference_based'].items():
       print(f"{metric}: {value:.4f}")
   
   print("\nOther Metrics:")
   print(f"Perplexity: {results['metrics']['perplexity']:.4f}")
   print(f"Context Similarity: {results['metrics']['context_similarity']:.4f}")
   print(f"Reference Similarity: {results['metrics']['reference_similarity']:.4f}")
   
   output_file = 'joke_evaluation_results.json'
   with open(output_file, 'w', encoding='utf-8') as f:
       json.dump(results, f, ensure_ascii=False, indent=2)
   print(f"\nResults saved to {output_file}")
   
   print("\nExample Generations:")
   for i, example in enumerate(results['example_generations']):
       print(f"\nExample {i+1}:")
       print("Context:", example['context'])
       print("Reference:", example['reference'])
       print("Prediction:", example['prediction'])
       print(f"Perplexity: {example['perplexity']:.4f}")
       print(f"Context Similarity: {example['context_similarity']:.4f}")
       print(f"Reference Similarity: {example['reference_similarity']:.4f}")

if __name__ == "__main__":
   main()