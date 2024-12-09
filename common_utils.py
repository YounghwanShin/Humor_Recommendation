import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from torch.utils.data import Dataset
import os
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

load_dotenv() 
hf_token = os.getenv('HUGGINGFACE_TOKEN')

class DialogueDataset(Dataset):
    def __init__(self, dialogues: list, summaries: list, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.dialogues = dialogues
        self.summaries = summaries
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dialogue = "summarize: " + self.dialogues[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(
            dialogue,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            summary,
            max_length=self.max_length // 4,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class ContextJokeDataset(Dataset):
    def __init__(self, contexts: List[str], jokes: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.jokes = jokes
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        context = self.contexts[idx].strip()
        joke = self.jokes[idx].strip()
        
        prompt = f"[INST]Given the following context, generate a relevant and humorous joke:\nContext: {context}\n\nGenerate a joke that is:[/INST]{joke}"
        
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze().masked_fill(
                encodings['input_ids'] == self.tokenizer.pad_token_id, -100
            )
        }

class JokeChatSystem:
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._initialize_models()

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_directories(self) -> None:
        os.makedirs(self.config['model_config']['summary_model_dir'], exist_ok=True)
        os.makedirs(self.config['model_config']['joke_model_dir'], exist_ok=True)

    def _initialize_models(self) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        ) if self.config['model_config']['device'] is None else torch.device(self.config['model_config']['device'])

        self.summary_tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_config']['summary_model_name']
        )
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['model_config']['summary_model_name']
        ).to(self.device)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config['training_config']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['training_config']['bnb_4bit_compute_dtype']),
            bnb_4bit_use_double_quant=True,
        )

        self.joke_tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_config']['joke_model_name'],
            padding_side="left",
            token=hf_token
        )
        self.joke_tokenizer.pad_token = self.joke_tokenizer.eos_token

        self.joke_model = AutoModelForCausalLM.from_pretrained(
            self.config['model_config']['joke_model_name'],
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token
        )

        self.joke_model = prepare_model_for_kbit_training(self.joke_model)

        lora_config = LoraConfig(
            r=self.config['training_config']['lora_r'],
            lora_alpha=self.config['training_config']['lora_alpha'],
            lora_dropout=self.config['training_config']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.joke_model = get_peft_model(self.joke_model, lora_config)
        self.joke_model.print_trainable_parameters()

    def save_models(self, epoch: Optional[Union[str, int]] = None, model_type: Optional[str] = None) -> None:
        if isinstance(epoch, str) and epoch == 'best':
            summary_path = os.path.join(self.config['model_config']['summary_model_dir'], 'best_model')
            joke_path = os.path.join(self.config['model_config']['joke_model_dir'], 'best_model')
        elif isinstance(epoch, (int, str)):
            summary_path = os.path.join(self.config['model_config']['summary_model_dir'], f'checkpoint_epoch_{epoch + 1}')
            joke_path = os.path.join(self.config['model_config']['joke_model_dir'], f'checkpoint_epoch_{epoch + 1}')
        else:
            summary_path = os.path.join(self.config['model_config']['summary_model_dir'], 'latest')
            joke_path = os.path.join(self.config['model_config']['joke_model_dir'], 'latest')

        if model_type is None or model_type == 'summary':
            os.makedirs(summary_path, exist_ok=True)
            self.summary_model.save_pretrained(summary_path)
            self.summary_tokenizer.save_pretrained(summary_path)
            print(f"Summary model saved to {summary_path}")

        if model_type is None or model_type == 'joke':
            os.makedirs(joke_path, exist_ok=True)
            self.joke_model.save_pretrained(joke_path)
            self.joke_tokenizer.save_pretrained(joke_path)
            print(f"Joke model (LoRA weights) saved to {joke_path}")

    def load_models(self, epoch: Optional[Union[str, int]] = None, model_type: Optional[str] = None) -> None:
        try:
            if isinstance(epoch, str) and epoch == 'best':
                summary_path = os.path.join(self.config['model_config']['summary_model_dir'], 'best_model')
                joke_path = os.path.join(self.config['model_config']['joke_model_dir'], 'best_model')
            elif isinstance(epoch, (int, str)):
                summary_path = os.path.join(self.config['model_config']['summary_model_dir'], f'checkpoint_epoch_{epoch}')
                joke_path = os.path.join(self.config['model_config']['joke_model_dir'], f'checkpoint_epoch_{epoch}')
            else:
                summary_path = os.path.join(self.config['model_config']['summary_model_dir'], 'latest')
                joke_path = os.path.join(self.config['model_config']['joke_model_dir'], 'latest')

            if model_type is None or model_type == 'summary':
                print(f"Loading summary model from {summary_path}")
                self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_path)
                self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_path)
                self.summary_model.to(self.device)
                print("Summary model loaded successfully")

            if model_type is None or model_type == 'joke':
                print(f"Loading joke model from {joke_path}")
                if not hasattr(self, 'joke_model'):
                    self._initialize_models()
                
                if os.path.exists(joke_path):
                    self.joke_model = PeftModel.from_pretrained(
                        self.joke_model,
                        joke_path,
                        is_trainable=False 
                    )
                    print("Joke model LoRA weights loaded successfully")
                else:
                    print(f"Warning: Joke model path {joke_path} does not exist")

                if hasattr(self, 'summary_model'):
                    self.summary_model.eval()
                if hasattr(self, 'joke_model'):
                    self.joke_model.eval()

        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def generate_summary(self, dialogue: str) -> str:
        dialogue = dialogue.strip()
        dialogue = "summarize: " + dialogue

        inputs = self.summary_tokenizer(
            dialogue,
            max_length=self.config['training_config']['max_source_length'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        summary_ids = self.summary_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.config['generation_config']['summary_max_length'],
            min_length=self.config['generation_config']['summary_min_length'],
            length_penalty=self.config['generation_config']['summary_length_penalty'],
            num_beams=self.config['generation_config']['summary_num_beams'],
            early_stopping=True,
            no_repeat_ngram_size=self.config['generation_config']['summary_no_repeat_ngram_size'],
            pad_token_id=self.summary_tokenizer.pad_token_id,
            bos_token_id=self.summary_tokenizer.bos_token_id,
            eos_token_id=self.summary_tokenizer.eos_token_id
        )

        return self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def recommend_joke(self, context: str) -> List[str]:
        context = context.strip()
        prompt =  f"""
[INST]
You are having a friendly conversation. Generate a humorous simple response based on the given context.

### Requirements:
1. Create a natural response that continues the conversation flow
2. Use "you" instead of specific names
3. Include both clever setup and punchline
4. Must include one of these elements:
   - Witty observation
   - Playful teasing
   - Ironic comparison
   - Clever wordplay
   - Situational humor
5. Follow this structure:
   - For one-liners: "Your [object/situation] is so [characteristic] that [humorous comparison]"
   - For short exchanges: Just the setup and punchline
   
### Style Guide:
- Keep tone light and friendly
- Focus on the main humorous elements
- Make it feel like a natural part of the dialogue
- Avoid explaining the joke

---

### Context:
{context}
---

Return only the simple humorous response, no additional text.
[/INST]        
"""

        inputs = self.joke_tokenizer(
            prompt,
            padding=True,
            truncation=True,
            max_length=self.config['training_config']['max_source_length'],
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            joke_ids = self.joke_model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.config['generation_config']['joke_max_length'],
                num_return_sequences=self.config['training_config']['num_return_sequences'],
                num_beams=self.config['generation_config']['joke_num_beams'],
                do_sample=True,
                temperature=self.config['generation_config']['joke_temperature'],
                top_k=self.config['generation_config']['joke_top_k'],
                top_p=self.config['generation_config']['joke_top_p'],
                pad_token_id=self.joke_tokenizer.pad_token_id,
                eos_token_id=self.joke_tokenizer.eos_token_id,
                no_repeat_ngram_size=self.config['generation_config']['joke_no_repeat_ngram_size'], 
                repetition_penalty=2.0
            )

        return [self.joke_tokenizer.decode(ids, skip_special_tokens=True) for ids in joke_ids]