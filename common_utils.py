import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)
from torch.utils.data import Dataset
import os
import evaluate

class DialogueDataset(Dataset):
    """대화 요약 데이터셋 클래스"""
    def __init__(self, dialogues, summaries, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.dialogues = dialogues
        self.summaries = summaries
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]

        dialogue = "summarize: " + dialogue

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

class JokeChatSystem:
    def __init__(self, config=None):
        if config is None:
            from config import SystemConfig
            self.config = SystemConfig()
        else:
            self.config = config
            
        self.summary_model_dir = self.config.model_config.summary_model_dir
        self.joke_model_dir = self.config.model_config.joke_model_dir
        
        os.makedirs(self.summary_model_dir, exist_ok=True)
        os.makedirs(self.joke_model_dir, exist_ok=True)
            
        # 토크나이저 초기화
        self.summary_tokenizer = AutoTokenizer.from_pretrained(self.config.model_config.summary_model_name)
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_config.summary_model_name)
        
        self.joke_tokenizer = AutoTokenizer.from_pretrained(self.config.model_config.joke_model_name)
        self.joke_model = AutoModelForCausalLM.from_pretrained(self.config.model_config.joke_model_name)
        
        # 토크나이저 특수 토큰 설정
        if self.joke_tokenizer.pad_token is None:
            self.joke_tokenizer.pad_token = self.joke_tokenizer.eos_token
            self.joke_model.config.pad_token_id = self.joke_tokenizer.pad_token_id
        
        self.device = torch.device(self.config.model_config.device)
        self.summary_model.to(self.device)
        self.joke_model.to(self.device)

    def save_models(self, epoch=None, model_type=None):
        if epoch is not None:
            summary_path = os.path.join(self.summary_model_dir, f'epoch_{epoch}')
            joke_path = os.path.join(self.joke_model_dir, f'epoch_{epoch}')
        else:
            summary_path = os.path.join(self.summary_model_dir, 'latest')
            joke_path = os.path.join(self.joke_model_dir, 'latest')
        
        if model_type is None or model_type == 'summary':
            self.summary_model.save_pretrained(summary_path)
            self.summary_tokenizer.save_pretrained(summary_path)
            print(f"Summary model saved to {summary_path}")
            
        if model_type is None or model_type == 'joke':
            self.joke_model.save_pretrained(joke_path)
            self.joke_tokenizer.save_pretrained(joke_path)
            print(f"Joke model saved to {joke_path}")

    def load_models(self, epoch=None, model_type=None):
        try:
            if epoch is not None:
                summary_path = os.path.join(self.summary_model_dir, f'epoch_{epoch}')
                joke_path = os.path.join(self.joke_model_dir, f'epoch_{epoch}')
            else:
                summary_path = os.path.join(self.summary_model_dir, 'latest')
                joke_path = os.path.join(self.joke_model_dir, 'latest')

            if model_type is None or model_type == 'summary':
                self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_path)
                self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_path)
                self.summary_model.to(self.device)
                print(f"Summary model loaded from {summary_path}")

            if model_type is None or model_type == 'joke':
                self.joke_model = AutoModelForCausalLM.from_pretrained(joke_path)
                self.joke_tokenizer = AutoTokenizer.from_pretrained(joke_path)
                self.joke_model.to(self.device)
                print(f"Joke model loaded from {joke_path}")

        except Exception as e:
            print(f"Error loading models: {e}")

    def generate_summary(self, dialogue):
        dialogue = dialogue.strip()
        dialogue = "summarize: " + dialogue
        
        inputs = self.summary_tokenizer(
            dialogue,
            max_length=self.config.training_config.max_source_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        summary_ids = self.summary_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.config.generation_config.summary_max_length,
            min_length=self.config.generation_config.summary_min_length,
            length_penalty=self.config.generation_config.summary_length_penalty,
            num_beams=self.config.generation_config.summary_num_beams,
            early_stopping=True,
            no_repeat_ngram_size=self.config.generation_config.summary_no_repeat_ngram_size,
            pad_token_id=self.summary_tokenizer.pad_token_id,
            bos_token_id=self.summary_tokenizer.bos_token_id,
            eos_token_id=self.summary_tokenizer.eos_token_id
        )

        summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def recommend_joke(self, context):
        """개선된 농담 추천 함수"""
        context = context.strip()
        
        # 프롬프트 예시를 분리하여 시스템 프롬프트로 사용
        system_prompt = """Here are some examples of context-related jokes:
    Context: A student is struggling with math homework
    Joke: Why did the math book look so sad? Because it had too many problems!

    Context: Someone is learning to cook
    Joke: Why did the cookie go to the doctor? Because it was feeling crumbly!"""

        user_prompt = f"Context: {context}\nJoke:"

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        inputs = self.joke_tokenizer(
            full_prompt,
            padding=True,
            truncation=True,
            max_length=self.config.training_config.max_source_length,
            return_tensors='pt'
        ).to(self.device)

        attention_mask = torch.ones_like(inputs['input_ids'])
        
        joke_ids = self.joke_model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_new_tokens=50,
            num_return_sequences=self.config.training_config.num_return_sequences,
            num_beams=self.config.generation_config.joke_num_beams,
            do_sample=True,
            temperature=self.config.generation_config.joke_temperature,
            top_k=self.config.generation_config.joke_top_k,
            top_p=self.config.generation_config.joke_top_p,
            pad_token_id=self.joke_tokenizer.pad_token_id,
            eos_token_id=self.joke_tokenizer.eos_token_id,
            no_repeat_ngram_size=self.config.generation_config.joke_no_repeat_ngram_size
        )

        prompt_length = len(inputs['input_ids'][0])
        jokes = []
        
        for ids in joke_ids:
            joke = self.joke_tokenizer.decode(ids[prompt_length:], skip_special_tokens=True)
            joke = joke.split('Context:')[0].split('Joke:')[0].strip()
            if joke:  
                jokes.append(joke)
        
        return jokes