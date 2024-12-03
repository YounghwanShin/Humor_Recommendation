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
    """대화와 요약을 위한 데이터셋 클래스"""
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

        # 입력 텍스트에 특수 토큰 추가
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
    """대화 요약과 농담 생성을 위한 시스템 클래스"""
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # BART 모델 초기화
        self.summary_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')
        
        # GPT-2 모델 초기화
        self.joke_tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
        self.joke_model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')
        
        # pad_token 설정
        if self.joke_tokenizer.pad_token is None:
            self.joke_tokenizer.pad_token = self.joke_tokenizer.eos_token
            self.joke_model.config.pad_token_id = self.joke_tokenizer.pad_token_id
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.summary_model.to(self.device)
        self.joke_model.to(self.device)

    def save_models(self, epoch=None):
        """모든 모델 저장"""
        if epoch is not None:
            summary_path = os.path.join(self.model_dir, f'summary_model_epoch_{epoch}')
            joke_path = os.path.join(self.model_dir, f'joke_model_epoch_{epoch}')
        else:
            summary_path = os.path.join(self.model_dir, 'summary_model')
            joke_path = os.path.join(self.model_dir, 'joke_model')
        
        self.summary_model.save_pretrained(summary_path)
        self.summary_tokenizer.save_pretrained(summary_path)
        self.joke_model.save_pretrained(joke_path)
        self.joke_tokenizer.save_pretrained(joke_path)
        print(f"Models saved to {self.model_dir}")

    def load_models(self, epoch=None):
        """모든 모델 로드"""
        try:
            if epoch is not None:
                summary_path = os.path.join(self.model_dir, f'summary_model_epoch_{epoch}')
                joke_path = os.path.join(self.model_dir, f'joke_model_epoch_{epoch}')
            else:
                summary_path = os.path.join(self.model_dir, 'summary_model')
                joke_path = os.path.join(self.model_dir, 'joke_model')

            self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_path)
            self.summary_tokenizer = AutoTokenizer.from_pretrained(summary_path)
            self.joke_model = AutoModelForCausalLM.from_pretrained(joke_path)
            self.joke_tokenizer = AutoTokenizer.from_pretrained(joke_path)
            
            self.summary_model.to(self.device)
            self.joke_model.to(self.device)
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")

    def generate_summary(self, dialogue):
        """대화 요약 생성"""
        # 입력 텍스트에 특수 토큰 추가
        dialogue = "summarize: " + dialogue
        
        inputs = self.summary_tokenizer(
            dialogue,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        ).to(self.device)

        summary_ids = self.summary_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        summary = self.summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    def recommend_joke(self, context, max_length=100):
        """개선된 농담 추천 함수"""
        # Few-shot 예시를 포함한 프롬프트 구성
        prompt = f"""Here are some examples of context-related jokes:

        Context: A student is struggling with math homework
        Joke: Why did the math book look so sad? Because it had too many problems!

        Context: Someone is learning to cook
        Joke: Why did the cookie go to the doctor? Because it was feeling crumbly!

        Now generate a related joke for this context:
        Context: {context}
        Joke:"""

        inputs = self.joke_tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)

        joke_ids = self.joke_model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=3,
            num_beams=5,
            temperature=0.7,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.joke_tokenizer.pad_token_id,
            eos_token_id=self.joke_tokenizer.eos_token_id
        )

        jokes = [self.joke_tokenizer.decode(ids, skip_special_tokens=True) for ids in joke_ids]
        return jokes

    def evaluate_summary(self, dialogue, generated_summary, reference_summary):
        """요약 품질 평가"""
        rouge = evaluate.load('rouge')
        results = rouge.compute(
            predictions=[generated_summary],
            references=[reference_summary],
            use_stemmer=True
        )
        return results