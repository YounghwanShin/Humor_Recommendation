import torch
import openai
from transformers import (
    AutoTokenizer, 
    PegasusForConditionalGeneration,
    AutoModelForCausalLM,
    BartForConditionalGeneration,
    PegasusTokenizer
)

class BaselineModel:
    def __init__(self, model_name: str, task: str, hf_token: str = None):
        self.model_name = model_name
        self.task = task  # 'summary' or 'joke'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hf_token = hf_token
        # Update to include GPT models
        self.is_decoder_only = "llama" in model_name.lower() or "gpt" in model_name.lower()
        self._initialize_model()
        
    def _initialize_model(self):
        if self.model_name == "gpt-4o-mini":
            self.is_api = True
            return
                
        self.is_api = False
        if "bart" in self.model_name.lower():
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif "llama" in self.model_name.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                token=self.hf_token
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        elif "pegasus" in self.model_name.lower():
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
                
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _post_process_decoder_output(self, output: str, task: str) -> str:
        """
        후처리 함수: 디코더 모델의 출력을 정제합니다.
        """
        # 불필요한 시스템 프롬프트나 메타 텍스트 제거
        if "Assistant:" in output:
            output = output.split("Assistant:")[-1]
        if "System:" in output:
            output = output.split("System:")[-1]
        if "Human:" in output:
            output = output.split("Human:")[0]
            
        # 줄바꿈 및 여백 정리
        output = output.strip()
        output = ' '.join(output.split())
        
        if task == "summary":
            # 요약 특화 후처리
            unwanted_prefixes = [
                "Here's a summary:",
                "Summary:",
                "The dialogue can be summarized as:",
                "To summarize:"
            ]
            for prefix in unwanted_prefixes:
                if output.startswith(prefix):
                    output = output[len(prefix):].strip()
                    
        elif task == "joke":
            # 농담 특화 후처리
            unwanted_prefixes = [
                "Here's a joke:",
                "Joke:",
                "Here's a humorous response:",
                "My response:"
            ]
            for prefix in unwanted_prefixes:
                if output.startswith(prefix):
                    output = output[len(prefix):].strip()
                    
            # 설명이나 메타 코멘트 제거
            if "This is funny because" in output:
                output = output.split("This is funny because")[0].strip()
                
        return output

    def generate_summary(self, dialogue: str) -> str:
        if self.is_api:
            try:
                response = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Summarize the following dialogue concisely:"},
                        {"role": "user", "content": dialogue}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                output = response.choices[0].message.content.strip()
                return self._post_process_decoder_output(output, "summary")
            except Exception as e:
                print(f"Error with GPT-4o-mini: {str(e)}")
                return ""
            
        inputs = self.tokenizer(
            dialogue, 
            max_length=1024,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=150,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.is_decoder_only:
            output = self._post_process_decoder_output(output, "summary")
        return output

    def generate_joke(self, context: str) -> str:
        if self.is_api:
            try:
                response = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are having a friendly conversation. Generate a humorous simple response based on the given context.
                        Requirements:
                        1. Create a natural response that continues the conversation flow
                        2. Use "you" instead of specific names
                        3. Include both clever setup and punchline
                        4. Must include one of these elements:
                        - Witty observation
                        - Playful teasing
                        - Ironic comparison
                        - Clever wordplay
                        - Situational humor"""},
                        {"role": "user", "content": context}
                    ],
                    max_tokens=100,
                    temperature=0.9
                )
                output = response.choices[0].message.content.strip()
                return self._post_process_decoder_output(output, "joke")
            except Exception as e:
                print(f"Error with GPT-3.5: {str(e)}")
                return ""
            
        prompt = f"""Given this context: {context}
        Generate a humorous response that is natural and includes clever wordplay or situational humor.
        Keep it concise and witty."""
        
        inputs = self.tokenizer(
            prompt,
            max_length=2000,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,
            num_beams=5,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=2
        )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if self.is_decoder_only:
            output = self._post_process_decoder_output(output, "joke")
        return output