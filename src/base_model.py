import torch
import openai
from transformers import (
    AutoTokenizer, 
    PegasusForConditionalGeneration,
    AutoModelForCausalLM,
    BartForConditionalGeneration,
    PegasusTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

class BaselineModel:
    def __init__(self, model_name: str, task: str, hf_token: str = None):
        self.model_name = model_name
        self.task = task  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hf_token = hf_token
        self.is_decoder_only = any(name in model_name.lower() for name in ["llama", "gpt", "qwen"])
        self._initialize_model()
        
    def _initialize_model(self):
        if self.model_name == "gpt-4o-mini":
            self.is_api = True
            return
                
        self.is_api = False
        if "bart" in self.model_name.lower():
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif "qwen" in self.model_name.lower():

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token
            )
            self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
                token=self.hf_token
            )
        elif "pegasus" in self.model_name.lower():
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
                
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _post_process_decoder_output(self, output: str, task: str) -> str:
        if "Assistant:" in output:
            output = output.split("Assistant:")[-1]
        if "System:" in output:
            output = output.split("System:")[-1]
        if "Human:" in output:
            output = output.split("Human:")[0]
            
        output = output.strip()
        output = ' '.join(output.split())
        
        if task == "summary":
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
            unwanted_prefixes = [
                "Here's a joke:",
                "Joke:",
                "Here's a humorous response:",
                "My response:"
            ]
            for prefix in unwanted_prefixes:
                if output.startswith(prefix):
                    output = output[len(prefix):].strip()
                    
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
            
        if "qwen" in self.model_name.lower():
            messages = [
                {"role": "system", "content": "Summarize the following dialogue concisely:"},
                {"role": "user", "content": dialogue}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                min_new_tokens=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            output_ids = [outputs[0][len(inputs.input_ids[0]):]]
            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            return self._post_process_decoder_output(output, "summary")
            
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

    def generate_joke(self, context: str, last_utterance: str) -> str:
        if self.is_api:
            try:
                response = openai.OpenAI().chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": """You are a witty assistant that generates humorous responses based on conversation context. 
    Your responses should:
    - Flow naturally with the conversation
    - Include witty observations, playful teasing, ironic comparisons, clever wordplay, or situational humor
    - Be concise and focused on the main humorous elements
    - Keep a light and friendly tone
    - Feel like a natural part of the dialogue
    - Use "you" instead of specific names
    - NOT explain the joke and NOT include metadata"""},
                        {"role": "user", "content": f"""Given the conversation context and the last message, create a humorous response.

    Context: {context}
    Last message: {last_utterance}

    Generate a natural and humorous response that continues this conversation."""}
                    ],
                    max_tokens=500,
                    temperature=0.9
                )
                output = response.choices[0].message.content.strip()
                return self._post_process_decoder_output(output, "joke")
            except Exception as e:
                print(f"Error with GPT-3.5: {str(e)}")
                return ""
            
        if "qwen" in self.model_name.lower():
            messages = [
                {"role": "system", "content": "You are a witty assistant that generates humorous responses based on conversation context."},
                {"role": "user", "content": f"""Given the conversation context and the last message, create a humorous response.

Context: {context}
Last message: {last_utterance}

Generate a natural and humorous response that continues this conversation."""}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                temperature=0.9,
                top_p=0.9,
                do_sample=True,
                no_repeat_ngram_size=2
            )
            
            output_ids = [outputs[0][len(inputs.input_ids[0]):]]
            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            return self._post_process_decoder_output(output, "joke")
            
        prompt = f"""Given the conversation context and the last message, create a humorous response.

    Context: {context}
    Last message: {last_utterance}

    Generate a natural and humorous response that continues this conversation."""
        
        inputs = self.tokenizer(
            prompt,
            max_length=256, 
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