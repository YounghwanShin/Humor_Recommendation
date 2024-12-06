import os
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    summary_model_dir: str = 'saved_models/summary'
    joke_model_dir: str = 'saved_models/joke'
    summary_model_name: str = 'facebook/bart-large'
    joke_model_name: str = 'openai-community/gpt2-large'
    device: Optional[str] = None 
    
@dataclass
class TrainingConfig:
    batch_size: int = 2
    learning_rate: float = 5e-5
    num_epochs: int = 100 
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_source_length: int = 384  
    max_target_length: int = 128
    max_joke_length: int = 128 
    num_return_sequences: int = 3

@dataclass
class GenerationConfig:
    summary_max_length: int = 128
    summary_min_length: int = 30
    summary_length_penalty: float = 1.0
    summary_num_beams: int = 4
    summary_no_repeat_ngram_size: int = 3
    joke_max_length: int = 100
    joke_num_beams: int = 5
    joke_temperature: float = 0.7
    joke_no_repeat_ngram_size: int = 2
    joke_top_k: int = 50
    joke_top_p: float = 0.9

class SystemConfig:
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        generation_config: Optional[GenerationConfig] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.generation_config = generation_config or GenerationConfig()
        
        if self.model_config.device is None:
            self.model_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        os.makedirs(self.model_config.summary_model_dir, exist_ok=True)
        os.makedirs(self.model_config.joke_model_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        model_config = ModelConfig(**config_dict.get('model_config', {}))
        training_config = TrainingConfig(**config_dict.get('training_config', {}))
        generation_config = GenerationConfig(**config_dict.get('generation_config', {}))
        return cls(model_config, training_config, generation_config)
    
    def to_dict(self) -> dict:
        return {
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'generation_config': self.generation_config.__dict__
        }
    
    def save_config(self, filepath: str):
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_config(cls, filepath: str):
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)