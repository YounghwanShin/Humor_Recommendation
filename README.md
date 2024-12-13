# Context-based Joke Generator

A context-aware humor recommendation system that generates witty responses based on dialogue understanding. The system addresses the challenge of non-verbal communication elements in text-based digital communication by incorporating contextual humor to enhance user engagement.

## Project Background

- **Digital Communication Challenge**: Growing prevalence of text messenger platforms in non-face-to-face interactions
- **Communication Gap**: Lack of non-verbal communication elements in text-based exchanges
- **Solution Approach**: Context-aware humor generation to improve emotional expression and user engagement

## Overview

This project implements a two-stage pipeline that combines dialogue understanding with contextual humor generation:

1. **Context Understanding**: Summarizes multi-turn dialogues using BART-large
2. **Humor Generation**: Creates contextually appropriate jokes using Qwen2.5-Instruct

### Architecture

#### Stage 1: Dialogue Summarization Module
- **Model**: BART-large (406M parameters)
- **Architecture**: Bidirectional and Auto-Regressive Transformer
- **Key Improvements**:
  - GeLU activation (replacing ReLU)
  - Gaussian parameter initialization (N(0,0.02))
  - Streamlined architecture without additional feed-forward networks
  - Text corruption training methodology

#### Stage 2: Humor Generation Module
- **Model**: Qwen2.5-Instruct (1.5B parameters)
- **Base Architecture**: LLaMa with significant improvements
- **Key Features**:
  - RoPE (Rotary Positional Embedding)
  - RMSNorm with pre-normalization
  - SwiGLU activation function
  - Enhanced BPE tokenizer (cl100k base)
  - Grouped Query Attention with KV cache optimization
  - YARN attention weight rescaling

## Datasets

### DialogSum Dataset
- **Total Size**: 13,460 dialogues
- **Language**: English
- **Content Coverage**: Daily-life topics
  - Education
  - Work
  - Healthcare
  - Shopping
  - Leisure
  - Travel
- **Dataset Split**:
  - Training: 12,460 dialogues
  - Validation: 500 dialogues
  - Test: 1,500 dialogues
  - Holdout: 100 dialogues
- **Characteristics**:
  - Expert-validated summaries
  - Real-world communication patterns
  - Multi-domain coverage
  - Person-to-person dialogue types

### Short Jokes Dataset
- **Total Size**: 231,657 jokes
- **Source**: Multiple websites & Reddit
- **Content Types**:
  - General Humor
  - Setup-Punchline Structure
  - Black Comedy
  - Satirical Elements
- **Dataset Split**:
  - Training: 15,000 jokes
  - Validation: 1,500 jokes
  - Test: 800 jokes
- **Customization Process**:
  1. Context Integration using GPT-4
  2. Token Length Optimization (<128 tokens)
  3. Quality Control & Filtering

## Technical Implementation

### Training Methodology

#### Summarization Model (Full Fine-tuning)
- **Rationale**: Better performance on specific task vs base knowledge
- **Configuration**:
  - Learning Rate: 0.0003 with scheduler
  - Batch Size: 32
  - Gradient Accumulation Steps: 2
  - Model Size: 406M parameters

#### Humor Generation Model (QLoRA)
- **Benefits**:
  - Maintains base knowledge
  - Prevents catastrophic forgetting
  - Generation diversity preservation
  - Efficient instruction tuning
- **Configuration**:
  - Learning Rate: 0.0003 with scheduler
  - Batch Size: 16
  - Gradient Accumulation Steps: 4
  - LoRA Settings:
    - r: 16
    - alpha: 32
    - compute_dtype: bfloat16
    - quant_type: nf4

### Performance Metrics

#### Summarization Evaluation
- **ROUGE Metrics**:
  - ROUGE-1 F1: 0.430
  - ROUGE-2 F1: 0.183
  - ROUGE-L F1: 0.341
- **BLEU Scores**:
  - BLEU-1: 77.61%
  - BLEU-2: 42.42%
  - BLEU-3: 20.00%
  - BLEU-4: 9.38%
- **Generation Time**: 0.52s per summary

#### Joke Generation Evaluation
- **Primary Metrics**:
  - BLEU Score: 32.03
  - Perplexity: 14.4
  - Context Similarity: 0.363
  - Last Utterance Similarity: 0.506
  - Reference Similarity: 0.331
- **Generation Time**: 5.46s per response

## System Requirements

### Hardware
- GPU: NVIDIA RTX 4090
- VRAM: 24GB minimum recommended

### Software
- OS: Ubuntu 22.04
- CUDA: 11.8.0
- Python: 3.10
- PyTorch: 2.1.0

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/context-based-joke-generator.git
cd context-based-joke-generator
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Add your Hugging Face token
```

## Usage

### Training

1. Train summarization model:
```bash
python src/train_summary.py
```

2. Train joke generation model:
```bash
python src/train_joke.py
```

### Testing
```bash
python src/test_system.py
```

## Project Structure
```
HUMOR_RECOMMENDATION/
├── config/
│   └── config.json           # Configuration settings
├── src/
│   ├── utils/
│   │   └── common_utils.py   # Shared utilities
│   ├── base_model.py         # Base model implementations
│   ├── train_summary.py      # Summarization training
│   ├── train_joke.py         # Joke generation training
│   ├── test_system.py        # System testing
│   └── visualization/
│       └── visualization.py  # Training visualization
├── requirements.txt
└── README.md
```

## Future Improvements

1. More comprehensive data enhancement and quality control
2. Model size optimization for improved inference speed
3. Hyperparameter tuning for better performance
4. Additional evaluation metrics development

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Submit a Pull Request

## Author

- ELLT학과 20230129 신영환

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for transformer models and libraries
- DialogSum dataset creators
- QLoRA paper authors for efficient fine-tuning techniques
- Open source community for various tools and libraries
