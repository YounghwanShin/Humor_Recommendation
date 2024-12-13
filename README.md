# Contextual Humor Assistant

## Project Overview

This project is an advanced conversational AI system that leverages machine learning to generate dialogue summaries and context-aware humorous responses. The system uses state-of-the-art transformer models with custom training techniques to create engaging and contextually relevant interactions.

## Key Features

- **Dialogue Summarization**: Generates concise summaries of conversations
- **Contextual Joke Generation**: Creates humorous responses based on conversation context
- **Advanced Model Training**: Utilizes techniques like LoRA, quantization, and custom training loops

## Technology Stack

- **Frameworks**: 
  - PyTorch
  - Transformers (Hugging Face)
  - PEFT (Parameter-Efficient Fine-Tuning)

- **Models**:
  - Seq2Seq model for summarization
  - Causal Language Model for joke generation

- **Training Techniques**:
  - LoRA (Low-Rank Adaptation)
  - 4-bit quantization
  - Gradient accumulation
  - Learning rate scheduling
  - Early stopping

## Project Structure

```
project_root/
│───.env
│───.gitignore
│───LICENSE
│───README.md
│───requirements.txt
│
├───config
│       config.json
│
├───data
│   ├───ctx_joke_tuning_data
│   └───processed_data
│
├───saved_models
│   ├───joke
│   └───summary
│
└───src
    ├───joke_eval_results
    ├───summary_eval_results
    ├───utils
    └───visualization
```

## Main Components

### 1. Summarization Model
- Uses a pre-trained seq2seq model
- Fine-tuned on DialogSum dataset
- Generates concise summaries of conversations

### 2. Joke Generation Model
- Uses a causal language model with LoRA
- Trained to generate contextually relevant jokes
- Employs advanced generation techniques like beam search and sampling

## Training Process

### Summarization Training
- Dataset: DialogSum
- Training techniques:
  - AdamW optimizer
  - Linear learning rate scheduling
  - Gradient clipping
  - Early stopping

### Joke Generation Training
- Custom dataset with context, last utterance, and target joke
- Training techniques:
  - 4-bit quantization
  - LoRA fine-tuning
  - Gradient accumulation
  - Temperature and top-k/top-p sampling

## Configuration

Configuration is managed through `config/config.json`, which includes:
- Model hyperparameters
- Training settings
- Generation parameters
- Device configuration

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Hugging Face token in `.env` file
4. Configure `config/config.json` as needed

## Usage

### Training Models
```bash
python src/train_summary.py
python src/train_joke.py
```

### Testing the System
```bash
python src/test_system.py
```

## Performance Tracking

The system includes visualization tools to track:
- Training losses
- Validation performance
- Model checkpointing

## Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss proposed changes.

## License

[Add your project's license information here]

## Acknowledgments

- Hugging Face Transformers
- DialogSum Dataset
- PyTorch Community
```

## Future Improvements

- Implement more advanced context understanding
- Expand joke generation capabilities
- Add multi-language support
- Improve model interpretability
