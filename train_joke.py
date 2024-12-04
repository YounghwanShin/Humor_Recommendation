from common_utils import JokeChatSystem
from config import SystemConfig
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.utils
from tqdm import tqdm

def train_joke_model(system, jokes_dataset):
    training_config = system.config.training_config
    
    print("Preparing joke dataset...")
    train_encodings = system.joke_tokenizer(
        jokes_dataset['Joke'],
        truncation=True,
        padding=True,
        max_length=training_config.max_joke_length,
        return_tensors='pt'
    )

    optimizer = AdamW(system.joke_model.parameters(), lr=training_config.learning_rate)
    total_steps = (len(train_encodings['input_ids']) // training_config.batch_size) * training_config.num_epochs
    warmup_steps = int(total_steps * training_config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        system.joke_model.train()
        total_loss = 0
        
        for i in tqdm(range(0, len(train_encodings['input_ids']), training_config.batch_size), desc='Training'):
            batch_input_ids = train_encodings['input_ids'][i:i+training_config.batch_size].to(system.device)
            batch_attention_mask = train_encodings['attention_mask'][i:i+training_config.batch_size].to(system.device)

            outputs = system.joke_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_input_ids
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                system.joke_model.parameters(),
                training_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / (len(train_encodings['input_ids']) // training_config.batch_size)
        print(f'Epoch {epoch + 1}, Average loss: {avg_loss:.4f}')
        
        system.save_models(epoch=epoch, model_type='joke')

    system.save_models(model_type='joke')

def main():
    print("Loading joke dataset...")
    jokes_dataset = load_dataset('Maximofn/short-jokes-dataset')

    config = SystemConfig()
    
    print("Initializing system...")
    system = JokeChatSystem(config=config)

    print("Starting joke model training...")
    train_joke_model(system, jokes_dataset['train'])
    print("Training completed!")
    
    config.save_config('joke_training_config.json')

if __name__ == "__main__":
    main()