# pip install transformers evaluate matplotlib pandas datasets
from common_utils import JokeChatSystem
from config import SystemConfig
from visualization import plot_training_losses, save_training_history
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn.utils
from tqdm import tqdm
import os
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class ContextJokeDataset(Dataset):
    def __init__(self, contexts, jokes, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.contexts = contexts
        self.jokes = jokes
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx].strip()
        joke = self.jokes[idx].strip()
        full_prompt = f"Context: {context}\nJoke: {joke}"
        
        encodings = self.tokenizer(
            full_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

def load_data(file_path):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Context', 'Joke'])
    print(f"Loaded {len(df)} samples")
    return df['Context'].tolist(), df['Joke'].tolist()

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            eval_steps += 1

    avg_val_loss = total_loss / eval_steps
    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def train_joke_model(system, train_dataloader, val_dataloader):
    training_config = system.config.training_config
    save_interval = 5  
    
    optimizer = torch.optim.AdamW(
        system.joke_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_dataloader) * training_config.num_epochs
    warmup_steps = int(total_steps * training_config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    print("\nStarting training...")
    print(f"Total epochs: {training_config.num_epochs}")
    print(f"Model saving interval: every {save_interval} epochs")
    print(f"Training steps per epoch: {len(train_dataloader)}")
    print(f"Validation steps per epoch: {len(val_dataloader)}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Device: {system.device}")
    
    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        system.joke_model.train()
        total_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc='Training', leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(system.device)
            attention_mask = batch['attention_mask'].to(system.device)
            labels = batch['labels'].to(system.device)

            outputs = system.joke_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                system.joke_model.parameters(),
                training_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

        avg_train_loss = total_loss / train_steps
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")

        val_loss = evaluate_model(
            system.joke_model,
            val_dataloader,
            system.device
        )
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(system.joke_model_dir, f'checkpoint_epoch_{epoch + 1}')
            system.save_models(epoch=epoch, model_type='joke')
            print(f"Checkpoint saved at epoch {epoch + 1}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(system.joke_model_dir, 'best_model')
            system.save_models(epoch='best', model_type='joke')
            print(f"New best model saved! (Val Loss: {val_loss:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs")
            
            if early_stopping_counter >= early_stopping_patience:
                print("\nEarly stopping triggered!")
                break
    
    plot_training_losses(train_losses, val_losses, 'joke', 'training_plots')
    save_training_history(train_losses, val_losses, 'joke', 'training_plots')
    print("\nTraining completed!")

def main():
    print("Initializing system...")
    config = SystemConfig()
    system = JokeChatSystem(config=config)

    print("\nLoading training and validation data...")
    train_contexts, train_jokes = load_data('ctx_joke_tuning_data/train_data.csv')
    val_contexts, val_jokes = load_data('ctx_joke_tuning_data/val_data.csv')

    train_dataset = ContextJokeDataset(
        train_contexts,
        train_jokes,
        system.joke_tokenizer,
        max_length=config.training_config.max_source_length
    )
    
    val_dataset = ContextJokeDataset(
        val_contexts,
        val_jokes,
        system.joke_tokenizer,
        max_length=config.training_config.max_source_length
    )

    print("\nPreparing data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_joke_model(system, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()