from common_utils import JokeChatSystem
from config import SystemConfig
from visualization import plot_training_losses, save_training_history
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn.utils
from tqdm import tqdm
import os

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
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Context', 'Joke'])
    return df['Context'].tolist(), df['Joke'].tolist()

def evaluate_model(model, val_dataloader, device, epoch):
    model.eval()
    total_loss = 0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='Validation'):
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

    return total_loss / eval_steps

def train_joke_model(system, train_dataloader, val_dataloader):
    training_config = system.config.training_config
    
    optimizer = AdamW(
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
    early_stopping_patience = 3
    
    for epoch in range(training_config.num_epochs):
        system.joke_model.train()
        total_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1} Training'):
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

        avg_train_loss = total_loss / train_steps

        val_loss = evaluate_model(
            system.joke_model,
            val_dataloader,
            system.device,
            epoch
        )
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            system.save_models(epoch=epoch, model_type='joke')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                break
    
    plot_training_losses(train_losses, val_losses, 'joke', 'training_plots')
    save_training_history(train_losses, val_losses, 'joke', 'training_plots')

def main():
    config = SystemConfig()
    system = JokeChatSystem(config=config)

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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size
    )

    train_joke_model(system, train_dataloader, val_dataloader)
    
    config.save_config('contextual_joke_training_config.json')

if __name__ == "__main__":
    main()