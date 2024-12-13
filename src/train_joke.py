from utils.common_utils import JokeChatSystem, ContextJokeDataset
import pandas as pd
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn.utils
from tqdm import tqdm
import os
import logging
import warnings
from visualization.visualization import plot_training_losses, save_training_history
import torch.cuda.amp as amp
from bitsandbytes.optim import PagedAdamW

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_data(file_path: str) -> tuple:
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Context', 'Last_Utterance', 'Joke'])
    print(f"Loaded {len(df)} samples")
    return df['Context'].tolist(), df['Last_Utterance'].tolist(), df['Joke'].tolist()

def evaluate_model(model, dataloader, device) -> float:
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

def train_joke_model(system: JokeChatSystem, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
    config = system.config
    save_interval = config['training_config']['save_interval']
    grad_acc_steps = config['training_config']['gradient_accumulation_steps']
    
    optimizer = PagedAdamW(
        system.joke_model.parameters(),
        lr=float(config['training_config']['learning_rate']),
        weight_decay=0.01
    )
    
    total_steps = len(train_dataloader) * config['training_config']['num_epochs']
    warmup_steps = int(total_steps * config['training_config']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = config['training_config']['early_stopping_patience']
    scaler = amp.GradScaler()
    
    print("\nStarting training...")
    print(f"Total epochs: {config['training_config']['num_epochs']}")
    print(f"Model saving interval: every {save_interval} epochs")
    print(f"Training steps per epoch: {len(train_dataloader)}")
    print(f"Validation steps per epoch: {len(val_dataloader)}")
    print(f"Batch size: {config['training_config']['batch_size']}")
    print(f"Gradient accumulation steps: {grad_acc_steps}")
    print(f"Effective batch size: {config['training_config']['batch_size'] * grad_acc_steps}")
    print(f"Learning rate: {config['training_config']['learning_rate']}")
    print(f"Device: {system.device}")
    
    for epoch in range(config['training_config']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training_config']['num_epochs']}")
        system.joke_model.train()
        total_loss = 0
        train_steps = 0
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_dataloader, desc='Training', leave=False)
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(system.device)
            attention_mask = batch['attention_mask'].to(system.device)
            labels = batch['labels'].to(system.device)

            with amp.autocast(dtype=torch.bfloat16):
                outputs = system.joke_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss / grad_acc_steps
                total_loss += loss.item()

            scaler.scale(loss).backward()
            
            if (step + 1) % grad_acc_steps == 0 or step == len(train_dataloader) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    system.joke_model.parameters(),
                    config['training_config']['max_grad_norm']
                )
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                train_steps += 1

                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{(total_loss/train_steps):.4f}",
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
            system.save_models(epoch=epoch, model_type='joke')
            print(f"Checkpoint saved at epoch {epoch + 1}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            system.save_models(epoch='best', model_type='joke')
            print(f"New best model saved! (Val Loss: {val_loss:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs")
            
            if early_stopping_counter >= early_stopping_patience:
                print("\nEarly stopping triggered!")
                system.save_models(model_type='joke')
                break
    
    if epoch == config['training_config']['num_epochs'] - 1:
        system.save_models(model_type='joke')
        
    plot_training_losses(train_losses, val_losses, 'joke', 'src\visualization\training_plots')
    save_training_history(train_losses, val_losses, 'joke', 'src\visualization\training_plots')
    print("\nTraining completed!")

def main():
    print("Initializing system...")
    system = JokeChatSystem()

    print("\nLoading training and validation data...")
    train_contexts, train_last_utterances, train_jokes = load_data('data/ctx_joke_tuning_data/train_data.csv')
    val_contexts, val_last_utterances, val_jokes = load_data('data/ctx_joke_tuning_data/val_data.csv')

    train_dataset = ContextJokeDataset(
        train_contexts,
        train_last_utterances,
        train_jokes,
        system.joke_tokenizer,
        max_length=system.config['training_config']['max_source_length']
    )
    
    val_dataset = ContextJokeDataset(
        val_contexts,
        val_last_utterances,
        val_jokes,
        system.joke_tokenizer,
        max_length=system.config['training_config']['max_source_length']
    )

    print("\nPreparing data loaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=system.config['training_config']['batch_size'],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=system.config['training_config']['batch_size']
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_joke_model(system, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()