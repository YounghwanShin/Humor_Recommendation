from common_utils import JokeChatSystem, DialogueDataset
from config import SystemConfig
from visualization import plot_training_losses, save_training_history
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.utils
from tqdm import tqdm
import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    eval_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
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

def train_summarizer(system, train_dataloader, val_dataloader):
    training_config = system.config.training_config
    
    optimizer = AdamW(
        system.summary_model.parameters(),
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
    early_stopping_patience = 3
    early_stopping_counter = 0
    
    for epoch in range(training_config.num_epochs):
        system.summary_model.train()
        total_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc='Training'):
            input_ids = batch['input_ids'].to(system.device)
            attention_mask = batch['attention_mask'].to(system.device)
            labels = batch['labels'].to(system.device)

            outputs = system.summary_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()
            train_steps += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                system.summary_model.parameters(),
                training_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / train_steps

        val_loss = evaluate_model(system.summary_model, val_dataloader, system.device)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            system.save_models(epoch=epoch, model_type='summary')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                break
    
    plot_training_losses(train_losses, val_losses, 'summary', 'training_plots')
    save_training_history(train_losses, val_losses, 'summary', 'training_plots')

def main():
    dialogsum_dataset = load_dataset('knkarthick/dialogsum')

    config = SystemConfig()
    system = JokeChatSystem(config=config)

    train_dataset = DialogueDataset(
        dialogsum_dataset['train']['dialogue'],
        dialogsum_dataset['train']['summary'],
        system.summary_tokenizer,
        max_length=config.training_config.max_source_length
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training_config.batch_size,
        shuffle=True
    )

    val_dataset = DialogueDataset(
        dialogsum_dataset['validation']['dialogue'],
        dialogsum_dataset['validation']['summary'],
        system.summary_tokenizer,
        max_length=config.training_config.max_source_length
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training_config.batch_size
    )

    train_summarizer(system, train_dataloader, val_dataloader)
    
    config.save_config('summary_training_config.json')

if __name__ == "__main__":
    main()