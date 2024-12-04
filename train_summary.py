from common_utils import JokeChatSystem, DialogueDataset
from config import SystemConfig, TrainingConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.utils
from tqdm import tqdm

def train_summarizer(system, train_dataloader, val_dataloader):
    training_config = system.config.training_config
    
    optimizer = AdamW(system.summary_model.parameters(), lr=training_config.learning_rate)
    total_steps = len(train_dataloader) * training_config.num_epochs
    warmup_steps = int(total_steps * training_config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")
        system.summary_model.train()
        total_loss = 0
        
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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                system.summary_model.parameters(),
                training_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss}')

        system.summary_model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc='Validating'):
                input_ids = batch['input_ids'].to(system.device)
                attention_mask = batch['attention_mask'].to(system.device)
                labels = batch['labels'].to(system.device)

                outputs = system.summary_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation loss: {avg_val_loss}')

        system.save_models(epoch=epoch, model_type='summary')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            system.save_models(model_type='summary')

def main():
    print("Loading DialogSum dataset...")
    dialogsum_dataset = load_dataset('knkarthick/dialogsum')

    config = SystemConfig()
    
    system = JokeChatSystem(config=config)

    print("Preparing datasets...")
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

    print("Starting training...")
    train_summarizer(system, train_dataloader, val_dataloader)
    print("Training completed!")
    
    config.save_config('training_config.json')

if __name__ == "__main__":
    main()