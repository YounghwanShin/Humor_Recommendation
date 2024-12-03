from common_utils import JokeChatSystem, DialogueDataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.utils
from tqdm import tqdm

def train_summarizer(system, train_dataloader, val_dataloader, epochs=3):
    """대화 요약 모델 학습 함수"""
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(system.summary_model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        system.summary_model.train()
        total_loss = 0
        
        # 학습 루프
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
            torch.nn.utils.clip_grad_norm_(system.summary_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Average training loss: {avg_train_loss}')

        # 검증 단계
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

        # 모델 저장
        system.save_models(epoch=epoch)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            system.save_models()

def main():
    """메인 학습 함수"""
    print("Loading DialogSum dataset...")
    dialogsum_dataset = load_dataset('knkarthick/dialogsum')

    # 시스템 초기화
    system = JokeChatSystem(model_dir='saved_models')

    print("Preparing datasets...")
    # 학습 데이터셋 준비
    train_dataset = DialogueDataset(
        dialogsum_dataset['train']['dialogue'],
        dialogsum_dataset['train']['summary'],
        system.summary_tokenizer
    )
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # 검증 데이터셋 준비
    val_dataset = DialogueDataset(
        dialogsum_dataset['validation']['dialogue'],
        dialogsum_dataset['validation']['summary'],
        system.summary_tokenizer
    )
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    print("Starting training...")
    train_summarizer(system, train_dataloader, val_dataloader)
    print("Training completed!")

if __name__ == "__main__":
    main()