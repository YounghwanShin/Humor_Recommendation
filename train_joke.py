from common_utils import JokeChatSystem
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.utils
from tqdm import tqdm

def train_joke_model(system, jokes_dataset, epochs=3, batch_size=16):
    """
    GPT-2 모델 기본 학습 함수
    
    Args:
        system (JokeChatSystem): 학습에 사용할 시스템 인스턴스
        jokes_dataset (Dataset): 농담 데이터셋
        epochs (int): 학습 에폭 수
        batch_size (int): 배치 크기
    """
    print("Preparing joke dataset...")
    # 기본적인 joke 생성을 위한 데이터 준비
    train_encodings = system.joke_tokenizer(
        jokes_dataset['Joke'],
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )

    # 옵티마이저와 스케줄러 설정
    optimizer = AdamW(system.joke_model.parameters(), lr=5e-5)
    total_steps = (len(train_encodings['input_ids']) // batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        system.joke_model.train()
        total_loss = 0
        
        # 배치 단위로 학습 진행
        for i in tqdm(range(0, len(train_encodings['input_ids']), batch_size), desc='Training'):
            # 현재 배치 데이터 준비
            batch_input_ids = train_encodings['input_ids'][i:i+batch_size].to(system.device)
            batch_attention_mask = train_encodings['attention_mask'][i:i+batch_size].to(system.device)

            # 모델 forward pass
            outputs = system.joke_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_input_ids
            )

            loss = outputs.loss
            total_loss += loss.item()

            # 역전파 및 옵티마이저 스텝
            loss.backward()
            torch.nn.utils.clip_grad_norm_(system.joke_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 메모리 관리를 위해 중간 진행상황 출력
            if i % (batch_size * 100) == 0:
                avg_loss = total_loss / (i // batch_size + 1)
                print(f'Current batch: {i//batch_size}, Average loss: {avg_loss:.4f}')

        # 에폭 종료 후 평균 손실 계산 및 출력
        avg_loss = total_loss / (len(train_encodings['input_ids']) // batch_size)
        print(f'Epoch {epoch + 1}, Average loss: {avg_loss:.4f}')
        
        # 현재 에폭의 모델 저장
        system.save_models(epoch=epoch)

def main():
    """
    메인 실행 함수
    농담 데이터셋을 로드하고 모델 학습을 실행
    """
    print("Loading joke dataset...")
    jokes_dataset = load_dataset('Maximofn/short-jokes-dataset')

    # JokeChatSystem 초기화
    print("Initializing system...")
    system = JokeChatSystem(model_dir='saved_models')

    print("Starting joke model training...")
    train_joke_model(system, jokes_dataset['train'])
    print("Training completed!")

    # 학습 완료 후 최종 모델 저장
    system.save_models()
    print("Final model saved!")

if __name__ == "__main__":
    main()