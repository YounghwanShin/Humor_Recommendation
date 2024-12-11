import matplotlib.pyplot as plt
import os
from typing import List

def plot_training_losses(train_losses: List[float], val_losses: List[float], model_type: str, save_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title(f'{model_type.capitalize()} Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    os.makedirs(save_dir, exist_ok=True)
    
    plot_path = os.path.join(save_dir, f'{model_type}_training_loss.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nTraining plot saved to: {plot_path}")

def save_training_history(train_losses: List[float], val_losses: List[float], model_type: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    history_path = os.path.join(save_dir, f'{model_type}_training_history.txt')
    
    with open(history_path, 'w') as f:
        f.write("epoch,train_loss,val_loss\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
    
    print(f"Training history saved to: {history_path}")