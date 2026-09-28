from weird_ai.model import WeirdAIModel
from weird_ai.tokenizer import SimpleCharacterTokenizer
from weird_ai.dataset import LyricsDataset
from weird_ai.trainer import save_checkpoint, load_checkpoint, train_model_simple
from weird_ai.config import SAMPLE_LYRICS_FILE,PROJECT_ROOT
from torch.utils.data import DataLoader
import torch
CONTEXT_LENGTH = 128   
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "lesson-05-pretrained" / "checkpoint.pt"

def main():
    print("Starting training...")
    batch_size =12

    text = SAMPLE_LYRICS_FILE.read_text(encoding="utf-8")
    tokenizer = SimpleCharacterTokenizer(text)

    vocab_size = len(tokenizer.stoi)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WeirdAIModel(vocab_size,CONTEXT_LENGTH,emb_dim = 256)
    model = model.to(device)
    train_ratio = .9
    split_idx = int(train_ratio * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)

    train_dataset = LyricsDataset(train_tokens, CONTEXT_LENGTH)
    val_dataset = LyricsDataset(val_tokens, CONTEXT_LENGTH)

    #setting up params
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    checkpoint = None
    try:
        checkpoint = load_checkpoint(model, optimizer, CHECKPOINT_PATH, device)
    except FileNotFoundError:
        print("No Checkpoint file found")
    num_epochs = 1
    eval_freq = 5
    eval_iter = 5
    start_context = "Fart"
    context_size = 128
    train_losses, val_losses, track_tokens_seen = [], [], []
    try:
        train_losses, val_losses, track_tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs, eval_freq, eval_iter, start_context, tokenizer, context_size
        )
    except KeyboardInterrupt:
        print("The Oh Shit Save")
    finally:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model,
            optimizer,
            num_epochs,
            train_losses,
            val_losses,
            track_tokens_seen,
            CHECKPOINT_PATH,
        )
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")
    # TODO:
    # 1. Load training data
    # 2. Create optimizer
    # 3. Implement training loop
    # 4. Save checkpoints

    print("Training complete.")
if __name__ == "__main__":
    main()