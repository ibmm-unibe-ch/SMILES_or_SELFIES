from transformers import Trainer, TrainingArguments
from constants import MODEL_PATH, SEED
from model import get_BART_model
from tokenisation import get_sentencepiece_tokenizer
from dataset import split_train_eval, PandasDataset
import torch

torch.manual_seed(SEED + 392487)

if __name__ == "__main__":
    """only for testing purposes"""
    # TODO: check correct hyperparams
    train_epochs = 10
    batch_size = 32
    learning_rate = 5e-5
    args = TrainingArguments(
        # output_dir: directory where the model checkpoints will be saved.
        output_dir=MODEL_PATH,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500000,
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="tensorboard",
    )
    dataset = PandasDataset("path/to/file", 210)
    tokenizer = get_sentencepiece_tokenizer()
    train_dataset, eval_dataset = split_train_eval(dataset, 10000)
    # TODO: optimizer:warm_start? Default is AdamW ✔️
    trainer = Trainer(
        get_BART_model(),
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(MODEL_PATH)
