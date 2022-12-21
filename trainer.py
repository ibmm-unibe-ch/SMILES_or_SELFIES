import torch
from transformers import Trainer, TrainingArguments

from constants import MODEL_PATH, SEED, TOKENIZER_PATH
from dataset import PandasDataset, split_train_eval
from model import get_BART_model
from tokenisation import get_tokenizer

torch.manual_seed(SEED + 392487)

if __name__ == "__main__":
    """only for testing purposes"""
    train_epochs = 10
    batch_size = 32
    learning_rate = 3e-5
    weight_decay = 0.001
    args = TrainingArguments(
        # output_dir: directory where the model checkpoints will be saved.
        output_dir=MODEL_PATH,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500000,
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        data_seed=SEED + 3294759848654387658,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=train_epochs,
        load_best_model_at_end=True,
        run_name="test_run",
        metric_for_best_model="loss",
        report_to="wandb",
    )
    dataset = PandasDataset("path/to/file", 210)
    tokenizer = get_tokenizer(TOKENIZER_PATH / "tokenizer")
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
