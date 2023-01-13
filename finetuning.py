""" only testing so far
SMILES or SELFIES, 2022
"""
import argparse
import logging
import os

from constants import MOLNET_DIRECTORY, TASK_PATH

EPOCHS = 10
BATCH_SIZE = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_prefix",
        choices=[
            "selfies_atom",
            "selfies_sentencepiece",
            "smiles_atom",
            "smiles_sentencepiece",
            "selfies_atom",
            "selfies_sentencepiece",
            "smiles_atom",
            "smiles_sentencepiece",
        ],
        required=True,
    )
    parser.add_argument("--cuda", required=True, help="VISIBLE_CUDA_DEVICE")
    parser.add_argument("--task", help="Which specific task as string, default is all.")
    args = parser.parse_args()
    all_tasks = list(MOLNET_DIRECTORY.keys())
    tokenizer_prefix = args.tokenizer_prefix
    if args.task is None:
        tasks = all_tasks
    else:
        assert args.task in all_tasks, f"{args.task} not in MOLNET tasks."
        tasks = [args.task]
    os.system("export CUDA_DEVICE_ORDER=PCI_BUS_ID")
    for task in tasks:
        task_path = TASK_PATH / task / tokenizer_prefix
        update_steps = int(
            MOLNET_DIRECTORY[task]["trainingset_size"] / BATCH_SIZE * EPOCHS
        )
        logging.info(
            f"Started training of configuration {tokenizer_prefix} with task {task}."
        )
        if MOLNET_DIRECTORY[task]["dataset_type"] == "classification":
            os.system(
                f'CUDA_VISIBLE_DEVICES={args.cuda} fairseq-train {task_path} --update-freq 8 --restore-file fairseq/{tokenizer_prefix[:-5]}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 1024 --max-source-positions 1024 --dropout 0.2 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 1e-05 --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --fp16 --keep-best-checkpoints 1 --keep-last-epochs 5 --num-classes 2 --save-dir {task_path}'
            )
        else:
            os.system(
                f'CUDA_VISIBLE_DEVICES={args.cuda} fairseq-train {task_path} --update-freq 8 --restore-file fairseq/{tokenizer_prefix[:-5]}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 1024 --max-source-positions 1024 --dropout 0.2 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 1e-05 --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --fp16 --keep-best-checkpoints 1 --keep-last-epochs 5 --num-classes 1 --save-dir {task_path} --best-checkpoint-metric loss --regression-target --init-token 0'
            )
        logging.info(
            f"Finished training of configuration {tokenizer_prefix} with task {task}."
        )
