""" Fine tuning on MolNet tasks
SMILES or SELFIES, 2022
"""
import logging
import os

from constants import (
    MOLNET_DIRECTORY,
    RETROSYNTHESIS_DIRECTORY,
    TASK_MODEL_PATH,
    TASK_PATH,
)
from utils import parse_arguments

EPOCHS = 10
BATCH_SIZE = 16

if __name__ == "__main__":
    args = parse_arguments(True, True, True)
    tasks_directory = MOLNET_DIRECTORY | RETROSYNTHESIS_DIRECTORY
    all_tasks = list(tasks_directory.keys())
    tokenizer_prefix = args["tokenizer"]
    if not ("task" in args):
        tasks = all_tasks
    else:
        assert args["task"] in all_tasks, f"{args['task']} not in tasks."
        tasks = [args["task"]]
    os.system("export CUDA_DEVICE_ORDER=PCI_BUS_ID")
    for task in tasks:
        task_path = TASK_PATH / task / tokenizer_prefix
        update_steps = int(
            tasks_directory[task]["trainingset_size"] / BATCH_SIZE * EPOCHS
        )
        for learning_rate in [5e-06, 1e-05, 5e-05]:
            for dropout in [0.1, 0.2, 0.3]:
                if tasks_directory[task]["dataset_type"] != "generation" and (
                    learning_rate != 1e-05 or dropout != 0.2
                ):
                    continue
                logging.info(
                    f"Started training of configuration {tokenizer_prefix} with task {task} and learning rate {learning_rate}, dropout {dropout}."
                )
                if tasks_directory[task]["dataset_type"] == "classification":
                    os.system(
                        f'CUDA_VISIBLE_DEVICES={args["cuda"]} fairseq-train {task_path} --update-freq 8 --restore-file fairseq_models/{tokenizer_prefix}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 1024 --max-source-positions 1024 --dropout {dropout} --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {learning_rate} --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --fp16 --keep-best-checkpoints 1 --keep-last-epochs 5 --num-classes 2 --save-dir {task_path/(str(learning_rate)+"_"+str(dropout)+"_"+"based"+"_"+"norm")}'
                    )
                elif tasks_directory[task]["dataset_type"] == "regression":
                    os.system(
                        f'CUDA_VISIBLE_DEVICES={args["cuda"]} fairseq-train {task_path} --update-freq 8 --restore-file fairseq_models/{tokenizer_prefix}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 1024 --max-source-positions 1024 --dropout {dropout} --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {learning_rate} --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --fp16 --keep-best-checkpoints 1 --keep-last-epochs 5 --num-classes 1 --save-dir {task_path/(str(learning_rate)+"_"+str(dropout)+"_"+"based"+"_"+"norm")} --best-checkpoint-metric loss --regression-target --init-token 0'
                    )
                else:
                    os.system(
                        f'CUDA_VISIBLE_DEVICES={args["cuda"]} fairseq-train {task_path/"pre-processed"} --update-freq 8 --restore-file fairseq_models/{tokenizer_prefix}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task translation --num-workers 1 --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_base --skip-invalid-size-inputs-valid-test --criterion label_smoothed_cross_entropy --max-target-positions 1024 --max-source-positions 1024 --dropout {dropout} --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {learning_rate} --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --fp16 --keep-best-checkpoints 1 --keep-last-epochs 5 --save-dir {task_path/(str(learning_rate)+"_"+str(dropout)+"_"+"based"+"_"+"norm")}'
                    )
                os.system(
                    f'mkdir --parents {TASK_MODEL_PATH/task/tokenizer_prefix/(str(learning_rate)+"_"+str(dropout)+"_"+"based"+"_"+"norm")} ; mv {task_path/(str(learning_rate)+"_"+str(dropout)+"_"+"based"+"_"+"norm")}/* /data/jgut/SoS_models/{task}/{tokenizer_prefix}/{(str(learning_rate)+"_"+str(dropout)+"_"+"based"+"_"+"norm")}'
                )
                logging.info(
                    f"Finished training of configuration {tokenizer_prefix} with task {task} and learning rate {learning_rate}, dropout {dropout}."
                )
