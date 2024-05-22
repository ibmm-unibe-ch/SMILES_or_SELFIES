""" Fine tuning on MolNet tasks
SMILES or SELFIES, 2022
"""
import logging
import os

from constants import MOLNET_DIRECTORY, TASK_MODEL_PATH, TASK_PATH, TOKENIZER_SUFFIXES
from utils import parse_arguments
from hyperparameters import EPOCHS, BATCH_SIZE, DROPOUT, BEST_PARAMS

if __name__ == "__main__":
    args = parse_arguments(True, True, True, True, True)
    tasks_directory = MOLNET_DIRECTORY
    all_tasks = list(tasks_directory.keys())
    if args["tokenizer"] is None:
        tokenizers =  TOKENIZER_SUFFIXES
    else:
        assert args["tokenizer"] in TOKENIZER_SUFFIXES, f"{args['tokenizer']} not in tokenizers."
        tokenizers = [args["tokenizer"]]
    if args["task"] is None:
        tasks = all_tasks
    else:
        assert args["task"] in all_tasks, f"{args['task']} not in tasks."
        tasks = [args["task"]]
    if args["seeds"] is None:
        NUM_SEEDS = 1
    else:
        NUM_SEEDS = int(args["seeds"])
    if not "modeltype" in args:
        model_types = ["roberta", "bart"]
    else:
        assert args["modeltype"] in ["roberta", "bart"], f"{args['modeltype']} not in ['roberta', 'bart']"
        model_types = [args["modeltype"]]
    for tokenizer_prefix in tokenizers:
        for task in tasks:
            task_path = TASK_PATH / task / tokenizer_prefix
            update_steps = int(
                tasks_directory[task]["trainingset_size"] / BATCH_SIZE * EPOCHS
            )
            for model_type in model_types:
                model_prefix = tokenizer_prefix+"_"+model_type
                learning_rate = float(BEST_PARAMS[task][model_prefix])
                for seed in range(NUM_SEEDS):
                    save_path = task_path/(str(learning_rate)+"_"+str(DROPOUT)+"_seed_"+str(seed)+"_arch_"+model_type)
                    logging.info(
                        f"Started training of configuration {model_prefix} with task {task} and learning rate {learning_rate}, dropout {DROPOUT}."
                    )
                    if tasks_directory[task]["dataset_type"] == "classification":
                        os.system(
                            f'CUDA_VISIBLE_DEVICES={args["cuda"]} fairseq-train {task_path} --seed {seed} --update-freq 8 --restore-file fairseq_models/{model_prefix}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task sentence_prediction --num-workers 1 --add-prev-output-tokens --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch {model_type}_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --dropout {DROPOUT} --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {learning_rate} --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --keep-best-checkpoints 1 --keep-last-epochs 1 --num-classes 2 --save-dir {save_path}'
                        )
                    elif tasks_directory[task]["dataset_type"] == "regression":
                        os.system(
                            f'CUDA_VISIBLE_DEVICES={args["cuda"]} fairseq-train {task_path} --seed {seed} --update-freq 8 --restore-file fairseq_models/{model_prefix}/checkpoint_last.pt --wandb-project {"Finetune_"+task} --batch-size {BATCH_SIZE} --task sentence_prediction --num-workers 1 --add-prev-output-tokens --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch {model_type}_base --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --dropout {DROPOUT} --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {learning_rate} --total-num-update {update_steps} --max-update {update_steps} --warmup-updates {int(update_steps*0.16)} --keep-best-checkpoints 1 --keep-last-epochs 1 --num-classes 1 --save-dir {save_path} --best-checkpoint-metric loss --regression-target --init-token 0'
                        )
                    final_save_path = TASK_MODEL_PATH/task/model_prefix/(str(learning_rate)+"_"+str(DROPOUT)+"_seed_"+str(seed))
                    os.system(
                        f'mkdir --parents {final_save_path} ; mv {save_path}/checkpoint_best.pt {final_save_path}/checkpoint_best.pt'
                    )
                    logging.info(
                        f"Finished training of configuration {model_prefix} with task {task} and learning rate {learning_rate}, dropout {DROPOUT}, seed {seed}."
                    )
