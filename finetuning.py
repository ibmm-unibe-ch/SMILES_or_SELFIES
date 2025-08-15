""" Fine tuning on MolNet tasks
SMILES or SELFIES, 2022
"""

import logging
import os
from typing import Dict, List

from constants import MOLNET_DIRECTORY, TASK_MODEL_PATH, TASK_PATH, TOKENIZER_SUFFIXES
from utils import parse_arguments
from hyperparameters import EPOCHS, BATCH_SIZE, DROPOUT, BEST_PARAMS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Configuration class for fine-tuning parameters."""
    
    def __init__(
        self,
        task: str,
        tokenizer: str,
        model_type: str,
        learning_rate: float,
        dropout: float,
        seed: int,
        cuda_device: str
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.seed = seed
        self.cuda_device = cuda_device
        
        # Derived properties
        self.model_prefix = f"{tokenizer}_{model_type}"
        self.task_info = MOLNET_DIRECTORY[task]
        self.is_classification = self.task_info["dataset_type"] == "classification"
        self.update_steps = int(self.task_info["trainingset_size"] / BATCH_SIZE * EPOCHS)
        
        # Paths
        self.task_path = TASK_PATH / task / tokenizer
        self.save_path = self.task_path / (
            f"{learning_rate}_{dropout}_seed_{seed}_arch_{model_type}"
        )
        self.final_save_path = (
            TASK_MODEL_PATH / task / self.model_prefix / 
            f"{learning_rate}_{dropout}_seed_{seed}"
        )


def validate_arguments(args: Dict) -> None:
    """Validate and sanitize command line arguments."""
    if args["tokenizer"] and args["tokenizer"] not in TOKENIZER_SUFFIXES:
        raise ValueError(f"{args['tokenizer']} not in tokenizers.")
    
    all_tasks = list(MOLNET_DIRECTORY.keys())
    if args["task"] and args["task"] not in all_tasks:
        raise ValueError(f"{args['task']} not in tasks.")
    
    if args["modeltype"] and args["modeltype"] not in ["roberta", "bart"]:
        raise ValueError(f"{args['modeltype']} not in ['roberta', 'bart']")


def get_training_configurations(args: Dict) -> List[TrainingConfig]:
    """Generate all training configurations based on arguments."""
    tasks = [args["task"]] if args["task"] else list(MOLNET_DIRECTORY.keys())
    tokenizers = [args["tokenizer"]] if args["tokenizer"] else TOKENIZER_SUFFIXES
    model_types = [args["modeltype"]] if args["modeltype"] else ["roberta", "bart"]
    num_seeds = int(args["seeds"]) if args["seeds"] else 1
    
    configs = []
    
    for tokenizer in tokenizers:
        for task in tasks:
            for model_type in model_types:
                model_prefix = f"{tokenizer}_{model_type}"
                learning_rate = float(BEST_PARAMS[task][model_prefix])
                for seed in range(num_seeds):
                    configs.append(
                        TrainingConfig(
                            task=task,
                            tokenizer=tokenizer,
                            model_type=model_type,
                            learning_rate=learning_rate,
                            dropout=DROPOUT,
                            seed=seed,
                            cuda_device=args["cuda"]
                        )
                    )
    
    return configs


def build_fairseq_command(config: TrainingConfig) -> str:
    """Construct the fairseq-train command for the given configuration."""
    base_cmd = (
        f'CUDA_VISIBLE_DEVICES={config.cuda_device} fairseq-train {config.task_path} '
        f'--seed {config.seed} --update-freq 8 '
        f'--restore-file fairseq_models/{config.model_prefix}/checkpoint_last.pt '
        f'--wandb-project {"Finetune_"+config.task} '
        f'--batch-size {BATCH_SIZE} '
        f'--task sentence_prediction '
        f'--num-workers 1 '
        f'--add-prev-output-tokens '
        f'--reset-optimizer --reset-dataloader --reset-meters '
        f'--required-batch-size-multiple 1 '
        f'--arch {config.model_type}_base '
        f'--skip-invalid-size-inputs-valid-test '
        f'--criterion sentence_prediction '
        f'--dropout {config.dropout} '
        f'--optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 '
        f'--weight-decay 0.01 --attention-dropout 0.2 --clip-norm 0.1 '
        f'--lr-scheduler polynomial_decay '
        f'--lr {config.learning_rate} '
        f'--total-num-update {config.update_steps} '
        f'--max-update {config.update_steps} '
        f'--warmup-updates {int(config.update_steps*0.16)} '
        f'--keep-best-checkpoints 1 --keep-last-epochs 1 '
        f'--save-dir {config.save_path}'
    )
    
    if config.is_classification:
        return base_cmd + ' --num-classes 2'
    else:
        return (
            base_cmd + 
            ' --num-classes 1 --best-checkpoint-metric loss '
            '--regression-target --init-token 0'
        )


def run_training(config: TrainingConfig) -> None:
    """Execute the training process for a single configuration."""
    logger.info(
        f"Started training {config.model_prefix} on {config.task} with "
        f"LR={config.learning_rate}, dropout={config.dropout}, seed={config.seed}"
    )
    
    # Run training
    train_cmd = build_fairseq_command(config)
    os.system(train_cmd)
    
    # Save best model
    os.makedirs(config.final_save_path, exist_ok=True)
    os.system(
        f'mv {config.save_path}/checkpoint_best.pt '
        f'{config.final_save_path}/checkpoint_best.pt'
    )
    
    logger.info(
        f"Finished training {config.model_prefix} on {config.task} with "
        f"seed {config.seed}"
    )


def main():
    """Main execution function for fine-tuning script."""
    args = parse_arguments(
        cuda=True, tokenizer=True, task=True, seeds=True, modeltype=True
    )
    validate_arguments(args)
    
    for config in get_training_configurations(args):
        run_training(config)


if __name__ == "__main__":
    main()