"""Fine-tuning on MoleculeNet tasks using SMILES or SELFIES representations.

This script handles the training process for molecular property prediction tasks,
supporting both classification and regression problems with various model architectures.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List

from constants import MOLNET_DIRECTORY, TASK_MODEL_PATH, TASK_PATH, TOKENIZER_SUFFIXES
from utils import parse_arguments
from hyperparameters import EPOCHS, BATCH_SIZE, DROPOUT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training parameters
LEARNING_RATES = [5e-06, 1e-05, 5e-05]
MODEL_TYPES = ["roberta", "bart"]
KEEP_LAST_EPOCHS = 5


class TrainingConfig:
    """Configuration for a single training run."""
    
    def __init__(
        self,
        task: str,
        tokenizer: str,
        model_type: str,
        learning_rate: float,
        dropout: float,
        cuda_device: str
    ):
        self.task = task
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.cuda_device = cuda_device
        
        # Derived properties
        self.model_prefix = f"{tokenizer}_{model_type}"
        self.task_info = MOLNET_DIRECTORY[task]
        self.is_classification = self.task_info["dataset_type"] == "classification"
        self.update_steps = int(self.task_info["trainingset_size"] / BATCH_SIZE * EPOCHS)
        
        # Paths
        self.task_path = TASK_PATH / task / tokenizer
        self.save_dir = self.task_path / f"{learning_rate}_{dropout}_based_norm"
        self.final_save_path = TASK_MODEL_PATH / task / self.model_prefix / f"{learning_rate}_{dropout}_based_norm"


def validate_args(args: Dict) -> None:
    """Validate command line arguments."""
    if args["tokenizer"] not in TOKENIZER_SUFFIXES:
        raise ValueError(f"{args['tokenizer']} not in tokenizers.")
    
    if args["task"] and args["task"] not in MOLNET_DIRECTORY:
        raise ValueError(f"{args['task']} not in tasks.")
    
    if args["modeltype"] and args["modeltype"] not in MODEL_TYPES:
        raise ValueError(f"{args['modeltype']} not in {MODEL_TYPES}")


def get_tasks_to_run(args: Dict) -> List[str]:
    """Determine which tasks to run based on arguments."""
    return [args["task"]] if args["task"] else list(MOLNET_DIRECTORY.keys())


def get_model_types_to_run(args: Dict) -> List[str]:
    """Determine which model types to run based on arguments."""
    return [args["modeltype"]] if args["modeltype"] else MODEL_TYPES


def build_fairseq_command(config: TrainingConfig) -> str:
    """Construct the fairseq-train command."""
    base_cmd = (
        f'CUDA_VISIBLE_DEVICES={config.cuda_device} fairseq-train {config.task_path} '
        f'--update-freq 8 '
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
        f'--keep-best-checkpoints 1 --keep-last-epochs {KEEP_LAST_EPOCHS} '
        f'--save-dir {config.save_dir}'
    )
    
    if config.is_classification:
        return base_cmd + ' --num-classes 2'
    else:
        return (
            base_cmd + ' --num-classes 1 '
            '--best-checkpoint-metric loss --regression-target --init-token 0'
        )


def run_training(config: TrainingConfig) -> None:
    """Execute a single training run."""
    logger.info(
        f"Starting training: task={config.task}, model={config.model_prefix}, "
        f"lr={config.learning_rate}, dropout={config.dropout}"
    )
    
    # Run training
    train_cmd = build_fairseq_command(config)
    os.system(train_cmd)
    
    # Save best model
    os.makedirs(config.final_save_path, exist_ok=True)
    os.system(
        f'mv {config.save_dir}/* {config.final_save_path}/'
    )
    
    logger.info(
        f"Completed training: task={config.task}, model={config.model_prefix}, "
        f"lr={config.learning_rate}"
    )


def main():
    """Main execution function."""
    args = parse_arguments(
        cuda=True, tokenizer=True, task=True, seeds=False, model_type=True
    )
    validate_args(args)
    
    tasks = get_tasks_to_run(args)
    model_types = get_model_types_to_run(args)
    
    for task in tasks:
        for model_type in model_types:
            for learning_rate in LEARNING_RATES:
                config = TrainingConfig(
                    task=task,
                    tokenizer=args["tokenizer"],
                    model_type=model_type,
                    learning_rate=learning_rate,
                    dropout=DROPOUT,
                    cuda_device=args["cuda"]
                )
                run_training(config)


if __name__ == "__main__":
    main()