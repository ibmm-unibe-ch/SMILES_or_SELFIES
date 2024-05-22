# Smiles or Selfies

## Training pipeline

1. Download data with _make_
1. Preprocess with _preprocessing.py_
1. Create tokenisers with _tokenisation.py_
1. Parse with _parsing.py_
1. Run `fairseq-preprocess --only-source --destdir fairseq_preprocess/selfies_atom_isomers --trainpref processed/selfies_atom_isomers  --validpref processed/selfies_atom_isomers_val`
1. Run `fairseq-train fairseq_preprocess/selfies_atom_isomers --save-dir fairseq/selfies_atom_isomers --wandb-project pre-train --batch-size 32 --tokens-per-sample 512 --total-num-update 500000 --max-update 500000 --warmup-updates 1500 --task masked_lm --save-interval 1 --arch roberta_base --optimizer adam --lr-scheduler polynomial_decay --lr 1e-05 --dropout 0.1 --criterion masked_lm --max-tokens 3200 --weight-decay 0.01 --attention-dropout 0.2 --clip-norm 1.0 --skip-invalid-size-inputs-valid-test --log-format json --log-interval 1000 --save-interval-updates 5000 --keep-interval-updates 1 --update-freq 4 --seed 4 --distributed-world-size 1 --no-epoch-checkpoints --dataset-impl mmap --num-workers 4`
1. Create MolNet datasets with _dataset.py_
1. Find correct hyperparams with *different_seeds.py*
1. Run finetuning with _finetuning.py_ giving the CUDA-GPU and the model configuration
1. Get scores with _scoring.py_
