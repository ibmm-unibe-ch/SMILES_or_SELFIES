# Smiles or Selfies

## Training pipeline

1. Install environment with _make build-conda-from-env_
1. Download data with _make download-10m_ or _make download-full-pubchem_
1. Preprocess with _preprocessing.py_ some path names might need to be adapted in the main method according to the choice of downloaded data
1. Parse with _parsing.py_
1. Run `fairseq-preprocess --only-source --destdir fairseq_preprocess/atom_selfies --trainpref processed/selfies_train  --validpref processed/selfies_val`
1. Run `fairseq-train fairseq_preprocess/atom_smiles --save-dir fairseq/smiles_atom_megamol --wandb-project smiles_atom_megamol --batch-size 32 --fp16 --mask 0.2 --tokens-per-sample 512 --total-num-update 500000 --max-update 500000 --warmup-updates 8000 --task denoising --save-interval 1 --arch bart_base --optimizer adam --lr-scheduler polynomial_decay --lr 5e-04 --dropout 0.1 --criterion cross_entropy --max-tokens 3200 --weight-decay 0.01 --attention-dropout 0.0 --relu-dropout 0.0 --share-decoder-input-output-embed --share-all-embeddings --clip-norm 1.0 --skip-invalid-size-inputs-valid-test --log-format json --log-interval 1000 --save-interval-updates 5000 --keep-interval-updates 1 --update-freq 4 --seed 4 --distributed-world-size 1 --no-epoch-checkpoints --mask-length span-poisson --replace-length 1 --encoder-learned-pos --decoder-learned-pos --rotate 0.0 --mask-random 0.0 --permute-sentences 1 --insert 0.0 --poisson-lambda 3.5 --dataset-impl mmap --num-workers 4`
1. Create MolNet datasets with _dataset.py_
1. Run finetuning with _finetuning.py_ giving the CUDA-GPU and the model configuration
1. Get scores with _scoring.py_
1. Translate model from fairseq to huggingface`python3 fairseq_to_huggingface.py ~/GitHub/SMILES_or_SELFIES/fairseq/smiles_trained/checkpoint_best.pt huggingface_models/smiles_trained  --hf_config bart-base`
