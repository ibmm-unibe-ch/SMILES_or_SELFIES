# Smiles or Selfies
## Training pipeline
1. Download data with *make* 
1. Preprocess with *preprocessing.py*
1. Parse with *parsing.py*
1. Run ```fairseq-preprocess --only-source --destdir fairseq_preprocess/atom_selfies --trainpref processed/selfies_train  --validpref processed/selfies_val```
1. Run ```fairseq-train fairseq_preprocess/atom_smiles --save-dir fairseq/smiles_atom_megamol --wandb-project smiles_atom_megamol --batch-size 32 --fp16 --mask 0.2 --tokens-per-sample 512 --total-num-update 500000 --max-update 500000 --warmup-updates 8000 --task denoising --save-interval 1 --arch bart_base --optimizer adam --lr-scheduler polynomial_decay --lr 5e-04 --dropout 0.1 --criterion cross_entropy --max-tokens 3200 --weight-decay 0.01 --attention-dropout 0.0 --relu-dropout 0.0 --share-decoder-input-output-embed --share-all-embeddings --clip-norm 1.0 --skip-invalid-size-inputs-valid-test --log-format json --log-interval 1000 --save-interval-updates 5000 --keep-interval-updates 1 --update-freq 4 --seed 4 --distributed-world-size 1 --no-epoch-checkpoints --mask-length span-poisson --replace-length 1 --encoder-learned-pos --decoder-learned-pos --rotate 0.0 --mask-random 0.0 --permute-sentences 1 --insert 0.0 --poisson-lambda 3.5 --dataset-impl mmap --num-workers 4```
1. Create MolNet datasets with *dataset.py*
1. Preprocess datasets with ```fairseq-preprocess --only-source --trainpref dataset/selfies_atom/hiv/train.input --validpref dataset/selfies_atom/hiv/valid.input --testpref dataset/selfies_atom/hiv/test.input --destdir test_dir/input0 --workers 60 --srcdict fairseq_preprocess/atom_selfies/dict.txt```
    ```fairseq-preprocess --only-source --trainpref dataset/selfies_atom/hiv/train.label --validpref dataset/selfies_atom/hiv/valid.label --testpref dataset/selfies_atom/hiv/test.label --destdir test_dir/label --workers 60 
    ```
1. Translate model from fairseq to huggingface```python3 fairseq_to_huggingface.py ~/GitHub/SMILES_or_SELFIES/fairseq/smiles_trained/checkpoint_best.pt huggingface_models/smiles_trained  --hf_config bart-base```