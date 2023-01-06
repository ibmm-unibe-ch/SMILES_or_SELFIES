""" only testing so far
SMILES or SELFIES, 2022
"""

from pathlib import Path

import torch
from transformers import BartForConditionalGeneration, BartModel, BartTokenizer

from constants import PROCESSED_PATH, SEED, TOKENIZER_PATH, VAL_SIZE
from tokenisation import get_tokenizer

model = BartForConditionalGeneration.from_pretrained(
    Path("huggingface_models/selfies_trained"), forced_bos_token_id=0
)
tok = get_tokenizer(TOKENIZER_PATH / "SELFIES")
example_english_phrase = "[C][N][C][N]"
batch = tok(example_english_phrase, return_tensors="pt")
print(batch["attention_mask"])
generated_ids = model.generate(batch["input_ids"])
print(len(generated_ids[0]))
print(tok.convert_ids_to_tokens(generated_ids[0]))
