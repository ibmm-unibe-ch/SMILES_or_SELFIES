""" Tokenisation 
SMILES or SELFIES, 2022
"""

from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
import pandas as pd

SMILES = pd.read_csv("processed/10m_dataframe.csv", usecols=[210])
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
tk_tokenizer = SentencePieceBPETokenizer()
tk_tokenizer.train_from_iterator(
    SMILES.values,
    vocab_size=4000,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens,
)
tk_tokenizer.save("./tokenizer/base_tokenizer")

tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer)
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
# and save for later!
tokenizer.save_pretrained("./tokenizer/fast_tokenizer")
