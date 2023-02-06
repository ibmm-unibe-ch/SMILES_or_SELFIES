import argparse
import pickle


def pickle_object(objekt, path):
    with open(path, "wb") as openfile:
        pickle.dump(objekt, openfile)


def unpickle(path):
    with open(path, "rb") as openfile:
        objekt = pickle.load(openfile)
    return objekt


def parse_arguments(cuda=False, tokenizer=False, task=False):
    parser = argparse.ArgumentParser()
    if cuda:
        parser.add_argument("--cuda", required=True, help="VISIBLE_CUDA_DEVICE")
    if task:
        parser.add_argument(
            "--task", required=True, help="Which specific task as string."
        )
    if tokenizer:
        parser.add_argument(
            "--tokenizer_prefix",
            choices=[
                "smiles_atom",
                "smiles_sentencepiece",
                "selfies_atom",
                "selfies_sentencepiece",
                "smiles_isomers_atom",
                "smiles_isomers_sentencepiece",
                "selfies_isomers_atom",
                "selfies_isomers_sentencepiece",
            ],
        )
    args = parser.parse_args()
    return vars(args)
