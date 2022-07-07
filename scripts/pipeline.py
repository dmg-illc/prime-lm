import argparse
import gc
import os
import time
from pprint import pprint
from copy import deepcopy

import torch
from torch.nn.functional import log_softmax, softmax
from tqdm import tqdm

import diagnnose
from diagnnose.models import LanguageModel, import_model
from diagnnose.tokenizer.create import create_tokenizer
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor

from init_corpora import init_corpora


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_activation_reader(corpus, sen_column):
    """Extract activations from corpus and return activation_reader"""

    if model.is_causal:

        def selection_func(w_idx, item):
            sen_len = len(getattr(item, sen_column))
            start_idx = getattr(item, COLUMNS[sen_column])

            # Shifted 1 position back to account for auto-regressive nature
            return (start_idx - 1) <= w_idx <= (sen_len - 2)

    else:

        def selection_func(w_idx, item):
            sen_len = len(getattr(item, sen_column))
            start_idx = getattr(item, COLUMNS[sen_column])

            return start_idx <= w_idx <= (sen_len - 1)

    config_dict["extract"]["batch_size"] = 128

    corpus.sen_column = sen_column
    extractor = Extractor(
        model, corpus, selection_func=selection_func, **config_dict["extract"]
    )
    activation_reader = extractor.extract()

    return activation_reader


def fetch_perplexity(corpus_item, hidden_state, sen_attr):
    """Calculate probabilities and perplexities for the hidden sentence"""
    sen = getattr(corpus_item, sen_attr)

    lm_decoder = model.decoder

    with torch.no_grad():
        logits = lm_decoder(hidden_state.to(DEVICE))
    probs = log_softmax(logits, dim=-1)
    start_idx = getattr(corpus_item, COLUMNS[sen_attr])
    token_ids = [tokenizer._convert_token_to_id(tok) for tok in sen[start_idx:]]
    probs = probs[range(probs.size(0)), token_ids]

    sen_prob = torch.sum(probs)

    sen_len = len(sen) - getattr(corpus_item, COLUMNS[sen_attr])
    perplexity = torch.exp(sen_prob) ** (-1 / sen_len)

    return perplexity.item(), sen_prob.item()


def fetch_all_corpus_scores(corpus_path, corpus_subset_size):
    # Add new corpus to config_dict
    config_dict["corpus"] = {
        "path": corpus_path,
        "header_from_first_line": True,
        "sen_column": next(iter(COLUMNS)),
        "tokenize_columns": list(COLUMNS.keys()),
        "convert_numerical": True,
        "sep": ",",
    }

    corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])

    # The corpus_scores that are writting to file contain the (un)primed sentences themselves, as well as log probs + perplexity scores.
    corpus_scores = [
        ",".join(
            list(COLUMNS.keys())
            + [f"logp_{col}" for col in COLUMNS]
            + [f"ppl_{col}" for col in COLUMNS]
        )
    ]

    for idx in tqdm(range(0, len(corpus), corpus_subset_size)):
        corpus_subset = corpus.slice(range(idx, idx + corpus_subset_size))

        corpus_subset._attach_sen_ids()

        fetch_corpus_scores(
            corpus_scores,
            corpus_subset,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()

    return corpus_scores


def fetch_corpus_scores(corpus_scores, corpus):
    sens = {k: [] for k in COLUMNS}
    ppls = {k: [] for k in COLUMNS}
    logps = {k: [] for k in COLUMNS}

    ## ACTIVATION EXTRACTION
    for column in COLUMNS:
        activation_reader = create_activation_reader(corpus, column)

        for item, state in zip(corpus, activation_reader[:]):
            ppl, logp = fetch_perplexity(item, state, column)

            tokenized_sen = tokenizer.convert_tokens_to_string(getattr(item, column))

            sens[column].append(tokenized_sen)
            ppls[column].append(ppl)
            logps[column].append(logp)

    corpus_scores.extend(
        [
            ",".join([str(x) for x in item_scores])
            for item_scores in zip(*sens.values(), *logps.values(), *ppls.values())
        ]
    )


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--model", type=str)
    arg_parser.add_argument("--data", type=str)
    arg_parser.add_argument("--save", type=str)

    args = arg_parser.parse_args()

    mode = "masked_lm" if "bert" in args.model else "causal_lm"

    config_dict = {
        "model": {"transformer_type": args.model, "mode": mode, "device": DEVICE},
        "tokenizer": {
            "path": args.model,
        },
        "extract": {
            "batch_size": 256,
            "activation_names": [(-1, "hx")],
        },
    }

    if mode == "masked_lm":
        config_dict["model"]["compute_pseudo_ll"] = True

    models = args.model.split(" ")
    for model_name in models:
        print(model_name.upper())
        config_dict["model"]["transformer_type"] = model_name
        config_dict["tokenizer"]["path"] = model_name

        model = import_model(**config_dict["model"])
        tokenizer = create_tokenizer(**config_dict["tokenizer"])

        primed_corpora, COLUMNS = init_corpora(args.data, tokenizer)

        score_dir = os.path.join(args.save, f"{model_name.split('/')[-1]}_scores")
        if not os.path.isdir(score_dir):
            os.mkdir(score_dir)

        init_time = time.time()

        for CORPUS, corpus_path in sorted(list(primed_corpora.items())):
            print(CORPUS)

            corpus_scores = fetch_all_corpus_scores(corpus_path, 512)

            scores_file = os.path.join(score_dir, f"{CORPUS}_scores.csv")
            with open(scores_file, "w", encoding="utf-8") as f:
                f.write("\n".join(corpus_scores))

    end_time = time.time()
    print(f"PIPELINE FINISHED IN {end_time-start_time:.2f} SECONDS")
    print(
        f"INIT TOOK {init_time-start_time:.2f}s, EXTRACTION {end_time-init_time:.2f}s"
    )
