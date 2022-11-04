## Structural Persistence in Language Models
**Paper:** [![](https://img.shields.io/badge/arxiv-2109.14989.pdf-green)](https://arxiv.org/pdf/2109.14989.pdf)

This repository contains the pipeline and data for the TACL paper _*Structural Persistence in Language Models: Priming as a Window into Abstract Language Representations*_, by Arabella Sinclair\*, Jaap Jumelet\*, Willem Zuidema, and Raquel Fernández.

:tada: The paper will be presented at EMNLP 2022!

---
### Pipeline
The pipeline can be run from the command line as follows, using the script found in the `scripts` folder:

`python3 pipeline.py --model "gpt2-large" --data data --save scores`

With `--model` a Huggingface `transformer` model, `--data` pointing towards the `corpora` folder of PrimeLM data that can be found in this repo ([here](https://github.com/dmg-illc/prime-lm/tree/main/PrimeLM/corpora)), and `--save` the directory where the priming performance will be stored.

The pipeline returns a `.csv` file for each corpus type. This file contains the original prime/target pairs in the 4 prime/target configurations (X->X, X->Y, Y->Y, Y->X), as well as the sentence log probabilities and perplexities of each prime/target configuration. The priming effect can than easily be computed by subtracting the log probabilities of the incongruent configuration from the congruent configurations, for example using `pandas`:

```
import pandas as pd

df = pd.read_csv('scores.csv')
pe_x = df.logp_x_px - df.logp_x_py
mean_pe = pe_x.mean()
```
---
### Citation
The paper can be cited as follows:
```
@article{10.1162/tacl_a_00504,
    author = {Sinclair, Arabella and Jumelet, Jaap and Zuidema, Willem and Fernández, Raquel},
    title = "{Structural Persistence in Language Models: Priming as a Window into Abstract Language Representations}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {1031-1050},
    year = {2022},
    month = {09},
    abstract = "{We investigate the extent to which modern neural language models are susceptible to structural priming, the phenomenon whereby the structure of a sentence makes the same structure more probable in a follow-up sentence. We explore how priming can be used to study the potential of these models to learn abstract structural information, which is a prerequisite for good performance on tasks that require natural language understanding skills. We introduce a novel metric and release Prime-LM, a large corpus where we control for various linguistic factors that interact with priming strength. We find that Transformer models indeed show evidence of structural priming, but also that the generalizations they learned are to some extent modulated by semantic information. Our experiments also show that the representations acquired by the models may not only encode abstract sequential structure but involve certain level of hierarchical syntactic information. More generally, our study shows that the priming paradigm is a useful, additional tool for gaining insights into the capacities of language models and opens the door to future priming-based investigations that probe the model’s internal states.1}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00504},
    url = {https://doi.org/10.1162/tacl\_a\_00504},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00504/2043729/tacl\_a\_00504.pdf},
}
```

