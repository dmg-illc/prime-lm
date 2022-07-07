
## Structural Persistence in Language Models
**Paper:** [![](https://img.shields.io/badge/arxiv-2109.14989.pdf-green)](https://arxiv.org/pdf/2109.14989.pdf)

This repository contains the pipeline and data for the TACL paper _*Structural Persistence in Language Models: Priming as a Window into Abstract Language Representations*_, by Arabella Sinclair, Jaap Jumelet, Willem Zuidema, and Raquel Fernández.

---
**Pipeline**
The pipeline can be run from the command line as follows, using the script found in the `scripts` folder:

`pipeline.py --model "gpt2-large" --data data --save scores`

With `--model` a Huggingface `transformer` model, `--data` pointing towards the PrimeLM data that can be found in this repo ([here](https://github.com/dmg-illc/prime-lm/tree/main/PrimeLM/corpora)), and `--save` the directory where the priming performance will be stored.
