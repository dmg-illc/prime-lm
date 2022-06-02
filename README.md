
## Structural Persistence in Language Models
**Paper:** [![](https://img.shields.io/badge/arxiv-2109.14989.pdf-green)](https://arxiv.org/pdf/2109.14989.pdf)

This repository contains the pipeline and data for the paper _Structural Persistence in Language Models: Priming as a Window into Abstract Language Representations_, by Arabella Sinclair, Jaap Jumelet, Willem Zuidema, and Raquel Fernández.

---
**Pipeline**
The pipeline can be run from the command line as follows:

`pipeline.py --model "gpt2-large" --data data --save scores`

With `--model` a Huggingface `transformer` model, `--data` pointing towards the PrimeLM data that can be found in this repo, and `--save` the directory where the priming performance will be stored.
