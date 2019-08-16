# FREDE

This repository is a reference implemenatation for the Space-Efficient and Anytime-Optimal Graph Embeddings paper.

## Usage

To compute Personalized Page-Rank (PPR) matrix used as a similarity measure in
the experiments, install `pprlib` using:

    cd cpp & ./make.sh

First, to compute PPR use `computeppr.py` script.
To run calculate embeddings run `embed.py`:

```python
python embed.py fd POS log 128 embs
```

To run donstream tasks (classification, link prediction) run `classify.py` and
`linkpred.py` scripts:

```python
python classify.py POS deepwalk embs/pos.bin 0.1 0.9 0.2 results
```
