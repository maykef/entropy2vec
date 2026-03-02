# entropy2vec-inference

Experiments testing whether entropy-aware token embeddings carry signal
independent of frequency, and whether that signal can be exploited as an
inference optimization in frozen pretrained language models.

The embeddings are 260-d `entropy2vec` vectors trained on FineWeb-Edu with a
skip-gram objective. The first 256 dimensions are semantic; the last 4 encode
entropy (contextual predictability). Two token classes are used throughout:

- **Low-entropy tail**: `ent_norm < 0.10` and rare. Syntactically constrained,
  rare-but-predictable tokens. ~2,600 tokens.
- **High-entropy common**: `ent_norm > 0.50` and frequent. Semantically rich,
  highly polysemous common tokens. ~48,000 tokens.

`ent_norm` is the L2 norm of the last 4 embedding dimensions.

---

## Repository layout

```
entropy2vec-inference/
├── experiments/
│   ├── 01_diagnostic/
│   │   └── entropy_diagnostic.py       # Verify entropy signal separates token classes
│   ├── 02_substitution_gpt2/
│   │   └── substitution_gpt2.py        # Replace GPT-2 embeddings with projected entropy2vec vectors
│   ├── 03_residual_convergence/
│   │   └── residual_convergence.py     # Track residual stream variance layer-by-layer in GPT-2
│   ├── 04_substitution_qwen3/
│   │   └── substitution_qwen3.py       # Same substitution experiment on Qwen3-14B
│   └── 05_native_entropy2vec/
│       └── train_entropy2vec_qwen3.py  # Train entropy2vec directly on Qwen3's subword vocabulary
├── results/
│   └── summary_table.md                # Per-experiment results tables with commentary
├── requirements.txt
└── LICENSE
```

---

## Running the experiments

Set environment variables first:

```bash
export ENTROPY2VEC_PATH=/path/to/entropy2vec_fineweb_edu.npy
export HF_DATASETS_CACHE=/path/to/hf/datasets
export HF_HOME=/path/to/hf/models
```

Then run any experiment from the repo root:

```bash
python experiments/01_diagnostic/entropy_diagnostic.py
python experiments/02_substitution_gpt2/substitution_gpt2.py
python experiments/03_residual_convergence/residual_convergence.py
python experiments/04_substitution_qwen3/substitution_qwen3.py
python experiments/05_native_entropy2vec/train_entropy2vec_qwen3.py
```

Each script is self-contained. Experiments 04 and 05 require a GPU with at
least 24 GB VRAM to run Qwen3-14B in bfloat16. Experiment 05 is the longest
(approximately 30 minutes for corpus streaming and training).

---

## Requirements

Python 3.10+, PyTorch >= 2.0, HuggingFace `transformers` and `datasets`.
Full pinned versions in `requirements.txt`.

```bash
pip install -r requirements.txt
```
