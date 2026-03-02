# entropy2vec-inference

## What this is

A research investigation into whether entropy-aware token embeddings carry
signal independent of frequency, and whether that signal can be exploited
as an inference optimization in frozen pretrained language models.

The embeddings used are `entropy2vec` vectors trained on the FineWeb-Edu corpus
with a skip-gram objective that separates semantic dimensions (first 256) from
entropy dimensions (last 4). Each 260-d vector encodes both meaning and
contextual predictability.

---

## Findings summary

### The +0.1 constant

Replacing low-entropy tail token embeddings with projected entropy2vec vectors
costs **+0.101–0.102 perplexity** across GPT-2 and Qwen3-14B — identical to
three decimal places despite an 18× difference in embedding dimension and
completely different architectures. This appears to be a property of the token
class, not any specific model.

### The 356× ratio

High-entropy common token substitution costs **356× more perplexity per
substituted token** than low-entropy tail substitution on GPT-2, and 91× on
Qwen3-14B. The two classes require fundamentally different treatment.

### The variance structure

Low-entropy tail tokens show a variance **peak at layers 3–5** of GPT-2's
residual stream — the model briefly diverges processing paths for different
instances of rare tokens before reconverging. High-entropy common tokens show
monotonically decreasing variance — consistent, brittle processing that
propagates embedding perturbations cleanly to the output.

### The honest conclusion

The token class distinction is real, stable, and model-agnostic. Exploiting it
for inference efficiency in existing models is not possible — the embedding
lookup is O(1) and is not the bottleneck. Attention and FFN layers process every
token regardless of class. Meaningful exploitation requires an architecture
designed around token class branching **from training**.

---

## Experiments

| # | Experiment | Key result |
|---|-----------|------------|
| 1 | [Entropy diagnostic](experiments/01_diagnostic/entropy_diagnostic.py) | ρ=0.25, MI=0.026, Cohen's d=1.16 |
| 2 | [GPT-2 substitution](experiments/02_substitution_gpt2/substitution_gpt2.py) | Low Δ=+0.102, High Δ=+36.314, ratio=356× |
| 3 | [Residual convergence](experiments/03_residual_convergence/residual_convergence.py) | Variance peak L3–5 for low-entropy class |
| 4 | [Qwen3-14B substitution](experiments/04_substitution_qwen3/substitution_qwen3.py) | Low Δ=+0.101, High Δ=+9.172, ratio=91× |
| 5 | [Native subword training](experiments/05_native_entropy2vec/train_entropy2vec_qwen3.py) | Signal collapsed, cos_sim=0.18, inconclusive |

Full results with commentary: [results/summary_table.md](results/summary_table.md)

---

## Vocabulary partitions

The entropy2vec embedding space supports two natural partitions:

- **Low-entropy tail**: `ent_norm < 0.10` AND rare (`sem_norm` large). These
  are syntactically constrained, rare-but-predictable words. 2,631 words.
- **High-entropy common**: `ent_norm > 0.50` AND frequent (`sem_norm` small).
  These are semantically rich, highly polysemous common words. 48,293 words.

`ent_norm` = L2 norm of the last 4 embedding dimensions.
`sem_norm` proxy for rarity: larger semantic norm ≈ rarer word (empirically
validated against fineweb-edu frequency counts; ρ ≈ 0.25).

---

## Running the experiments

All scripts are self-contained. Run from the repo root or from within each
experiment directory. Set environment variables as needed:

```bash
# Point to your entropy2vec embeddings
export ENTROPY2VEC_PATH=/path/to/entropy2vec_fineweb_edu.npy

# Point to your HuggingFace datasets cache
export HF_DATASETS_CACHE=/mnt/nvme8tb/huggingface_cache/datasets/datasets

# Point to your HuggingFace model cache (for Qwen3 experiments)
export HF_HOME=/mnt/nvme8tb/huggingface_cache

python experiments/01_diagnostic/entropy_diagnostic.py
python experiments/02_substitution_gpt2/substitution_gpt2.py
python experiments/03_residual_convergence/residual_convergence.py
python experiments/04_substitution_qwen3/substitution_qwen3.py
python experiments/05_native_entropy2vec/train_entropy2vec_qwen3.py
```

Experiments 04 and 05 require a GPU with ≥24 GB VRAM for Qwen3-14B in bfloat16.
Experiment 05 is the longest (≈30 min for corpus streaming + training).

---

## Requirements

```
Python 3.10+
PyTorch ≥ 2.0
HuggingFace transformers, datasets
numpy, scipy, matplotlib, scikit-learn, tqdm
```

See [requirements.txt](requirements.txt) for pinned versions.
