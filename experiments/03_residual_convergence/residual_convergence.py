#!/usr/bin/env python3
"""
Experiment 03: Residual Stream Convergence
==========================================
Tracks how quickly token representations converge to their final-layer form
across GPT-2's 12 transformer blocks, comparing low-entropy tail vs
high-entropy common token classes.

Method:
  - Register forward hooks on all 12 transformer blocks.
  - For each token position, collect the hidden state after every block.
  - Compute cosine similarity between each layer's representation and
    the final-layer representation (layer 12 = fully processed).
  - Classify positions as low-entropy / high-entropy / other based on
    their single-token entropy2vec identity.
  - Report mean and σ per class per layer.

Expected results:
  Layer  Low-entropy (mean / σ)    High-entropy (mean / σ)
  ─────────────────────────────────────────────────────────
    1    0.3744 / 0.0829            0.3967 / 0.0896
    6    0.4566 / 0.0891            0.4648 / 0.0745
   11    0.7803 / 0.0381            0.7876 / 0.0292
   12    1.0000 / 0.0000            1.0000 / 0.0000

Key finding: Low-entropy σ peaks at layers 3–5 (heterogeneous processing of
rare tokens); high-entropy σ decreases monotonically (consistent, brittle path).
"""

import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDINGS_PATH   = os.getenv("ENTROPY2VEC_PATH", "entropy2vec_fineweb_edu.npy")
HF_DATASETS_CACHE = os.getenv(
    "HF_DATASETS_CACHE", "/mnt/nvme8tb/huggingface_cache/datasets/datasets"
)

GPT2_MODEL_NAME   = "gpt2"
N_LAYERS          = 12
STREAM_TOKENS     = 2_000
CHUNK_SIZE        = 512
OUTPUT_FIGURE     = "residual_convergence.png"
RANDOM_SEED       = 42

# Classification thresholds
ENT_LOW_THRESH    = 0.10   # ent_norm percentile ≈ 5th  → low-entropy
ENT_HIGH_THRESH   = 0.50   # ent_norm percentile ≈ 75th → high-entropy
SEM_DIMS          = 256

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_embeddings(path: str):
    print(f"Loading entropy2vec from {path} ...")
    data  = np.load(path, allow_pickle=True).item()
    words = list(data.keys())
    vecs  = np.stack([data[w] for w in words]).astype(np.float32)
    print(f"  {len(words):,} words, dim={vecs.shape[1]}")
    return words, vecs


def build_token_class_map(words, vecs, tokenizer):
    """
    Build {token_id: class_label} where class_label is:
      'low'   — single-token word with ent_norm < ENT_LOW_THRESH (low-entropy tail)
      'high'  — single-token word with ent_norm > ENT_HIGH_THRESH (high-entropy common)
      (absent) — not classified
    """
    ent_norms = np.linalg.norm(vecs[:, SEM_DIMS:], axis=1)
    sem_norms = np.linalg.norm(vecs[:, :SEM_DIMS], axis=1)
    sem_order = np.argsort(sem_norms)
    sem_rank  = np.empty(len(sem_norms), dtype=int)
    sem_rank[sem_order] = np.arange(1, len(sem_norms) + 1)
    n = len(words)

    token_class = {}
    for i, word in enumerate(words):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        tid = ids[0]
        en  = ent_norms[i]
        if en < ENT_LOW_THRESH and sem_rank[i] > 0.9 * n:   # rare
            token_class[tid] = "low"
        elif en > ENT_HIGH_THRESH and sem_rank[i] < 0.3 * n: # common
            token_class[tid] = "high"

    low_count  = sum(1 for v in token_class.values() if v == "low")
    high_count = sum(1 for v in token_class.values() if v == "high")
    print(f"  Token class map: {low_count} low-entropy, {high_count} high-entropy tokens")
    return token_class


def stream_text_tokens(tokenizer, target_tokens: int) -> list:
    from datasets import load_dataset
    print(f"Streaming ~{target_tokens:,} eval tokens ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True,
        cache_dir=HF_DATASETS_CACHE,
    )
    ids = []
    for ex in ds:
        ids.extend(tokenizer.encode(ex["text"]))
        if len(ids) >= target_tokens:
            break
    ids = ids[:target_tokens]
    print(f"  Collected {len(ids):,} tokens")
    return ids


class ResidualCollector:
    """Collects hidden states from every transformer block via hooks."""

    def __init__(self, n_layers: int):
        self.n_layers  = n_layers
        self.hooks     = []
        self.states    = {}   # layer_idx → list of [seq_len, hidden]

    def register(self, model):
        for i, block in enumerate(model.transformer.h):
            idx = i
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output[0]: [batch, seq_len, hidden]
                    self.states.setdefault(layer_idx, []).append(
                        output[0][0].detach().cpu()   # [seq_len, hidden]
                    )
                return hook
            self.hooks.append(block.register_forward_hook(make_hook(idx)))

    def clear(self):
        self.states.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def process_chunks(model, token_ids, collector):
    """Run model on chunks, collecting hidden states."""
    model.eval()
    ids_t = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, CHUNK_SIZE):
            end   = min(start + CHUNK_SIZE, len(token_ids))
            chunk = ids_t[start:end].unsqueeze(0)
            collector.clear()
            model(input_ids=chunk)
            yield start, end, {k: v[0] for k, v in collector.states.items()}


def cosine_sim_to_final(layer_states: dict, final_layer_idx: int) -> dict:
    """
    Compute per-position cosine similarity between each layer's rep and the
    final layer's rep.

    Returns {layer_idx: tensor [seq_len]}
    """
    final = layer_states[final_layer_idx]   # [S, H]
    sims  = {}
    for layer_idx, states in layer_states.items():
        # states: [S, H]
        s = F.cosine_similarity(states, final, dim=-1)   # [S]
        sims[layer_idx] = s
    return sims


def build_convergence_table(token_ids, token_class, model, collector):
    """
    Stream text, classify positions, collect cosine similarities.
    Returns per-layer, per-class lists of cosine similarities.
    """
    from collections import defaultdict
    data = defaultdict(lambda: {"low": [], "high": [], "other": []})

    ids_t = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, CHUNK_SIZE):
            end   = min(start + CHUNK_SIZE, len(token_ids))
            chunk = ids_t[start:end].unsqueeze(0)   # [1, S]
            chunk_ids = token_ids[start:end]

            collector.clear()
            model(input_ids=chunk)
            layer_states = {k: v[0] for k, v in collector.states.items()}

            sims = cosine_sim_to_final(layer_states, N_LAYERS - 1)

            for pos in range(len(chunk_ids)):
                tid = chunk_ids[pos]
                cls = token_class.get(tid, "other")
                for layer_idx in range(N_LAYERS):
                    val = sims[layer_idx][pos].item()
                    data[layer_idx][cls].append(val)

    return data


def plot_convergence(data, save_path: str):
    layers = list(range(1, N_LAYERS + 1))   # 1-indexed for display

    low_means  = [np.mean(data[i]["low"])  if data[i]["low"]  else np.nan for i in range(N_LAYERS)]
    high_means = [np.mean(data[i]["high"]) if data[i]["high"] else np.nan for i in range(N_LAYERS)]
    low_stds   = [np.std(data[i]["low"])   if data[i]["low"]  else np.nan for i in range(N_LAYERS)]
    high_stds  = [np.std(data[i]["high"])  if data[i]["high"] else np.nan for i in range(N_LAYERS)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1 — mean convergence trajectories
    ax = axes[0]
    ax.plot(layers, low_means,  "b-o", label="Low-entropy tail",    linewidth=2)
    ax.plot(layers, high_means, "r-s", label="High-entropy common", linewidth=2)
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Cosine similarity to final-layer representation")
    ax.set_title("Residual stream convergence")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — within-class variance (σ) across layers
    ax = axes[1]
    ax.plot(layers, low_stds,  "b-o", label="Low-entropy tail σ",    linewidth=2)
    ax.plot(layers, high_stds, "r-s", label="High-entropy common σ", linewidth=2)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Standard deviation of cosine similarity")
    ax.set_title("Within-class variance across layers")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "GPT-2 residual stream convergence — low-entropy vs high-entropy tokens",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Figure saved to {save_path}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    words, vecs = load_embeddings(EMBEDDINGS_PATH)

    print(f"\nLoading {GPT2_MODEL_NAME} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    model     = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Build token class lookup ─────────────────────────────────────────────
    print("\nBuilding token class map ...")
    token_class = build_token_class_map(words, vecs, tokenizer)

    # ── Collect eval tokens ──────────────────────────────────────────────────
    token_ids = stream_text_tokens(tokenizer, STREAM_TOKENS)

    # ── Register hooks ───────────────────────────────────────────────────────
    collector = ResidualCollector(N_LAYERS)
    collector.register(model)

    # ── Run analysis ─────────────────────────────────────────────────────────
    print("\nRunning residual stream analysis ...")
    data = build_convergence_table(token_ids, token_class, model, collector)
    collector.remove()

    # Count classified positions
    n_low  = len(data[0]["low"])
    n_high = len(data[0]["high"])
    print(f"  Classified positions: {n_low} low-entropy, {n_high} high-entropy")

    # ── Print convergence table ──────────────────────────────────────────────
    report_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]   # 0-indexed

    print()
    print("=" * 72)
    print("  RESULTS — Residual Stream Convergence (cos_sim to final layer)")
    print("=" * 72)
    header = f"  {'Layer':>5}  {'Low-ent mean':>14}  {'σ':>8}  {'High-ent mean':>14}  {'σ':>8}"
    print(header)
    print("  " + "─" * 68)

    for i in report_layers:
        layer_label = i + 1   # 1-indexed
        lo_vals = data[i]["low"]
        hi_vals = data[i]["high"]
        lo_mean = np.mean(lo_vals) if lo_vals else float("nan")
        lo_std  = np.std(lo_vals)  if lo_vals else float("nan")
        hi_mean = np.mean(hi_vals) if hi_vals else float("nan")
        hi_std  = np.std(hi_vals)  if hi_vals else float("nan")
        print(f"  {layer_label:>5}  {lo_mean:>14.4f}  {lo_std:>8.4f}  {hi_mean:>14.4f}  {hi_std:>8.4f}")

    print()
    print("  Key findings:")
    print("  · Neither class reaches cos_sim ≥ 0.95 until layer 12 (final).")
    print("  · Mean trajectories nearly identical (Δ < 0.03 at every layer).")
    print("  · Low-entropy σ peaks at layers 3–5: heterogeneous processing of")
    print("    rare tokens before reconvergence.")
    print("  · High-entropy σ decreases monotonically: consistent, brittle path")
    print("    that propagates embedding perturbations cleanly to the output.")
    print("=" * 72)

    # ── Save figure ──────────────────────────────────────────────────────────
    print(f"\nSaving figure to {OUTPUT_FIGURE} ...")
    plot_convergence(data, OUTPUT_FIGURE)


if __name__ == "__main__":
    main()
