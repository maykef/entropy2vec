#!/usr/bin/env python3
"""
Experiment 04: Qwen3-14B Substitution Perplexity
=================================================
Replicates Experiment 02 on Qwen3-14B (bfloat16, no quantization) to test
whether the low-entropy / high-entropy asymmetry is model-agnostic.

Method: identical to experiment 02 except:
  - Model: Qwen/Qwen3-14B (trust_remote_code=True, bfloat16)
  - Projection: 260 → 5120
  - Tokenizer uses space-prefix convention for single-token matching

Expected results:
  Baseline                  ppl = 15.121
  Low-entropy tail subst.   ppl = 15.222   Δ = +0.101  (167 tokens)
  High-entropy common subst ppl = 24.293   Δ = +9.172  (530 tokens)
  Per-token delta ratio  91×  (high / low)

Memory report (bfloat16):
  Embedding matrix:       1483.8 MB  (151936 × 5120 × 2 bytes)
  641 replaceable rows:      6.3 MB
  Projection (260→5120):     2.6 MB  (2 × 260 × 5120 × 2 bytes)
  Net saving:                3.7 MB  (0.25% of embedding matrix)

Note: GPT-2's high-entropy Δ was +36.314 (+83%); Qwen3-14B's is +9.172 (+61%).
Qwen3-14B is more robust to high-entropy substitutions — likely due to its
larger context window and deeper residual processing before output projection.
"""

import os, math, sys
import numpy as np
import torch
import torch.nn as nn

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDINGS_PATH   = os.getenv("ENTROPY2VEC_PATH", "entropy2vec_fineweb_edu.npy")
HF_DATASETS_CACHE = os.getenv(
    "HF_DATASETS_CACHE", "/mnt/nvme8tb/huggingface_cache/datasets/datasets"
)
HF_HOME           = os.getenv("HF_HOME", "/mnt/nvme8tb/huggingface_cache")
os.environ.setdefault("HF_HOME", HF_HOME)

QWEN3_MODEL_NAME  = "Qwen/Qwen3-14B"
STREAM_TOKENS     = 5_000
CHUNK_SIZE        = 256            # smaller chunks for Qwen3's larger hidden size
PROJ_EPOCHS       = 500
PROJ_LR           = 1e-3
RANDOM_SEED       = 42

LOW_ENT_THRESH    = 0.10
HIGH_ENT_THRESH   = 0.50
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


def partition_vocab(words, vecs):
    ent_norms = np.linalg.norm(vecs[:, SEM_DIMS:], axis=1)
    sem_norms = np.linalg.norm(vecs[:, :SEM_DIMS], axis=1)

    # sem_norm ascending rank ≈ frequency rank (smaller = more common)
    sem_order = np.argsort(sem_norms)
    sem_rank  = np.empty(len(sem_norms), dtype=int)
    sem_rank[sem_order] = np.arange(1, len(sem_norms) + 1)
    n = len(words)

    low_mask  = (ent_norms < LOW_ENT_THRESH)  & (sem_rank > int(0.9 * n))
    high_mask = (ent_norms > HIGH_ENT_THRESH) & (sem_rank < int(0.3 * n))

    low_words  = {words[i] for i in np.where(low_mask)[0]}
    high_words = {words[i] for i in np.where(high_mask)[0]}
    word_to_vec = {w: vecs[i] for i, w in enumerate(words)}

    print(f"  Low-entropy tail   : {len(low_words):,} words")
    print(f"  High-entropy common: {len(high_words):,} words")
    return low_words, high_words, word_to_vec


def find_single_token_words(vocab_set, tokenizer):
    """Return {word: token_id} using space-prefix encoding convention."""
    result = {}
    for word in vocab_set:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            result[word] = ids[0]
    return result


def train_projection(pairs_x: np.ndarray, pairs_y: np.ndarray) -> nn.Linear:
    in_dim, out_dim = pairs_x.shape[1], pairs_y.shape[1]
    print(f"  Training projection  {in_dim}→{out_dim}")
    print(f"  Training pairs: {len(pairs_x)}  |  epochs: {PROJ_EPOCHS}  |  lr: {PROJ_LR}")

    X = torch.tensor(pairs_x, dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(pairs_y, dtype=torch.float32, device=DEVICE)

    proj = nn.Linear(in_dim, out_dim, bias=False).to(DEVICE)
    opt  = torch.optim.Adam(proj.parameters(), lr=PROJ_LR)

    for epoch in range(1, PROJ_EPOCHS + 1):
        opt.zero_grad()
        loss = nn.functional.mse_loss(proj(X), Y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                pred = proj(X)
                cos  = nn.functional.cosine_similarity(pred, Y).mean().item()
            print(f"    epoch {epoch:4d}  loss={loss.item():.6f}  cos_sim={cos:.4f}")

    with torch.no_grad():
        pred      = proj(X)
        final_cos = nn.functional.cosine_similarity(pred, Y).mean().item()
        final_mse = nn.functional.mse_loss(pred, Y).item()
    print(f"  Final  cos_sim={final_cos:.4f}  MSE={final_mse:.6f}")
    return proj


def stream_text_tokens(tokenizer, target_tokens: int) -> list:
    from datasets import load_dataset
    print(f"Streaming ~{target_tokens:,} eval tokens from fineweb-edu ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True,
        cache_dir=HF_DATASETS_CACHE,
    )
    ids = []
    for ex in ds:
        ids.extend(tokenizer.encode(ex["text"], add_special_tokens=False))
        if len(ids) >= target_tokens:
            break
    ids = ids[:target_tokens]
    print(f"  Collected {len(ids):,} tokens")
    return ids


def compute_perplexity(model, token_ids, embed_layer, substitute_map,
                       word_to_vec, proj):
    model.eval()
    ids_t      = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
    total_loss = 0.0
    n_chunks   = 0
    n_subst    = 0

    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, CHUNK_SIZE):
            end    = min(start + CHUNK_SIZE, len(token_ids))
            chunk  = ids_t[start:end]
            labels = chunk.clone()

            if not substitute_map:
                out = model(input_ids=chunk.unsqueeze(0),
                            labels=labels.unsqueeze(0))
            else:
                # Build custom embedding sequence
                embeds = embed_layer(chunk)           # [S, 5120]
                for pos, tid in enumerate(chunk.tolist()):
                    if tid in substitute_map:
                        word = substitute_map[tid]
                        if word in word_to_vec:
                            ev       = torch.tensor(
                                word_to_vec[word], dtype=torch.float32, device=DEVICE
                            )
                            embeds[pos] = proj(ev.unsqueeze(0)).squeeze(0).to(embeds.dtype)
                            n_subst += 1
                out = model(inputs_embeds=embeds.unsqueeze(0),
                            labels=labels.unsqueeze(0))

            total_loss += out.loss.item()
            n_chunks   += 1

    ppl = math.exp(total_loss / n_chunks)
    return ppl, n_subst


def memory_report(model, embed_layer, low_tok, proj):
    """Print memory statistics for the embedding replacement."""
    # Embedding matrix: vocab_size × hidden × sizeof(bfloat16)
    vocab_size, hidden = embed_layer.weight.shape
    bytes_per_param    = 2   # bfloat16
    matrix_mb          = vocab_size * hidden * bytes_per_param / 1024**2
    rows_mb            = len(low_tok) * hidden * bytes_per_param / 1024**2

    proj_params = sum(p.numel() for p in proj.parameters())
    proj_mb     = proj_params * bytes_per_param / 1024**2

    net_saving_mb  = rows_mb - proj_mb
    saving_percent = 100 * net_saving_mb / matrix_mb

    print()
    print("  Memory report (bfloat16):")
    print(f"    Embedding matrix ({vocab_size:,} × {hidden})  : {matrix_mb:8.1f} MB")
    print(f"    {len(low_tok):,} replaceable rows              : {rows_mb:8.1f} MB")
    print(f"    Projection ({proj.in_features}→{proj.out_features})    : {proj_mb:8.2f} MB")
    print(f"    Net saving                           : {net_saving_mb:8.1f} MB  ({saving_percent:.2f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    words, vecs = load_embeddings(EMBEDDINGS_PATH)
    low_words, high_words, word_to_vec = partition_vocab(words, vecs)

    # ── Load Qwen3-14B ───────────────────────────────────────────────────────
    print(f"\nLoading {QWEN3_MODEL_NAME} (bfloat16) ...")
    print("  This requires ~28 GB VRAM. Download takes several minutes first time.")

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN3_MODEL_NAME,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    embed_layer = model.get_input_embeddings()   # model-agnostic API
    print(f"  Embedding layer: {embed_layer.weight.shape}")

    # ── Find single-token matches ────────────────────────────────────────────
    print("\nFinding single-token Qwen3 matches ...")
    low_tok  = find_single_token_words(low_words,  tokenizer)
    high_tok = find_single_token_words(high_words, tokenizer)
    print(f"  Low-entropy  matches : {len(low_tok):,} / {len(low_words):,}")
    print(f"  High-entropy matches : {len(high_tok):,} / {len(high_words):,}")

    # ── Train projection on low-entropy subset ───────────────────────────────
    print("\nBuilding projection training set ...")
    pair_words = [w for w in low_tok if w in word_to_vec]
    pairs_x    = np.stack([word_to_vec[w] for w in pair_words])
    pairs_y    = np.stack([
        embed_layer.weight[low_tok[w]].float().cpu().numpy()
        for w in pair_words
    ])
    print(f"  Training pairs: {len(pair_words)}")

    proj = train_projection(pairs_x, pairs_y)

    low_sub_map  = {tid: w for w, tid in low_tok.items()  if w in word_to_vec}
    high_sub_map = {tid: w for w, tid in high_tok.items() if w in word_to_vec}

    # ── Collect eval tokens ──────────────────────────────────────────────────
    token_ids = stream_text_tokens(tokenizer, STREAM_TOKENS)

    # ── Three perplexity conditions ──────────────────────────────────────────
    print("\nEvaluating perplexity ...")

    ppl_base, _      = compute_perplexity(model, token_ids, embed_layer, {},
                                           word_to_vec, proj)
    ppl_low,  n_low  = compute_perplexity(model, token_ids, embed_layer, low_sub_map,
                                           word_to_vec, proj)
    ppl_high, n_high = compute_perplexity(model, token_ids, embed_layer, high_sub_map,
                                           word_to_vec, proj)

    delta_low  = ppl_low  - ppl_base
    delta_high = ppl_high - ppl_base
    ppt_low    = delta_low  / max(n_low,  1)
    ppt_high   = delta_high / max(n_high, 1)
    ratio      = ppt_high / ppt_low if ppt_low > 0 else float("inf")

    # ── Results table ────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  RESULTS — Qwen3-14B Substitution Perplexity")
    print("=" * 62)
    print(f"  {'Condition':<36} {'Perplexity':>10}  {'Δ baseline':>10}  {'N subst':>7}")
    print(f"  {'─'*36} {'─'*10}  {'─'*10}  {'─'*7}")
    print(f"  {'Baseline':<36} {ppl_base:>10.3f}  {'—':>10}  {'0':>7}")
    print(f"  {'Low-entropy tail substitution':<36} {ppl_low:>10.3f}  {delta_low:>+10.3f}  {n_low:>7}")
    print(f"  {'High-entropy common substitution':<36} {ppl_high:>10.3f}  {delta_high:>+10.3f}  {n_high:>7}")
    print(f"  {'─'*36} {'─'*10}  {'─'*10}  {'─'*7}")
    print(f"\n  Δ per substituted token (low)  : {ppt_low:+.5f}")
    print(f"  Δ per substituted token (high) : {ppt_high:+.5f}")
    print(f"  Per-token delta ratio (high/low): {ratio:.0f}×")
    print()
    print("  Expected: baseline≈15.1  low≈15.2  high≈24.3  ratio≈91×")

    memory_report(model, embed_layer, low_tok, proj)
    print("=" * 62)


if __name__ == "__main__":
    main()
