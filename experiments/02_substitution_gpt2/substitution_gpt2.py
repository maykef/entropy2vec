#!/usr/bin/env python3
"""
Experiment 02: GPT-2 Substitution Perplexity
=============================================
Tests whether replacing token embeddings with projected entropy2vec vectors
degrades language model perplexity differently for low-entropy vs high-entropy
token classes.

Method:
  1. Identify vocabulary partitions from entropy2vec space.
  2. Find tokens with single-subword GPT-2 matches.
  3. Train a linear projection  260 → 768  (MSE, Adam 500 epochs).
  4. Stream ~5000 tokens from fineweb-edu.
  5. Compute perplexity under three conditions:
       (a) Baseline              — original GPT-2 embeddings
       (b) Low-entropy subst.   — projected entropy2vec for low-entropy tail
       (c) High-entropy subst.  — projected entropy2vec for high-entropy common

Expected results:
  Baseline                 ppl = 43.814
  Low-entropy tail subst.  ppl = 43.916   Δ = +0.102  (155 tokens substituted)
  High-entropy common subst ppl = 80.128  Δ = +36.314 (465 tokens substituted)
  Per-token delta ratio    356×  (high / low)
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

GPT2_MODEL_NAME   = "gpt2"
STREAM_TOKENS     = 5_000          # target evaluation tokens
CHUNK_SIZE        = 512            # tokens per perplexity chunk
PROJ_EPOCHS       = 500
PROJ_LR           = 1e-3
RANDOM_SEED       = 42

# Vocab partition thresholds (matched to saved partitions)
LOW_ENT_THRESH    = 0.10           # ent_norm  < threshold  → low-entropy
HIGH_ENT_THRESH   = 0.50           # ent_norm  > threshold  → high-entropy
LOW_RANK_CUTOFF   = 150_000        # sem_norm rank (ascending) > this → rare
HIGH_RANK_CUTOFF  = 50_000         # sem_norm rank (ascending) < this → common

SEM_DIMS = 256
ENT_DIMS = 4

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_embeddings(path: str):
    print(f"Loading entropy2vec from {path} ...")
    data = np.load(path, allow_pickle=True).item()
    words = list(data.keys())
    vecs  = np.stack([data[w] for w in words]).astype(np.float32)
    print(f"  {len(words):,} words, dim={vecs.shape[1]}")
    return words, vecs


def partition_vocab(words, vecs):
    ent_norms = np.linalg.norm(vecs[:, SEM_DIMS:], axis=1)
    sem_norms = np.linalg.norm(vecs[:, :SEM_DIMS], axis=1)

    # sem_norm rank ascending: small sem_norm → common; large → rare
    sem_order  = np.argsort(sem_norms)            # index 0 = smallest sem_norm
    sem_rank   = np.empty(len(sem_norms), dtype=int)
    sem_rank[sem_order] = np.arange(1, len(sem_norms) + 1)

    low_mask  = (ent_norms < LOW_ENT_THRESH)  & (sem_rank > LOW_RANK_CUTOFF)
    high_mask = (ent_norms > HIGH_ENT_THRESH) & (sem_rank < HIGH_RANK_CUTOFF)

    low_words  = {words[i] for i in np.where(low_mask)[0]}
    high_words = {words[i] for i in np.where(high_mask)[0]}
    word_to_vec = {w: vecs[i] for i, w in enumerate(words)}

    print(f"  Low-entropy tail   : {len(low_words):,} words")
    print(f"  High-entropy common: {len(high_words):,} words")
    return low_words, high_words, word_to_vec


def find_single_token_words(vocab_set, tokenizer):
    """Return {word: token_id} for words that encode to exactly one GPT-2 token."""
    result = {}
    for word in vocab_set:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            result[word] = ids[0]
    return result


def train_projection(pairs_x: np.ndarray, pairs_y: np.ndarray) -> nn.Linear:
    """
    Train a linear 260→768 projection with MSE loss.
    pairs_x: (N, 260) entropy2vec vectors
    pairs_y: (N, 768) GPT-2 embedding vectors
    """
    print(f"  Training projection  {pairs_x.shape[1]}→{pairs_y.shape[1]}")
    print(f"  Training pairs: {len(pairs_x)}  |  epochs: {PROJ_EPOCHS}  |  lr: {PROJ_LR}")

    X = torch.tensor(pairs_x, dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(pairs_y, dtype=torch.float32, device=DEVICE)

    proj = nn.Linear(pairs_x.shape[1], pairs_y.shape[1], bias=False).to(DEVICE)
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
        pred = proj(X)
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
    token_ids = []
    for example in ds:
        ids = tokenizer.encode(example["text"])
        token_ids.extend(ids)
        if len(token_ids) >= target_tokens:
            break
    token_ids = token_ids[:target_tokens]
    print(f"  Collected {len(token_ids):,} tokens")
    return token_ids


def compute_perplexity(model, token_ids, embed_matrix, substitute_map,
                       word_to_vec, proj, tokenizer):
    """
    Compute perplexity over token_ids.

    substitute_map: {token_id: word}  — tokens to replace with projected vectors.
    If substitute_map is empty, uses standard input_ids path (baseline).
    """
    model.eval()
    ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
    total_loss = 0.0
    n_chunks   = 0
    n_subst    = 0

    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, CHUNK_SIZE):
            end    = min(start + CHUNK_SIZE, len(token_ids))
            chunk  = ids_tensor[start:end]             # [S]
            labels = chunk.clone()

            if not substitute_map:
                # Baseline: standard forward pass
                out  = model(input_ids=chunk.unsqueeze(0),
                             labels=labels.unsqueeze(0))
            else:
                # Substitution: build custom embedding sequence
                embeds = embed_matrix[chunk]            # [S, 768]
                for pos, tid in enumerate(chunk.tolist()):
                    if tid in substitute_map:
                        word = substitute_map[tid]
                        if word in word_to_vec:
                            ev  = torch.tensor(
                                word_to_vec[word], dtype=torch.float32, device=DEVICE
                            )
                            embeds[pos] = proj(ev.unsqueeze(0)).squeeze(0)
                            n_subst += 1
                out = model(inputs_embeds=embeds.unsqueeze(0),
                            labels=labels.unsqueeze(0))

            total_loss += out.loss.item()
            n_chunks   += 1

    ppl = math.exp(total_loss / n_chunks)
    return ppl, n_subst


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # ── Load embeddings and partition ────────────────────────────────────────
    words, vecs = load_embeddings(EMBEDDINGS_PATH)
    low_words, high_words, word_to_vec = partition_vocab(words, vecs)

    # ── Load GPT-2 ───────────────────────────────────────────────────────────
    print(f"\nLoading {GPT2_MODEL_NAME} ...")
    tokenizer  = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    model      = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME).to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    embed_matrix = model.transformer.wte.weight.detach()   # [50257, 768]
    print(f"  Embedding matrix: {embed_matrix.shape}")

    # ── Find single-token matches ────────────────────────────────────────────
    print("\nFinding single-token GPT-2 matches ...")
    low_tok  = find_single_token_words(low_words,  tokenizer)
    high_tok = find_single_token_words(high_words, tokenizer)
    print(f"  Low-entropy  matches : {len(low_tok):,} / {len(low_words):,}")
    print(f"  High-entropy matches : {len(high_tok):,} / {len(high_words):,}")

    # ── Train projection on low-entropy subset ───────────────────────────────
    print("\nBuilding projection training set ...")
    pair_words = [w for w in low_tok if w in word_to_vec]
    pairs_x    = np.stack([word_to_vec[w] for w in pair_words])          # (N, 260)
    pairs_y    = np.stack([
        embed_matrix[low_tok[w]].cpu().numpy() for w in pair_words
    ])                                                                     # (N, 768)
    print(f"  Training pairs: {len(pair_words)}")

    proj = train_projection(pairs_x, pairs_y)

    # Build token-id → word lookup maps
    low_sub_map  = {tid: w for w, tid in low_tok.items()  if w in word_to_vec}
    high_sub_map = {tid: w for w, tid in high_tok.items() if w in word_to_vec}

    # ── Collect eval tokens ──────────────────────────────────────────────────
    token_ids = stream_text_tokens(tokenizer, STREAM_TOKENS)

    # ── Three perplexity conditions ──────────────────────────────────────────
    print("\nEvaluating perplexity ...")

    ppl_base, _       = compute_perplexity(model, token_ids, embed_matrix, {},
                                            word_to_vec, proj, tokenizer)
    ppl_low,  n_low   = compute_perplexity(model, token_ids, embed_matrix, low_sub_map,
                                            word_to_vec, proj, tokenizer)
    ppl_high, n_high  = compute_perplexity(model, token_ids, embed_matrix, high_sub_map,
                                            word_to_vec, proj, tokenizer)

    delta_low  = ppl_low  - ppl_base
    delta_high = ppl_high - ppl_base

    # Per-token delta ratio
    ppt_low  = delta_low  / max(n_low,  1)
    ppt_high = delta_high / max(n_high, 1)
    ratio    = ppt_high / ppt_low if ppt_low > 0 else float("inf")

    # ── Results table ────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  RESULTS — GPT-2 Substitution Perplexity")
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
    print("  Expected: baseline≈43.8  low≈43.9  high≈80.1  ratio≈356×")
    print("=" * 62)


if __name__ == "__main__":
    main()
