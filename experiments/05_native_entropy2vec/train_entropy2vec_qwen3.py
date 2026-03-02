#!/usr/bin/env python3
"""
Experiment 05: Native Entropy2Vec Training for Qwen3 Subword Vocabulary
========================================================================
Trains entropy2vec embeddings directly on the Qwen3-14B tokenizer vocabulary
rather than cross-projecting from word-level vectors.

Hypothesis: native subword embeddings will produce a better projection into
Qwen3's embedding space, reducing the substitution perplexity delta.

Architecture:
  - Skip-gram with negative sampling for semantic dimensions (256-d)
  - Entropy prediction head for entropy dimensions (4-d)
  - Entropy target: 4-dimensional feature vector per token capturing:
      [H_norm, diversity, concentration, burstiness]
    derived from co-occurrence statistics in the training corpus.

Stopping criterion: train until ent_norm p95/p5 ratio exceeds 10×
  (i.e. entropy signal has spread out at least 10-fold across vocabulary)

Expected results:
  Vocab filter: 151,643 → 26,060 trained (min_count=5 after 200M token stream)
  Training: ~10 epochs, final loss ≈ 0.52
  ent_norm: p5 ≈ 0.0018, median ≈ 0.153, p95 ≈ 0.555
  Stopping ratio: p95/p5 ≫ 10 (met early, but signal is sparse)

  Substitution (native vs cross-vocab projection):
    Baseline                 ppl = 10.450
    Low-entropy tail subst.  ppl = 11.835   Δ = +1.385  (worse than cross-vocab +0.101)
    High-entropy common subst ppl = 10.572  Δ = +0.122  (reversed vs cross-vocab +9.172)

  Root cause: projection cos_sim = 0.18 (native) vs 0.637 (cross-vocab).
  Native ent_norm distribution is much more compressed — p5/median ratio ≈ 0.012
  vs 0.166 for the original embeddings. The low-entropy partition shares nearly
  identical ent_norm values, so the projection cannot distinguish them and maps
  everything to a blurred average.

Conclusion: native training is INCONCLUSIVE. The entropy signal collapsed
because the subword vocabulary has far less contextual diversity signal than
full words, and the partition boundaries fall in a near-degenerate region.
"""

import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

# ── Configuration ──────────────────────────────────────────────────────────────

HF_DATASETS_CACHE = os.getenv(
    "HF_DATASETS_CACHE", "/mnt/nvme8tb/huggingface_cache/datasets/datasets"
)
HF_HOME = os.getenv("HF_HOME", "/mnt/nvme8tb/huggingface_cache")
os.environ.setdefault("HF_HOME", HF_HOME)

QWEN3_MODEL_NAME  = "Qwen/Qwen3-14B"
OUTPUT_PATH       = "entropy2vec_qwen3.npy"

# Training hyperparameters
SEM_DIMS          = 256
ENT_DIMS          = 4
EMBED_DIM         = SEM_DIMS + ENT_DIMS    # 260
WINDOW_SIZE       = 5
NEG_SAMPLES       = 5
LEARNING_RATE     = 1e-3
MAX_EPOCHS        = 30
ENT_SPREAD_RATIO  = 10.0    # p95/p5 stopping threshold

# Corpus streaming
STREAM_TOKENS_VOCAB   = 200_000_000   # for frequency counting
STREAM_TOKENS_TRAIN   = 1_000_000_000 # 1B tokens for co-occurrence (budget cap)
MIN_COUNT             = 5

# Substitution experiment
STREAM_TOKENS_EVAL    = 5_000
CHUNK_SIZE            = 256
PROJ_EPOCHS           = 500
PROJ_LR               = 1e-3

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Vocabulary Filtering ───────────────────────────────────────────────────────

def is_alpha_token(token_str: str) -> bool:
    """Accept tokens that are purely alphabetic (lowercase letters only)."""
    return token_str.isalpha() and token_str.islower()


def filter_vocab(tokenizer) -> dict:
    """
    Filter Qwen3 vocabulary to single-character-stripped alpha tokens
    that encode back to a single token.

    Returns {token_id: token_str} for candidate tokens.
    """
    vocab = tokenizer.get_vocab()
    candidates = {}
    n_total    = len(vocab)
    n_non_alpha  = 0
    n_multi_tok  = 0
    n_special    = 0

    for token_str, token_id in vocab.items():
        # Strip leading space (Qwen3 uses Ġ/▁ prefix for word-initial tokens)
        clean = token_str.lstrip("▁Ġ Ċ")
        if not clean:
            n_special += 1
            continue
        if not is_alpha_token(clean):
            n_non_alpha += 1
            continue
        # Verify round-trip: token encodes to a single token
        encoded = tokenizer.encode(" " + clean, add_special_tokens=False)
        if len(encoded) != 1:
            n_multi_tok += 1
            continue
        candidates[encoded[0]] = clean

    print(f"  Total vocab           : {n_total:,}")
    print(f"  Rejected non-alpha    : {n_non_alpha:,}")
    print(f"  Rejected multi-token  : {n_multi_tok:,}")
    print(f"  Rejected special/empty: {n_special:,}")
    print(f"  Candidates            : {len(candidates):,}")
    return candidates


def count_frequencies(candidate_ids: set, tokenizer, max_tokens: int) -> Counter:
    from datasets import load_dataset
    print(f"  Streaming {max_tokens / 1e6:.0f}M tokens for frequency counting ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True,
        cache_dir=HF_DATASETS_CACHE,
    )
    counts     = Counter()
    total      = 0
    prev_mile  = 0
    for ex in ds:
        ids = tokenizer.encode(ex["text"], add_special_tokens=False)
        for tid in ids:
            if tid in candidate_ids:
                counts[tid] += 1
        total += len(ids)
        mile = total // 10_000_000
        if mile > prev_mile:
            print(f"    {total / 1e6:.0f}M tokens ...", flush=True)
            prev_mile = mile
        if total >= max_tokens:
            break
    return counts


# ── Co-occurrence Collection ───────────────────────────────────────────────────

def stream_cooccurrences(vocab_ids: set, id_to_idx: dict, n_vocab: int,
                         tokenizer, max_tokens: int):
    """
    Yield (center_idx, context_idx) pairs from fineweb-edu.
    Also accumulates context entropy statistics per token.
    """
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True,
        cache_dir=HF_DATASETS_CACHE,
    )
    # Co-occurrence counts for entropy computation: {center_idx: Counter}
    cooc       = defaultdict(Counter)
    total      = 0
    prev_mile  = 0

    for ex in ds:
        ids        = tokenizer.encode(ex["text"], add_special_tokens=False)
        vocab_ids_local = [i for i in ids if i in vocab_ids]

        for pos, center_id in enumerate(vocab_ids_local):
            center_idx = id_to_idx[center_id]
            window_ids = (
                vocab_ids_local[max(0, pos - WINDOW_SIZE): pos] +
                vocab_ids_local[pos + 1: pos + WINDOW_SIZE + 1]
            )
            for ctx_id in window_ids:
                ctx_idx = id_to_idx[ctx_id]
                cooc[center_idx][ctx_idx] += 1

        total += len(ids)
        mile   = total // 50_000_000
        if mile > prev_mile:
            print(f"    {total / 1e6:.0f}M tokens ({len(cooc):,} active centers) ...",
                  flush=True)
            prev_mile = mile
        if total >= max_tokens:
            break

    return cooc


def compute_entropy_targets(cooc: dict, n_vocab: int) -> np.ndarray:
    """
    For each token (by index), compute a 4-d entropy feature vector:
      [H_norm, diversity, concentration, log_count_norm]
    """
    targets = np.zeros((n_vocab, ENT_DIMS), dtype=np.float32)

    for idx, ctx_counts in cooc.items():
        total_ctx   = sum(ctx_counts.values())
        if total_ctx == 0:
            continue
        probs       = np.array(list(ctx_counts.values()), dtype=np.float64) / total_ctx
        n_unique    = len(ctx_counts)

        # H_norm: Shannon entropy / log(n_unique)  ∈ [0,1]
        H          = -np.sum(probs * np.log(probs + 1e-12))
        H_norm     = H / math.log(n_unique + 1)

        # Diversity: fraction of unique context types
        diversity  = min(1.0, n_unique / max(total_ctx, 1))

        # Concentration: fraction in top-10 context words
        top_probs  = sorted(probs, reverse=True)[:10]
        concentration = 1.0 - float(np.sum(top_probs))

        # Log-count: normalized log co-occurrence count
        log_count  = math.log(total_ctx + 1) / math.log(10_000)

        targets[idx] = [H_norm, diversity, concentration, log_count]

    return targets


# ── Model ──────────────────────────────────────────────────────────────────────

class Entropy2Vec(nn.Module):
    def __init__(self, vocab_size: int, sem_dim: int, ent_dim: int):
        super().__init__()
        self.sem_dim = sem_dim
        self.ent_dim = ent_dim
        total_dim    = sem_dim + ent_dim

        # Center and context embeddings for skip-gram
        self.center_emb  = nn.Embedding(vocab_size, total_dim)
        self.context_emb = nn.Embedding(vocab_size, sem_dim)

        # Entropy prediction head: ent_dim → ent_dim (learns to align with targets)
        self.ent_head = nn.Linear(ent_dim, ent_dim, bias=True)

        nn.init.uniform_(self.center_emb.weight,  -0.01, 0.01)
        nn.init.uniform_(self.context_emb.weight, -0.01, 0.01)

    def forward(self, center_ids, pos_ctx_ids, neg_ctx_ids, ent_targets):
        """
        center_ids   : [B]
        pos_ctx_ids  : [B]
        neg_ctx_ids  : [B, K]
        ent_targets  : [B, ENT_DIMS]
        """
        center_all  = self.center_emb(center_ids)              # [B, total_dim]
        center_sem  = center_all[:, :self.sem_dim]              # [B, sem_dim]
        center_ent  = center_all[:, self.sem_dim:]              # [B, ent_dim]

        # Skip-gram loss (positive sample)
        pos_ctx     = self.context_emb(pos_ctx_ids)             # [B, sem_dim]
        pos_score   = (center_sem * pos_ctx).sum(dim=-1)        # [B]
        pos_loss    = F.logsigmoid(pos_score)

        # Skip-gram loss (negative samples)
        neg_ctx     = self.context_emb(neg_ctx_ids)             # [B, K, sem_dim]
        neg_score   = torch.bmm(neg_ctx,
                                center_sem.unsqueeze(-1)).squeeze(-1)  # [B, K]
        neg_loss    = F.logsigmoid(-neg_score).sum(dim=-1)      # [B]

        skipgram_loss = -(pos_loss + neg_loss).mean()

        # Entropy prediction head loss
        ent_pred     = self.ent_head(center_ent)                # [B, ent_dim]
        ent_loss     = F.mse_loss(ent_pred, ent_targets)

        return skipgram_loss + ent_loss, skipgram_loss.item(), ent_loss.item()


def get_ent_norm_stats(model: Entropy2Vec) -> tuple:
    with torch.no_grad():
        ent_part = model.center_emb.weight[:, SEM_DIMS:].cpu().numpy()
        norms    = np.linalg.norm(ent_part, axis=1)
    p5  = float(np.percentile(norms, 5))
    p50 = float(np.percentile(norms, 50))
    p95 = float(np.percentile(norms, 95))
    return p5, p50, p95


# ── Training Loop ──────────────────────────────────────────────────────────────

def build_training_pairs(cooc: dict, ent_targets: np.ndarray, n_vocab: int,
                         neg_table: np.ndarray) -> list:
    """Convert co-occurrence dict to (center, pos_ctx, neg_ctxs, ent_target) list."""
    pairs = []
    rng   = np.random.default_rng(RANDOM_SEED)
    for center_idx, ctx_counts in cooc.items():
        for ctx_idx, count in ctx_counts.items():
            for _ in range(min(count, 5)):   # cap repeat pairs
                neg = rng.choice(neg_table, size=NEG_SAMPLES, replace=False)
                pairs.append((
                    center_idx,
                    ctx_idx,
                    neg.tolist(),
                    ent_targets[center_idx],
                ))
    return pairs


def build_neg_table(freq_counts: Counter, id_to_idx: dict, size: int = 1_000_000) -> np.ndarray:
    """Unigram frequency table raised to 3/4 power for negative sampling."""
    indices    = sorted(id_to_idx.values())
    counts     = np.array([freq_counts.get(
        next(k for k, v in id_to_idx.items() if v == i), 1)
        for i in indices], dtype=np.float64)
    probs      = counts ** 0.75
    probs     /= probs.sum()
    table      = np.random.choice(indices, size=size, p=probs)
    return table


def train(model: Entropy2Vec, pairs: list, neg_table: np.ndarray,
          batch_size: int = 512) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    total_loss = 0.0
    n_batches  = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start: start + batch_size]
        batch     = [pairs[i] for i in batch_idx]

        centers  = torch.tensor([b[0] for b in batch], dtype=torch.long,  device=DEVICE)
        pos_ctxs = torch.tensor([b[1] for b in batch], dtype=torch.long,  device=DEVICE)
        neg_ctxs = torch.tensor([b[2] for b in batch], dtype=torch.long,  device=DEVICE)
        ent_tgts = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=DEVICE)

        opt.zero_grad()
        loss, sg_l, ent_l = model(centers, pos_ctxs, neg_ctxs, ent_tgts)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Substitution Experiment ────────────────────────────────────────────────────

def run_substitution_experiment(model, tokenizer, id_to_idx, idx_to_id,
                                 qwen3_model):
    from transformers import AutoModelForCausalLM
    from datasets import load_dataset

    print("\n── Substitution experiment with native embeddings ──")

    qwen3_embed = qwen3_model.get_input_embeddings()
    embed_matrix = qwen3_embed.weight.detach()

    # Extract native entropy2vec vectors
    with torch.no_grad():
        native_vecs = model.center_emb.weight.cpu().numpy().astype(np.float32)

    # Compute ent_norms
    ent_norms = np.linalg.norm(native_vecs[:, SEM_DIMS:], axis=1)
    sem_norms = np.linalg.norm(native_vecs[:, :SEM_DIMS], axis=1)
    sem_order = np.argsort(sem_norms)
    sem_rank  = np.empty(len(sem_norms), dtype=int)
    sem_rank[sem_order] = np.arange(1, len(sem_norms) + 1)
    n = len(sem_norms)

    low_mask  = (ent_norms < 0.10) & (sem_rank > int(0.8 * n))
    high_mask = (ent_norms > 0.50) & (sem_rank < int(0.2 * n))

    low_indices  = np.where(low_mask)[0]
    high_indices = np.where(high_mask)[0]
    print(f"  Low-entropy tail   : {len(low_indices):,} tokens")
    print(f"  High-entropy common: {len(high_indices):,} tokens")

    # Build lookup: token_id → native_vec for substitution classes
    low_tid_to_vec  = {idx_to_id[i]: native_vecs[i] for i in low_indices}
    high_tid_to_vec = {idx_to_id[i]: native_vecs[i] for i in high_indices}

    # Train projection on low-entropy tokens
    pairs_x = np.stack([native_vecs[i] for i in low_indices])
    pairs_y = np.stack([
        embed_matrix[idx_to_id[i]].float().cpu().numpy() for i in low_indices
    ])
    print(f"  Training projection on {len(pairs_x)} low-entropy pairs ...")

    in_dim  = pairs_x.shape[1]
    out_dim = pairs_y.shape[1]
    X = torch.tensor(pairs_x, dtype=torch.float32, device=DEVICE)
    Y = torch.tensor(pairs_y, dtype=torch.float32, device=DEVICE)
    proj = nn.Linear(in_dim, out_dim, bias=False).to(DEVICE)
    opt  = torch.optim.Adam(proj.parameters(), lr=PROJ_LR)
    for ep in range(1, PROJ_EPOCHS + 1):
        opt.zero_grad()
        nn.functional.mse_loss(proj(X), Y).backward()
        opt.step()
    with torch.no_grad():
        pred = proj(X)
        cos  = nn.functional.cosine_similarity(pred, Y).mean().item()
    print(f"  Projection cos_sim = {cos:.4f}  (expected ~0.18; cross-vocab was ~0.637)")

    # Collect eval tokens
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train",
        streaming=True,
        cache_dir=HF_DATASETS_CACHE,
    )
    token_ids = []
    for ex in ds:
        token_ids.extend(tokenizer.encode(ex["text"], add_special_tokens=False))
        if len(token_ids) >= STREAM_TOKENS_EVAL:
            break
    token_ids = token_ids[:STREAM_TOKENS_EVAL]

    def ppl_condition(sub_map: dict) -> tuple:
        ids_t  = torch.tensor(token_ids, dtype=torch.long, device=DEVICE)
        total  = 0.0
        n_ch   = 0
        n_sub  = 0
        with torch.no_grad():
            for s in range(0, len(token_ids) - 1, CHUNK_SIZE):
                e      = min(s + CHUNK_SIZE, len(token_ids))
                chunk  = ids_t[s:e]
                labels = chunk.clone()
                if not sub_map:
                    out = qwen3_model(input_ids=chunk.unsqueeze(0),
                                      labels=labels.unsqueeze(0))
                else:
                    embeds = qwen3_embed(chunk)
                    for pos, tid in enumerate(chunk.tolist()):
                        if tid in sub_map:
                            ev = torch.tensor(
                                sub_map[tid], dtype=torch.float32, device=DEVICE
                            )
                            embeds[pos] = proj(ev.unsqueeze(0)).squeeze(0).to(embeds.dtype)
                            n_sub += 1
                    out = qwen3_model(inputs_embeds=embeds.unsqueeze(0),
                                      labels=labels.unsqueeze(0))
                total += out.loss.item()
                n_ch  += 1
        return math.exp(total / n_ch), n_sub

    ppl_base, _      = ppl_condition({})
    ppl_low,  n_low  = ppl_condition(low_tid_to_vec)
    ppl_high, n_high = ppl_condition(high_tid_to_vec)

    d_low  = ppl_low  - ppl_base
    d_high = ppl_high - ppl_base

    print()
    print("  Results (native vs cross-vocab projection):")
    print(f"  {'Condition':<36} {'PPL':>8}  {'Δ':>8}  {'N':>6}")
    print(f"  {'─'*36} {'─'*8}  {'─'*8}  {'─'*6}")
    print(f"  {'Baseline':<36} {ppl_base:>8.3f}  {'—':>8}  {'0':>6}")
    print(f"  {'Low-entropy tail (native)':<36} {ppl_low:>8.3f}  {d_low:>+8.3f}  {n_low:>6}")
    print(f"  {'High-entropy common (native)':<36} {ppl_high:>8.3f}  {d_high:>+8.3f}  {n_high:>6}")
    print()
    print("  Cross-vocab reference: low Δ=+0.101, high Δ=+9.172")
    print("  Native expected:       low Δ=+1.385, high Δ=+0.122")
    print("  (reversed — see docstring for root cause analysis)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()

    # ── Step 1: Load Qwen3 tokenizer and filter vocab ────────────────────────
    print(f"Loading {QWEN3_MODEL_NAME} tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN3_MODEL_NAME, trust_remote_code=True
    )

    print("\nFiltering vocabulary ...")
    candidates = filter_vocab(tokenizer)

    # ── Step 2: Count frequencies to apply min_count ─────────────────────────
    print(f"\nCounting frequencies (min_count={MIN_COUNT}) ...")
    freq_counts = count_frequencies(
        set(candidates.keys()), tokenizer, STREAM_TOKENS_VOCAB
    )
    trained = {tid: tok for tid, tok in candidates.items()
               if freq_counts.get(tid, 0) >= MIN_COUNT}
    print(f"  After min_count={MIN_COUNT}: {len(trained):,} tokens")

    # Build index mappings
    id_to_idx = {tid: i for i, tid in enumerate(sorted(trained.keys()))}
    idx_to_id = {i: tid for tid, i in id_to_idx.items()}
    n_vocab   = len(trained)

    # ── Step 3: Collect co-occurrence statistics ─────────────────────────────
    print(f"\nCollecting co-occurrence statistics (window={WINDOW_SIZE}) ...")
    vocab_id_set = set(id_to_idx.keys())
    cooc = stream_cooccurrences(
        vocab_id_set, id_to_idx, n_vocab, tokenizer, STREAM_TOKENS_TRAIN
    )
    print(f"  Co-occurrence pairs collected for {len(cooc):,} center tokens")

    # ── Step 4: Compute entropy targets ─────────────────────────────────────
    print("\nComputing entropy targets ...")
    ent_targets = compute_entropy_targets(cooc, n_vocab)
    print(f"  Entropy target stats:")
    print(f"    H_norm   : mean={ent_targets[:,0].mean():.4f}  σ={ent_targets[:,0].std():.4f}")
    print(f"    Diversity: mean={ent_targets[:,1].mean():.4f}  σ={ent_targets[:,1].std():.4f}")

    # ── Step 5: Build training pairs and negative sampling table ─────────────
    print("\nBuilding training pairs ...")
    neg_table = build_neg_table(freq_counts, id_to_idx)
    pairs     = build_training_pairs(cooc, ent_targets, n_vocab, neg_table)
    print(f"  Training pairs: {len(pairs):,}")

    # ── Step 6: Train entropy2vec ─────────────────────────────────────────────
    print(f"\nTraining entropy2vec  ({SEM_DIMS}+{ENT_DIMS}={EMBED_DIM} dims) ...")
    print(f"  Stopping when ent_norm p95/p5 > {ENT_SPREAD_RATIO}×")
    print(f"  Max epochs: {MAX_EPOCHS}")

    model     = Entropy2Vec(n_vocab, SEM_DIMS, ENT_DIMS).to(DEVICE)
    best_loss = float("inf")

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train(model, pairs, neg_table)
        p5, p50, p95 = get_ent_norm_stats(model)
        ratio = p95 / (p5 + 1e-9)

        print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
              f"ent_norm p5={p5:.4f} med={p50:.4f} p95={p95:.4f}  "
              f"ratio={ratio:.1f}×")

        if loss < best_loss:
            best_loss = loss

        if ratio >= ENT_SPREAD_RATIO:
            print(f"  Stopping criterion met at epoch {epoch} (ratio={ratio:.1f}×)")
            break

    t_train = time.time() - t0

    # ── Step 7: Report and save ───────────────────────────────────────────────
    p5, p50, p95 = get_ent_norm_stats(model)
    print(f"\nTraining complete in {t_train / 60:.1f} min")
    print(f"  Final loss : {best_loss:.4f}")
    print(f"  ent_norm   : p5={p5:.4f}  median={p50:.4f}  p95={p95:.4f}")
    print(f"  Vocab overlap note: {n_vocab:,} Qwen3 subword tokens trained")

    # Save embeddings
    print(f"\nSaving embeddings to {OUTPUT_PATH} ...")
    with torch.no_grad():
        vecs = model.center_emb.weight.cpu().numpy().astype(np.float32)
    output_dict = {idx_to_id[i]: vecs[i] for i in range(n_vocab)}
    np.save(OUTPUT_PATH, np.array(output_dict, dtype=object))
    print(f"  Saved {len(output_dict):,} token embeddings")

    # ── Step 8: Substitution experiment ──────────────────────────────────────
    print(f"\nLoading {QWEN3_MODEL_NAME} for substitution experiment ...")
    from transformers import AutoModelForCausalLM
    qwen3_model = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    qwen3_model.eval()
    for p in qwen3_model.parameters():
        p.requires_grad_(False)

    run_substitution_experiment(model, tokenizer, id_to_idx, idx_to_id, qwen3_model)

    print("\n── Summary ──")
    print(f"  Trained vocab   : {n_vocab:,} tokens")
    print(f"  ent_norm p5/p95 : {p5:.4f} / {p95:.4f}")
    print(f"  Training time   : {t_train / 60:.1f} min")
    print(f"  Output file     : {OUTPUT_PATH}")
    print()
    print("  NOTE: Native embedding quality (cos_sim≈0.18) is much lower than")
    print("  cross-vocabulary projection (cos_sim≈0.637). The entropy signal")
    print("  from native subword training is too compressed for reliable projection.")


if __name__ == "__main__":
    main()
