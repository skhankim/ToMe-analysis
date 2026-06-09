"""
Token reduction algorithms for E5 (Table 2 reproduction).

All operate on ViT-L/16 MAE, r=8, same setup as Table 1.
Algorithms:
  - prune_random      : drop r random tokens
  - prune_attn        : drop r tokens with least attention received (Kim et al. 2021 style)
  - kmeans2 / kmeans5 : cluster N->N-r centroids, 2 or 5 Lloyd iterations, merge clusters
  - greedy            : greedily merge most-similar pair, repeat r times (no replacement)
  - bipartite         : ToMe's bipartite_soft_matching (reference)

Each returns a new token tensor [B, N-r, C] and updated size [B, N-r, 1].
cls token (index 0) is always protected and kept at output index 0.
"""

import torch

from tome.merge import bipartite_soft_matching, merge_wavg


def _wavg_scatter(x, size, groups, n_out):
    """
    Average x weighted by size into n_out groups given group assignment per token.
    groups: [B, N] long in [0, n_out). Returns merged [B,n_out,C], size [B,n_out,1].
    """
    B, N, C = x.shape
    idx = groups.unsqueeze(-1)
    xs = x * size
    out = torch.zeros(B, n_out, C, device=x.device, dtype=x.dtype)
    out.scatter_add_(1, idx.expand(B, N, C), xs)
    out_s = torch.zeros(B, n_out, 1, device=x.device, dtype=x.dtype)
    out_s.scatter_add_(1, idx.expand(B, N, 1), size)
    out = out / out_s.clamp(min=1e-6)
    return out, out_s


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_random(x, size, metric, attn, r, class_token=True):
    B, N, C = x.shape
    keep = N - r
    scores = torch.rand(B, N, device=x.device)
    if class_token:
        scores[:, 0] = float("inf")  # always keep cls
    keep_idx = scores.argsort(dim=1, descending=True)[:, :keep]
    keep_idx, _ = keep_idx.sort(dim=1)  # preserve order, cls first
    out = x.gather(1, keep_idx.unsqueeze(-1).expand(B, keep, C))
    out_s = size.gather(1, keep_idx.unsqueeze(-1).expand(B, keep, 1))
    return out, out_s


def prune_attn(x, size, metric, attn, r, class_token=True):
    """attn: [B, N] attention received per token (summed over queries/heads)."""
    B, N, C = x.shape
    keep = N - r
    scores = attn.clone()
    if class_token:
        scores[:, 0] = float("inf")
    keep_idx = scores.argsort(dim=1, descending=True)[:, :keep]
    keep_idx, _ = keep_idx.sort(dim=1)
    out = x.gather(1, keep_idx.unsqueeze(-1).expand(B, keep, C))
    out_s = size.gather(1, keep_idx.unsqueeze(-1).expand(B, keep, 1))
    return out, out_s


# ---------------------------------------------------------------------------
# kmeans clustering
# ---------------------------------------------------------------------------

def _kmeans(x, size, metric, attn, r, iters, class_token=True):
    B, N, C = x.shape
    n_out = N - r
    m = metric / metric.norm(dim=-1, keepdim=True)

    # init centroids: evenly spaced tokens (deterministic)
    init = torch.linspace(0, N - 1, n_out, device=x.device).long()
    centroids = m.gather(1, init.view(1, -1, 1).expand(B, n_out, m.shape[-1]).clone())

    groups = None
    for _ in range(iters):
        # assign each token to nearest centroid (cosine)
        cn = centroids / centroids.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sim = m @ cn.transpose(-1, -2)  # [B,N,n_out]
        groups = sim.argmax(dim=-1)  # [B,N]
        # force cls to its own cluster 0 to protect it
        if class_token:
            groups[:, 0] = 0
        # update centroids = mean of assigned token metrics
        new_c = torch.zeros_like(centroids)
        cnt = torch.zeros(B, n_out, 1, device=x.device)
        new_c.scatter_add_(1, groups.unsqueeze(-1).expand(B, N, m.shape[-1]), m)
        cnt.scatter_add_(1, groups.unsqueeze(-1).expand(B, N, 1),
                         torch.ones(B, N, 1, device=x.device))
        nonempty = cnt.squeeze(-1) > 0
        new_c = torch.where(nonempty.unsqueeze(-1), new_c / cnt.clamp(min=1e-6), centroids)
        centroids = new_c

    out, out_s = _wavg_scatter(x, size, groups, n_out)
    return out, out_s


def kmeans2(x, size, metric, attn, r, class_token=True):
    return _kmeans(x, size, metric, attn, r, iters=2, class_token=class_token)


def kmeans5(x, size, metric, attn, r, class_token=True):
    return _kmeans(x, size, metric, attn, r, iters=5, class_token=class_token)


# ---------------------------------------------------------------------------
# kmeans variants for paper strawman test (E6)
# ---------------------------------------------------------------------------

def _kmeans_lloyd(m, centroids, max_iters, class_token=True):
    """Lloyd iterations. Stops early if assignment stable. Returns groups."""
    B, N, D = m.shape
    n_out = centroids.shape[1]
    groups_prev = None
    for _ in range(max_iters):
        cn = centroids / centroids.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        sim = m @ cn.transpose(-1, -2)
        groups = sim.argmax(dim=-1)
        if class_token:
            groups[:, 0] = 0
        if groups_prev is not None and (groups == groups_prev).all():
            break
        groups_prev = groups
        new_c = torch.zeros_like(centroids)
        cnt = torch.zeros(B, n_out, 1, device=m.device, dtype=m.dtype)
        new_c.scatter_add_(1, groups.unsqueeze(-1).expand(B, N, D), m)
        cnt.scatter_add_(1, groups.unsqueeze(-1).expand(B, N, 1),
                         torch.ones(B, N, 1, device=m.device, dtype=m.dtype))
        nonempty = cnt.squeeze(-1) > 0
        centroids = torch.where(nonempty.unsqueeze(-1),
                                new_c / cnt.clamp(min=1e-6), centroids)
    return groups


def kmeans_random5(x, size, metric, attn, r, class_token=True):
    """Random token-position init + 5 Lloyd iter. Paper-style strawman estimate.

    NOTE: cluster 0 centroid is NOT initialized from cls token. Combined with
    cls force-assignment in _kmeans_lloyd, this contaminates the cls cluster
    across 24 layers. Use kmeans_random5_clsfix for fair "random init kmeans"
    comparison.
    """
    B, N, C = x.shape
    n_out = N - r
    m = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    idx_list = [torch.randperm(N, device=m.device)[:n_out] for _ in range(B)]
    init_idx = torch.stack(idx_list)
    centroids = m.gather(1, init_idx.unsqueeze(-1).expand(B, n_out, m.shape[-1]))
    groups = _kmeans_lloyd(m, centroids, max_iters=5, class_token=class_token)
    return _wavg_scatter(x, size, groups, n_out)


def kmeans_random5_clsfix(x, size, metric, attn, r, class_token=True):
    """Random init with cluster 0 = cls embedding. Fair test of "random init kmeans"
    when cls protection is correctly handled.

    Cluster 0 centroid = K[cls]. Remaining n_out-1 centroids = random token K's
    drawn from [1, N). Lloyd 5 iter, cls forced to cluster 0.

    Hypothesis: matches paper kmeans 80.2 if init is the dominant factor.
    """
    B, N, C = x.shape
    n_out = N - r
    m = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    if class_token:
        # First centroid index = 0 (cls). Remaining = random from [1, N).
        idx_list = []
        for _ in range(B):
            rest = torch.randperm(N - 1, device=m.device)[:n_out - 1] + 1
            full = torch.cat([torch.zeros(1, device=m.device, dtype=torch.long), rest])
            idx_list.append(full)
        init_idx = torch.stack(idx_list)
    else:
        idx_list = [torch.randperm(N, device=m.device)[:n_out] for _ in range(B)]
        init_idx = torch.stack(idx_list)
    centroids = m.gather(1, init_idx.unsqueeze(-1).expand(B, n_out, m.shape[-1]))
    groups = _kmeans_lloyd(m, centroids, max_iters=5, class_token=class_token)
    return _wavg_scatter(x, size, groups, n_out)


def kmeans_kpp_full(x, size, metric, attn, r, class_token=True):
    """Greedy max-min cosine-distance init (kmeans++ approx) + full convergence.

    Best-effort kmeans baseline. Init picks each next centroid as the token
    farthest (in cosine distance) from any already-picked centroid. Then Lloyd
    iter until assignment stable (max 20).

    Hypothesis: if this approaches bipartite (84.25), paper's "kmeans fails by
    cluster collapse" claim is a strawman that better init resolves.
    """
    B, N, C = x.shape
    n_out = N - r
    D = metric.shape[-1]
    m = metric / metric.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    centroids = torch.empty(B, n_out, D, device=m.device, dtype=m.dtype)
    centroids[:, 0] = m[:, 0]  # token 0 (cls if class_token)
    max_sim = (m * centroids[:, 0:1]).sum(dim=-1)  # [B, N]: max sim to picked set
    for k in range(1, n_out):
        farthest = max_sim.argmin(dim=-1)  # [B]: token least similar to picked
        bi = torch.arange(B, device=m.device)
        new_c = m[bi, farthest]
        centroids[:, k] = new_c
        new_sim = (m * new_c.unsqueeze(1)).sum(dim=-1)
        max_sim = torch.max(max_sim, new_sim)
    groups = _kmeans_lloyd(m, centroids, max_iters=20, class_token=class_token)
    return _wavg_scatter(x, size, groups, n_out)


# ---------------------------------------------------------------------------
# Greedy matching (sequential)
# ---------------------------------------------------------------------------

def greedy(x, size, metric, attn, r, class_token=True):
    B, N, C = x.shape
    m = metric / metric.norm(dim=-1, keepdim=True)
    scores = m @ m.transpose(-1, -2)  # [B,N,N]
    diag = torch.eye(N, device=x.device, dtype=torch.bool)
    scores.masked_fill_(diag.unsqueeze(0), float("-inf"))
    if class_token:
        scores[:, 0, :] = float("-inf")
        scores[:, :, 0] = float("-inf")

    # group id per token; merged tokens share the group of their target
    group = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N).clone()
    alive = torch.ones(B, N, dtype=torch.bool, device=x.device)

    for _ in range(r):
        flat = scores.view(B, -1)
        best = flat.argmax(dim=1)
        i = best // N
        j = best % N
        bi = torch.arange(B, device=x.device)
        # merge i into j: relabel i's group to j's group
        gi = group[bi, i]
        gj = group[bi, j]
        # all tokens with group gi -> gj
        relabel = group == gi.unsqueeze(1)
        group = torch.where(relabel, gj.unsqueeze(1), group)
        # remove i from future matching
        scores[bi, i, :] = float("-inf")
        scores[bi, :, i] = float("-inf")
        alive[bi, i] = False

    # compact group ids to [0, n_out)
    n_out = N - r
    out = torch.zeros(B, n_out, C, device=x.device, dtype=x.dtype)
    out_s = torch.zeros(B, n_out, 1, device=x.device, dtype=x.dtype)
    for b in range(B):
        uniq, inv = torch.unique(group[b], return_inverse=True)
        ob, sb = _wavg_scatter(x[b:b+1], size[b:b+1], inv.unsqueeze(0), uniq.shape[0])
        out[b:b+1, :uniq.shape[0]] = ob
        out_s[b:b+1, :uniq.shape[0]] = sb
    return out, out_s


def bipartite(x, size, metric, attn, r, class_token=True):
    merge, _ = bipartite_soft_matching(metric, r, class_token=class_token)
    return merge_wavg(merge, x, size)


ALGOS = {
    "prune_random": (prune_random, "prune"),
    "prune_attn": (prune_attn, "prune"),
    "kmeans2": (kmeans2, "merge"),
    "kmeans5": (kmeans5, "merge"),
    "greedy": (greedy, "merge"),
    "bipartite": (bipartite, "merge"),
    # E6: kmeans strawman test
    "kmeans_random5": (kmeans_random5, "merge"),
    "kmeans_kpp_full": (kmeans_kpp_full, "merge"),
    "kmeans_random5_clsfix": (kmeans_random5_clsfix, "merge"),
}
