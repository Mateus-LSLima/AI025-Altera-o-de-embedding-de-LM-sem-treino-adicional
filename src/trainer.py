"""
Seção 7 de Metodo_A_Validacao_Escalavel.ipynb:
- Treino de vários alvos em paralelo:
  * cosine: MSE entre similaridades padronizadas (z-score) do receptor e do professor
  * recall_surrogate: pairwise logistic loss entre os top-K positivos e os demais negativos
  * combined: soma ponderada das duas perdas
- Execução de cada repetição experimental (run_trial)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    DEVICE,
    TOP_K,
    LR,
    RANK_MARGIN,
    LAMBDA_RANK,
    INIT_NOISE,
    STANDARDIZE_RELATIONS,
    CONTROL_OBJECTIVE,
    TRAIN_TEMPLATE_POOL,
    HELDOUT_TEMPLATES,
)
from .student_model import StudentContext
from .teacher_signatures import TeacherSignatureStore
from .metrics import eval_block


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)


def per_target_losses(
    V: torch.Tensor,
    rel: torch.Tensor,
    A_tr: torch.Tensor,
    mu: torch.Tensor,
    k: int = TOP_K,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = F.normalize(V - mu, dim=1) @ A_tr.T                    # [T, Atr]
    if STANDARDIZE_RELATIONS:
        l_cos = ((zscore(pred) - zscore(rel)) ** 2).mean(1)
    else:
        l_cos = ((pred - rel) ** 2).mean(1)
    kk = min(k, rel.shape[1] - 1)
    top = rel.topk(kk, dim=1).indices
    pos = torch.zeros_like(rel, dtype=torch.bool).scatter_(1, top, True)
    pair = pos.unsqueeze(2) & (~pos).unsqueeze(1)                 # [T, Atr, Atr]
    diff = pred.unsqueeze(2) - pred.unsqueeze(1)
    l_rank = (F.softplus(RANK_MARGIN - diff) * pair).sum((1, 2)) / pair.sum((1, 2)).clamp(min=1)
    return l_cos, l_rank


def fit_vectors(
    ctx: StudentContext,
    init: torch.Tensor,
    rel_tr: torch.Tensor,
    tr_idx: np.ndarray,
    objective: str,
) -> torch.Tensor:
    V = nn.Parameter(init.clone())
    opt = torch.optim.Adam([V], lr=LR)
    A_tr = ctx.A_C[torch.as_tensor(tr_idx, device=DEVICE)]
    rel = rel_tr.to(DEVICE).float()
    for _ in range(ctx.cfg.epochs):
        opt.zero_grad()
        lc, lr_ = per_target_losses(V, rel, A_tr, ctx.MU)
        loss = {"cosine": lc, "recall_surrogate": lr_, "combined": lc + LAMBDA_RANK * lr_}[objective].sum()
        loss.backward()
        opt.step()
    return V.detach()


@dataclass
class TrialArtifacts:
    VECTORS: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = field(default_factory=dict)
    TOPK: Dict[Tuple[str, int, str], np.ndarray] = field(default_factory=dict)


def run_trial(
    ctx: StudentContext,
    sig_store: TeacherSignatureStore,
    artifacts: TrialArtifacts,
    seed: int,
    frac: float,
    layer: int,
    objectives: Sequence[str],
    split_seed: Optional[int] = None,
    tag: str = "all_vary",
    control: bool = True,
    keep: bool = False,
) -> List[Dict[str, Any]]:
    """
    Uma repetição: sorteia split de âncoras e subconjunto de templates, treina e avalia em:
    (a) âncoras de treino, (b) âncoras NOVAS, (c) âncoras novas com relações do professor em contextos NOVOS.
    """
    split_seed = seed if split_seed is None else split_seed
    rs = np.random.RandomState(1000 + split_seed)
    perm = rs.permutation(ctx.N_A)
    n_tr = int(min(max(round(frac * ctx.N_A), TOP_K + 2), ctx.N_A - TOP_K))
    tr, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])

    rt = np.random.RandomState(2000 + seed)
    tmpl_idx = tuple(
        sorted(rt.choice(len(TRAIN_TEMPLATE_POOL), size=ctx.cfg.n_train_templates, replace=False).tolist())
    )

    rel_all = sig_store.teacher_rel(sig_store.SIG_TRAIN, "train", tmpl_idx, layer)                      # [T, A] (CPU)
    rel_held = sig_store.teacher_rel(sig_store.SIG_HELD, "held", tuple(range(len(HELDOUT_TEMPLATES))), layer)

    g = torch.Generator().manual_seed(3000 + seed)
    noise = torch.randn(ctx.E_INIT.shape, generator=g).to(DEVICE)
    rms = (ctx.E_INIT - ctx.MU).norm(dim=1, keepdim=True) / math.sqrt(ctx.HS)
    init = ctx.E_INIT + INIT_NOISE * rms * noise

    methods: Dict[str, torch.Tensor] = {"baseline": ctx.E_INIT}
    rel_tr = rel_all[:, tr]
    for obj in objectives:
        methods[obj] = fit_vectors(ctx, init, rel_tr, tr, obj)
    if control:   # controle negativo: embaralha, por alvo, quais âncoras recebem qual relação
        shuf = torch.rand(rel_tr.shape, generator=g).argsort(1)
        methods["control_shuffled"] = fit_vectors(ctx, init, torch.gather(rel_tr, 1, shuf), tr, CONTROL_OBJECTIVE)

    rows: List[Dict[str, Any]] = []
    for name, V in methods.items():
        meta = dict(seed=seed, mode=tag, frac=frac, layer=layer, method=name)
        sc_tr = ctx.student_scores(V, tr).cpu().numpy()
        sc_te = ctx.student_scores(V, te).cpu().numpy()
        rows += eval_block(sc_tr, rel_all[:, tr].numpy(), ctx.GOLD[:, tr], {**meta, "split": "train"}, ctx.TARGETS)
        rows += eval_block(sc_te, rel_all[:, te].numpy(), ctx.GOLD[:, te], {**meta, "split": "test"}, ctx.TARGETS)
        rows += eval_block(sc_te, rel_held[:, te].numpy(), ctx.GOLD[:, te], {**meta, "split": "test_newctx"}, ctx.TARGETS)
        if keep:
            kk = min(TOP_K, len(te))
            artifacts.TOPK[(tag, seed, name)] = te[np.argsort(-sc_te, axis=1, kind="stable")[:, :kk]]

    # teto: o próprio professor (em contextos de treino) como "pontuador"
    sc = rel_all[:, te].numpy()
    meta = dict(seed=seed, mode=tag, frac=frac, layer=layer, method="teacher_ceiling")
    rows += eval_block(sc, None, ctx.GOLD[:, te], {**meta, "split": "test"}, ctx.TARGETS, teacher_metrics=False)
    rows += eval_block(sc, rel_held[:, te].numpy(), ctx.GOLD[:, te], {**meta, "split": "test_newctx"}, ctx.TARGETS)

    if keep:
        artifacts.VECTORS[(tag, seed)] = {n: ctx.finalize(V).cpu() for n, V in methods.items()}
    return rows
