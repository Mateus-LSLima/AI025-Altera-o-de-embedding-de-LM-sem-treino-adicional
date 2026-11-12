"""
Seção 6 de Metodo_A_Validacao_Escalavel.ipynb:
- Métricas de fidelidade ao professor (Recall@K, nDCG@K, Spearman)
- Métricas de qualidade semântica independente contra a régua externa GOLD (P@K_gold, nDCG@K_gold, Spearman_gold)
- Bloco de avaliação por alvo (eval_block)
"""

from typing import Dict, List, Any, Optional, Sequence
import numpy as np
from scipy.stats import rankdata

from .config import TOP_K

M_TEACHER = ["Recall@K", "nDCG@K", "Spearman"]
M_GOLD = ["P@K_gold", "nDCG@K_gold", "Spearman_gold"]
ALL_METRICS = M_TEACHER + M_GOLD
METHOD_ORDER = ["baseline", "cosine", "recall_surrogate", "combined", "control_shuffled", "teacher_ceiling"]


def _dcg(gains: np.ndarray, order: np.ndarray, k: int) -> float:
    g = gains[order[:k]]
    return float((g / np.log2(np.arange(2, len(g) + 2))).sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else np.nan


def rank_metrics(score: np.ndarray, rel: np.ndarray, k: int) -> Dict[str, float]:
    order = np.argsort(-score, kind="stable")
    ideal = np.argsort(-rel, kind="stable")
    hit = len(set(order[:k].tolist()) & set(ideal[:k].tolist()))
    gains = rel - rel.min()
    idcg = _dcg(gains, ideal, k)
    return {
        "Recall@K": hit / k,
        "nDCG@K": _dcg(gains, order, k) / idcg if idcg > 0 else np.nan,
        "Spearman": _spearman(score, rel),
    }


def gold_metrics(score: np.ndarray, gold: np.ndarray, k: int) -> Dict[str, float]:
    order = np.argsort(-score, kind="stable")
    ideal = np.argsort(-gold, kind="stable")
    idcg = _dcg(gold, ideal, k)
    return {
        "P@K_gold": float((gold[order[:k]] > 0).mean()),
        "nDCG@K_gold": _dcg(gold, order, k) / idcg if idcg > 0 else np.nan,
        "Spearman_gold": _spearman(score, gold),
    }


def eval_block(
    scores: np.ndarray,
    rel: Optional[np.ndarray],
    gold: np.ndarray,
    meta: Dict[str, Any],
    targets: Sequence[str],
    teacher_metrics: bool = True,
) -> List[Dict[str, Any]]:
    rows = []
    k = min(TOP_K, scores.shape[1])
    for i in range(scores.shape[0]):
        r = {**meta, "target": targets[i]}
        if teacher_metrics and rel is not None:
            r.update(rank_metrics(scores[i], rel[i], k))
        r.update(gold_metrics(scores[i], gold[i], k))
        rows.append(r)
    return rows
