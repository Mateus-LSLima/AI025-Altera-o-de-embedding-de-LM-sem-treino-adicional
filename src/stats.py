"""
Seção 8 de Metodo_A_Validacao_Escalavel.ipynb:
- Estatística: intervalos de confiança por bootstrap agrupado em dois níveis (cluster_bootstrap: seeds × alvos)
- Sumarização (summarize), comparações pareadas (paired_delta, deltas_vs) e formatação de tabelas (fmt, ordered, show)
"""

from typing import List, Tuple, Sequence, Optional
import numpy as np
import pandas as pd

from .config import N_BOOT
from .metrics import METHOD_ORDER


def cluster_bootstrap(M: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> Tuple[float, float, float]:
    """
    Recebe uma matriz [seeds × alvos] e reamostra as duas dimensões (n_boot vezes),
    capturando simultaneamente a incerteza das seeds e da amostra de palavras-alvo.
    """
    M = np.asarray(M, float)
    S, T = M.shape
    rng = np.random.RandomState(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = np.nanmean(M[np.ix_(rng.randint(0, S, S), rng.randint(0, T, T))])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.nanmean(M)), float(lo), float(hi)


def _mat(sub: pd.DataFrame, m: str) -> pd.DataFrame:
    return sub.pivot_table(index="seed", columns="target", values=m, aggfunc="mean")


def summarize(df: pd.DataFrame, keys: List[str], metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for gk, sub in df.groupby(keys, sort=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        for m in metrics:
            M = _mat(sub, m)
            if M.size == 0 or M.isna().all().all():
                continue
            mean, lo, hi = cluster_bootstrap(M.values)
            sm = np.nanmean(M.values, axis=1)
            rows.append({
                **dict(zip(keys, gk)),
                "metric": m,
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "std_seeds": float(sm.std(ddof=1)) if len(sm) > 1 else np.nan,
                "n_seeds": M.shape[0],
            })
    return pd.DataFrame(rows)


def paired_delta(
    df: pd.DataFrame,
    keys: List[str],
    col: str,
    a: str,
    b: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Δ = (col==a) − (col==b), pareado por seed×alvo, dentro de cada grupo de `keys`."""
    rows = []
    for gk, sub in df.groupby(keys, sort=False):
        gk = gk if isinstance(gk, tuple) else (gk,)
        for m in metrics:
            Ma, Mb = _mat(sub[sub[col] == a], m), _mat(sub[sub[col] == b], m)
            if Ma.empty or Mb.empty:
                continue
            ri, ci = Ma.index.intersection(Mb.index), Ma.columns.intersection(Mb.columns)
            D = Ma.loc[ri, ci].values - Mb.loc[ri, ci].values
            if np.isnan(D).all():
                continue
            mean, lo, hi = cluster_bootstrap(D)
            rows.append({
                **dict(zip(keys, gk)),
                "metric": m,
                "comparação": f"{a} − {b}",
                "Δ": mean,
                "lo": lo,
                "hi": hi,
                "significativo": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


def fmt(summ: pd.DataFrame, index: List[str], columns: str = "metric") -> pd.DataFrame:
    s = summ.copy()
    s["txt"] = s.apply(lambda r: f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]", axis=1)
    return s.pivot_table(index=index, columns=columns, values="txt", aggfunc="first")


def ordered(df: pd.DataFrame, col: str = "method") -> pd.DataFrame:
    present = [m for m in METHOD_ORDER if m in set(df[col])]
    return df.set_index(col).loc[present].reset_index()


def show(summ: pd.DataFrame, index: List[str], metrics: List[str]) -> pd.DataFrame:
    t = fmt(summ[summ.metric.isin(metrics)], index)[metrics]
    return t.sort_index(key=lambda idx: idx.map(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x))


def deltas_vs(
    df: pd.DataFrame,
    keys: List[str],
    metrics: Sequence[str],
    base: str = "baseline",
    methods: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    methods = methods or [m for m in METHOD_ORDER if m in set(df.method) and m != base]
    out = []
    for m in methods:
        d = paired_delta(df, keys, "method", m, base, metrics)
        d.insert(0, "método", m)
        out.append(d.drop(columns=["comparação"]))
    return pd.concat(out, ignore_index=True)
