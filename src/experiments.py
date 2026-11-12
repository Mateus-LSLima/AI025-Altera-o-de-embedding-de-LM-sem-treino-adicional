"""
Seções 9 a 13 de Metodo_A_Validacao_Escalavel.ipynb:
- Seção 9: Rodar todas as repetições (run_all_trials)
- Experimento 1: Separar âncoras de treino e teste + Curva de aprendizado (run_experiment_1)
- Experimento 2: Estabilidade entre seeds e Jaccard médio dos top-K (run_experiment_2)
- Experimento 3a: Generalização para novos contextos de template (run_experiment_3a)
- Experimento 3b: Teste funcional no receptor — NLL da definição e Identificação QA (run_experiment_3b)
- Experimento 4: Varredura das camadas do MLP do professor (run_experiment_4)
- Resumo executivo, checagens de integridade e exportação para ./results/ (run_summary_and_export)
- Teste qualitativo de geração com os vetores aprendidos (run_qualitative_generation)
"""

import os
import json
import time
import itertools
from dataclasses import asdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .config import (
    DEVICE,
    TOP_K,
    FRACS,
    MAIN_FRAC,
    OBJECTIVES,
    LAYER_OBJECTIVE,
    FUNC_PREFIXES,
    QA_PROMPTS,
)
from .student_model import StudentContext
from .teacher_signatures import TeacherSignatureStore
from .metrics import ALL_METRICS, METHOD_ORDER
from .trainer import TrialArtifacts, run_trial
from .stats import cluster_bootstrap, _mat, summarize, paired_delta, fmt, show, deltas_vs


def _display_table(df: pd.DataFrame) -> None:
    """Usa IPython.display.display se disponível; caso contrário imprime via print()."""
    try:
        from IPython.display import display
        display(df)
    except Exception:
        print(df.to_string())


# ---------------------------------------------------------------------------
# Seção 9: Rodar todas as repetições (Exp. 1, 2 e 3a compartilham as execuções)
# ---------------------------------------------------------------------------
def run_all_trials(
    ctx: StudentContext,
    sig_store: TeacherSignatureStore,
    artifacts: Optional[TrialArtifacts] = None,
) -> Tuple[pd.DataFrame, TrialArtifacts, List[int]]:
    """
    Para cada seed variamos três fontes de aleatoriedade:
    (1) o split de âncoras, (2) o subconjunto de templates de treino e (3) o ruído na inicialização.
    Há ainda um modo `split_fixed` e um controle negativo (`control_shuffled`).
    """
    assert MAIN_FRAC in FRACS
    if artifacts is None:
        artifacts = TrialArtifacts()

    seeds = list(range(ctx.cfg.n_seeds))
    all_rows = []
    t0 = time.time()
    for seed in seeds:
        for frac in FRACS:
            all_rows += run_trial(
                ctx,
                sig_store,
                artifacts,
                seed,
                frac,
                sig_store.LAYER_MAIN,
                OBJECTIVES,
                tag="all_vary",
                keep=(frac == MAIN_FRAC),
            )
        all_rows += run_trial(
            ctx,
            sig_store,
            artifacts,
            seed,
            MAIN_FRAC,
            sig_store.LAYER_MAIN,
            OBJECTIVES,
            split_seed=0,
            tag="split_fixed",
            keep=True,
        )
        print(f"seed {seed} concluída ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(all_rows)
    print(df.shape)
    return df, artifacts, seeds


# ---------------------------------------------------------------------------
# Experimento 1 — Separar âncoras de treino e teste
# ---------------------------------------------------------------------------
def run_experiment_1(
    ctx: StudentContext,
    sig_store: TeacherSignatureStore,
    df: pd.DataFrame,
    save_fig_path: Optional[str] = "assets/figures/exp1_curva_aprendizado.png",
    show_plot: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Pergunta: o vetor aprendido ordena corretamente âncoras que nunca viu?
    Calcula métricas em train vs test, gap de generalização (treino − teste),
    ganho no teste sobre a baseline e curva de aprendizado por fração de âncoras.
    """
    D1 = df[(df["mode"] == "all_vary") & (df.layer == sig_store.LAYER_MAIN)]
    D1m = D1[D1.frac == MAIN_FRAC]
    n_te = ctx.N_A - int(round(MAIN_FRAC * ctx.N_A))
    print(f"Fração de treino = {MAIN_FRAC} → ~{ctx.N_A - n_te} âncoras de treino / ~{n_te} de teste.")
    print(f"Nível de acaso de Recall@{TOP_K} no teste ≈ {TOP_K}/{n_te} = {TOP_K / n_te:.2f}; de Spearman ≈ 0.\n")

    s1 = summarize(D1m[D1m.split.isin(["train", "test"])], ["split", "method"], ALL_METRICS)
    print("Média [IC95%] por split e método")
    _display_table(show(s1, ["split", "method"], ["Recall@K", "nDCG@K", "Spearman", "nDCG@K_gold", "Spearman_gold"]))

    sig_cols = ["Recall@K", "Spearman", "nDCG@K_gold", "Spearman_gold"]
    print("\nGap de generalização, Δ = treino − teste (pareado); positivo = o método decora")
    gap = paired_delta(
        D1m[D1m.split.isin(["train", "test"]) & D1m.method.isin(["cosine", "recall_surrogate", "combined", "control_shuffled"])],
        ["method"],
        "split",
        "train",
        "test",
        sig_cols,
    )
    _display_table(gap.round(3))

    print("\nGanho no TESTE sobre a baseline: Δ = método − baseline (Δ > 0 e IC fora de 0 = transferência real)")
    dt = deltas_vs(D1m[D1m.split == "test"], ["split"], sig_cols)
    _display_table(dt.round(3))

    # curva de aprendizado: teste vs fração de âncoras usadas no treino
    lc = summarize(D1[D1.split == "test"], ["frac", "method"], ["Recall@K", "Spearman_gold"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["Recall@K", "Spearman_gold"]):
        for m in ["baseline", "cosine", "recall_surrogate", "combined", "control_shuffled"]:
            s = lc[(lc.method == m) & (lc.metric == metric)].sort_values("frac")
            ax.plot(s.frac, s["mean"], marker="o", label=m)
            ax.fill_between(s.frac, s.lo, s.hi, alpha=0.15)
        ax.set_xlabel("fração de âncoras no treino")
        ax.set_ylabel(metric + " (teste)")
        ax.set_title(f"Curva de aprendizado: {metric}")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        fig.savefig(save_fig_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {"D1": D1, "D1m": D1m, "s1": s1, "gap": gap, "dt": dt, "lc": lc}


# ---------------------------------------------------------------------------
# Experimento 2 — Estabilidade entre seeds
# ---------------------------------------------------------------------------
def run_experiment_2(
    ctx: StudentContext,
    sig_store: TeacherSignatureStore,
    df: pd.DataFrame,
    artifacts: TrialArtifacts,
    save_fig_path: Optional[str] = "assets/figures/exp2_estabilidade_seeds.png",
    show_plot: bool = False,
) -> Dict[str, Any]:
    """
    Pergunta: o resultado depende da seed?
    Compara all_vary vs split_fixed, calcula taxa de vitória sobre a baseline e
    Jaccard médio entre os conjuntos de top-K vizinhos de pares de seeds.
    """
    D2 = df[(df.layer == sig_store.LAYER_MAIN) & (df.frac == MAIN_FRAC) & (df.split == "test")]
    per_seed = D2.groupby(["mode", "method", "seed"])[["Recall@K", "Spearman", "Spearman_gold"]].mean().reset_index()
    agg = per_seed.groupby(["mode", "method"])[["Recall@K", "Spearman", "Spearman_gold"]].agg(["mean", "std", "min", "max"]).round(3)
    print("Distribuição das médias POR SEED (teste)")
    _display_table(agg.sort_index(key=lambda idx: idx.map(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else x)))

    # taxa de vitória sobre a baseline entre seeds
    piv = per_seed[per_seed["mode"] == "all_vary"].pivot(index="seed", columns="method", values="Recall@K")
    wins = {m: float((piv[m] > piv["baseline"]).mean()) for m in piv.columns if m not in ("baseline", "teacher_ceiling")}
    print("\nFração de seeds em que o método supera a baseline (Recall@K no teste, all_vary):", {k: round(v, 2) for k, v in wins.items()})

    def mean_pairwise_jaccard(tag: str, method: str) -> float:
        seeds = sorted(s for (t, s, m) in artifacts.TOPK if t == tag and m == method)
        vals = []
        for ti in range(ctx.N_T):
            sets = [set(artifacts.TOPK[(tag, s, method)][ti].tolist()) for s in seeds]
            vals += [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
        return float(np.mean(vals)) if vals else np.nan

    jac = pd.DataFrame({
        "Jaccard médio dos top-K entre seeds (split fixo)": {
            m: mean_pairwise_jaccard("split_fixed", m)
            for m in ["baseline", "cosine", "recall_surrogate", "combined", "control_shuffled"]
        }
    }).round(3)
    _display_table(jac)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ["all_vary", "split_fixed"]):
        sub = per_seed[per_seed["mode"] == mode]
        ms = [m for m in METHOD_ORDER if m in set(sub.method) and m != "teacher_ceiling"]
        ax.boxplot([sub[sub.method == m]["Recall@K"].values for m in ms], showmeans=True)
        ax.set_xticks(range(1, len(ms) + 1))
        ax.set_xticklabels(ms, rotation=20)
        ax.set_title(f"Recall@K no teste por seed — {mode}")
    plt.tight_layout()
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        fig.savefig(save_fig_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return {"D2": D2, "per_seed": per_seed, "agg": agg, "wins": wins, "jac": jac}


# ---------------------------------------------------------------------------
# Experimento 3a e 3b — Novos contextos e uso funcional no receptor
# ---------------------------------------------------------------------------
def run_experiment_3a(D1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    3a. Relações do professor em contextos novos (palavra no meio da frase)
    nas âncoras de teste (duplamente fora da amostra: âncoras novas × frases novas).
    """
    D3 = D1m[D1m.split.isin(["test", "test_newctx"])]
    s3 = summarize(D3, ["split", "method"], ["Recall@K", "nDCG@K", "Spearman", "nDCG@K_gold", "Spearman_gold"])
    print("3a — relações do professor em contextos de TREINO (test) vs NOVOS (test_newctx), âncoras de teste")
    _display_table(show(s3, ["split", "method"], ["Recall@K", "nDCG@K", "Spearman", "nDCG@K_gold", "Spearman_gold"]))
    print("\nGanho sobre a baseline em contextos NOVOS")
    d3_newctx = deltas_vs(D3[D3.split == "test_newctx"], ["split"], ["Recall@K", "Spearman", "Spearman_gold"])
    _display_table(d3_newctx.round(3))
    return {"D3": D3, "s3": s3, "d3_newctx": d3_newctx}


def run_experiment_3b(
    ctx: StudentContext,
    artifacts: TrialArtifacts,
    seeds: List[int],
) -> Dict[str, Any]:
    """
    3b. Uso funcional no receptor (GPT-2):
    - NLL da definição (gloss_nll): menor é melhor.
    - Identificação QA (identification): Acc@1 e MRR entre os N_T novos tokens.
    """
    ITEMS = []
    for t, w in enumerate(ctx.TARGETS):
        suf = ctx.student_tok(" " + ctx.GLOSS[w] + ".", add_special_tokens=False)["input_ids"]
        w_orig = ctx.student_tok(" " + w, add_special_tokens=False)["input_ids"]
        for p in FUNC_PREFIXES:
            ITEMS.append(dict(t=t, pre=ctx.student_tok(p, add_special_tokens=False)["input_ids"], w=w_orig, suf=suf))
    QA_ITEMS = [
        (t, ctx.student_tok(q.format(g=ctx.GLOSS[w]), add_special_tokens=False)["input_ids"])
        for t, w in enumerate(ctx.TARGETS)
        for q in QA_PROMPTS
    ]

    def _pad(seqs: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), ctx.PAD_ID, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        return ids.to(DEVICE), att.to(DEVICE)

    @torch.inference_mode()
    def gloss_nll(variant: str, bs: int = 32) -> np.ndarray:
        """NLL médio por token da definição, dado prefixo + palavra. variant: 'new' (token novo) | 'orig' (subtokens)."""
        seqs, starts = [], []
        for it in ITEMS:
            wi = [ctx.NEW_IDS[it["t"]]] if variant == "new" else it["w"]
            seqs.append(it["pre"] + wi + it["suf"])
            starts.append(len(it["pre"]) + len(wi))
        out = np.zeros(len(seqs))
        for i in range(0, len(seqs), bs):
            ids, att = _pad(seqs[i:i + bs])
            logp = F.log_softmax(ctx.student(input_ids=ids, attention_mask=att).logits.float()[:, :-1], -1)
            tok_lp = logp.gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            for b in range(ids.shape[0]):
                out[i + b] = -tok_lp[b, starts[i + b] - 1: len(seqs[i + b]) - 1].mean().item()
        return np.bincount([it["t"] for it in ITEMS], weights=out, minlength=ctx.N_T) / len(FUNC_PREFIXES)

    @torch.inference_mode()
    def identification(bs: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Rank do token novo correto entre os N_T tokens novos, pelos logits do próximo token."""
        ranks = np.zeros(len(QA_ITEMS))
        for i in range(0, len(QA_ITEMS), bs):
            chunk = QA_ITEMS[i:i + bs]
            ids, att = _pad([c[1] for c in chunk])
            logits = ctx.student(input_ids=ids, attention_mask=att).logits.float()
            last = att.sum(1) - 1
            for b, (t, _) in enumerate(chunk):
                l = logits[b, last[b], ctx.NEW_IDS]
                ranks[i + b] = int((l > l[t]).sum().item())
        tgt = np.array([c[0] for c in QA_ITEMS])
        acc = np.bincount(tgt, weights=(ranks == 0).astype(float), minlength=ctx.N_T) / len(QA_PROMPTS)
        mrr = np.bincount(tgt, weights=1.0 / (ranks + 1), minlength=ctx.N_T) / len(QA_PROMPTS)
        return acc, mrr

    nll_sub = gloss_nll("orig")
    FUNC_ROWS = []
    for seed in seeds:
        vecs = dict(artifacts.VECTORS[("all_vary", seed)])
        g = torch.Generator().manual_seed(4000 + seed)
        vecs["random"] = ctx.finalize(ctx.MU + torch.randn(ctx.N_T, ctx.HS, generator=g).to(DEVICE)).cpu()   # controle: direção aleatória
        for name, V in vecs.items():
            ctx.set_new_rows(V.to(DEVICE))
            nll = gloss_nll("new")
            acc, mrr = identification()
            for t in range(ctx.N_T):
                FUNC_ROWS.append(dict(seed=seed, method=name, target=ctx.TARGETS[t], NLL=nll[t], **{"Acc@1": acc[t], "MRR": mrr[t]}))
        for t in range(ctx.N_T):
            FUNC_ROWS.append(dict(seed=seed, method="subwords", target=ctx.TARGETS[t], NLL=nll_sub[t], **{"Acc@1": np.nan, "MRR": np.nan}))
    DFF = pd.DataFrame(FUNC_ROWS)

    F_ORDER = ["subwords", "baseline", "random", "cosine", "recall_surrogate", "combined", "control_shuffled"]
    sf = summarize(DFF, ["method"], ["NLL", "Acc@1", "MRR"])
    t_table = fmt(sf, ["method"])[["NLL", "Acc@1", "MRR"]].reindex([m for m in F_ORDER if m in set(DFF.method)])
    print("3b — Uso funcional (NLL: menor é melhor; Acc@1/MRR: maior é melhor)")
    _display_table(t_table)
    chance_mrr = float(np.mean(1.0 / np.arange(1, ctx.N_T + 1)))
    print(f"Acaso: Acc@1 = {1 / ctx.N_T:.3f} | MRR = {chance_mrr:.3f}")

    # Δ NLL pareado: método − random  e  método − baseline  (negativo = melhor)
    DFF["grupo"] = "todos"
    dd = []
    for ref in ["random", "baseline"]:
        for m in ["cosine", "recall_surrogate", "combined", "control_shuffled", "subwords"]:
            d = paired_delta(DFF, ["grupo"], "method", m, ref, ["NLL"])
            d.insert(0, "referência", ref)
            d.insert(0, "método", m)
            dd.append(d.drop(columns=["grupo"]))
    print("\nΔ NLL pareado = método − referência (negativo = método melhor que a referência)")
    dd_df = pd.concat(dd, ignore_index=True)
    _display_table(dd_df.round(4))

    return {"DFF": DFF, "sf": sf, "dd": dd, "dd_df": dd_df}


# ---------------------------------------------------------------------------
# Experimento 4 — Varredura das camadas do MLP do professor
# ---------------------------------------------------------------------------
def run_experiment_4(
    ctx: StudentContext,
    sig_store: TeacherSignatureStore,
    artifacts: TrialArtifacts,
    seeds: List[int],
    save_fig_path: Optional[str] = "assets/figures/exp4_varredura_camadas_mlp.png",
    show_plot: bool = False,
) -> Dict[str, Any]:
    """
    Pergunta: qual camada do MLP do professor produz a melhor representação para transferir?
    Critério externo ao professor: Spearman_gold nas âncoras de teste.
    """
    LAYER_PRIMARY = "Spearman_gold"
    N_LAYERS = sig_store.N_LAYERS
    LAYERS = list(range(N_LAYERS)) if N_LAYERS <= 28 else sorted(set(list(range(0, N_LAYERS, 2)) + [N_LAYERS - 1]))
    rows, t0 = [], time.time()
    for layer in LAYERS:
        for seed in seeds:
            rows += run_trial(ctx, sig_store, artifacts, seed, MAIN_FRAC, layer, [LAYER_OBJECTIVE], tag="layers", control=False)
        if layer % 4 == 0 or layer == LAYERS[-1]:
            print(f"camada {layer} ({time.time() - t0:.0f}s)")
    DFL = pd.DataFrame(rows)

    DL = DFL[DFL.split == "test"]
    sl = summarize(DL, ["layer", "method"], [LAYER_PRIMARY, "nDCG@K_gold", "P@K_gold"])
    sl_nc = summarize(DFL[DFL.split == "test_newctx"], ["layer", "method"], ["Recall@K", "Spearman"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, metric, splt in [(axes[0], LAYER_PRIMARY, sl), (axes[1], "nDCG@K_gold", sl)]:
        for m, c in [("combined", "C0"), ("teacher_ceiling", "C2"), ("baseline", "C3")]:
            s = splt[(splt.method == m) & (splt.metric == metric)].sort_values("layer")
            ax.plot(s.layer, s["mean"], marker="o", ms=3, color=c, label=m)
            ax.fill_between(s.layer, s.lo, s.hi, color=c, alpha=0.15)
        ax.set_xlabel("camada do MLP do professor (0 = primeira)")
        ax.set_ylabel(metric + " (âncoras de teste)")
        ax.set_title(metric)
    axes[0].legend()
    plt.tight_layout()
    if save_fig_path:
        os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
        fig.savefig(save_fig_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    tab = sl[sl.method.isin(["combined", "teacher_ceiling", "baseline"])].pivot_table(
        index="layer", columns=["metric", "method"], values="mean"
    )
    ceil_ = tab[(LAYER_PRIMARY, "teacher_ceiling")]
    eff = (tab[(LAYER_PRIMARY, "combined")] / ceil_.where(ceil_ > 0.05)).rename("eficiência (combinado/teto)")
    out = fmt(sl[(sl.method == "combined")], ["layer"])[[LAYER_PRIMARY, "nDCG@K_gold", "P@K_gold"]].join(eff.round(3))
    out["Recall@K (ctx novos, fidelidade)"] = fmt(sl_nc[(sl_nc.method == "combined")], ["layer"])["Recall@K"]
    _display_table(out)

    def layer_mat(layer: int) -> pd.DataFrame:
        return _mat(DL[(DL.layer == layer) & (DL.method == "combined")], LAYER_PRIMARY)

    means = {l: np.nanmean(layer_mat(l).values) for l in LAYERS}
    BEST_LAYER = max(means, key=means.get)
    Mb = layer_mat(BEST_LAYER)
    indist = []
    for l in LAYERS:
        if l == BEST_LAYER:
            continue
        Ml = layer_mat(l)
        ri, ci = Mb.index.intersection(Ml.index), Mb.columns.intersection(Ml.columns)
        mean, lo, hi = cluster_bootstrap(Mb.loc[ri, ci].values - Ml.loc[ri, ci].values)
        if lo <= 0 <= hi:
            indist.append(l)
    base_ref = float(np.nanmean(_mat(DL[(DL.layer == BEST_LAYER) & (DL.method == "baseline")], LAYER_PRIMARY).values))
    print(f"Melhor camada por {LAYER_PRIMARY}: {BEST_LAYER} (média {means[BEST_LAYER]:.3f}; baseline {base_ref:.3f})")
    print(f"Camadas estatisticamente indistinguíveis da melhor (IC95% do Δ contém 0): {indist}")
    print(f"\n>>> Para repetir os Exp. 1–3 com a camada vencedora, defina MAIN_LAYER = {BEST_LAYER} e rode de novo.")

    return {
        "DFL": DFL,
        "DL": DL,
        "sl": sl,
        "sl_nc": sl_nc,
        "out": out,
        "BEST_LAYER": BEST_LAYER,
        "indist": indist,
    }


# ---------------------------------------------------------------------------
# Resumo executivo, checagens de integridade e exportação
# ---------------------------------------------------------------------------
def run_summary_and_export(
    ctx: StudentContext,
    sig_store: TeacherSignatureStore,
    seeds: List[int],
    df: pd.DataFrame,
    exp1_res: Dict[str, Any],
    exp2_res: Dict[str, Any],
    exp3a_res: Dict[str, Any],
    exp3b_res: Dict[str, Any],
    exp4_res: Dict[str, Any],
    results_dir: str = "results",
) -> None:
    """Imprime o resumo executivo dos 4 experimentos, valida a integridade dos pesos e exporta CSVs/JSON."""
    def line(label: str, d: pd.DataFrame) -> str:
        if d is None or len(d) == 0:
            return f"  {label}: n/d"
        r = d.iloc[0]
        col = "Δ"
        flag = "SIGNIFICATIVO" if r["significativo"] else "não significativo"
        return f"  {label}: Δ = {r[col]:+.3f}  IC95% [{r['lo']:+.3f}, {r['hi']:+.3f}]  → {flag}"

    dt = exp1_res["dt"]
    gap = exp1_res["gap"]
    s1 = exp1_res["s1"]
    per_seed = exp2_res["per_seed"]
    wins = exp2_res["wins"]
    jac = exp2_res["jac"]
    D3 = exp3a_res["D3"]
    DFF = exp3b_res["DFF"]
    dd = exp3b_res["dd"]
    DFL = exp4_res["DFL"]
    sl = exp4_res["sl"]
    BEST_LAYER = exp4_res["BEST_LAYER"]
    indist = exp4_res["indist"]

    print("=" * 78)
    print(f"RESUMO — φ={ctx.cfg.phi} | {ctx.cfg.teacher_id} → {ctx.cfg.student_id}")
    print(f"{ctx.N_T} alvos, {ctx.N_A} âncoras, {len(seeds)} seeds, camada principal {sig_store.LAYER_MAIN}/{sig_store.N_LAYERS - 1}, K={TOP_K}")
    print("=" * 78)

    print("\n[1] Generalização (âncoras novas, combined vs baseline)")
    print(line("Recall@K   ", dt[(dt["método"] == "combined") & (dt.metric == "Recall@K")]))
    print(line("Spearman_gold", dt[(dt["método"] == "combined") & (dt.metric == "Spearman_gold")]))
    print(line("Gap treino−teste Recall@K (combined)", gap[(gap.method == "combined") & (gap.metric == "Recall@K")]))
    print(line("Controle embaralhado vs baseline, Recall@K", dt[(dt["método"] == "control_shuffled") & (dt.metric == "Recall@K")]))

    print("\n[2] Estabilidade entre seeds (Recall@K no teste, combined)")
    r = per_seed[(per_seed["mode"] == "all_vary") & (per_seed.method == "combined")]["Recall@K"]
    r2 = per_seed[(per_seed["mode"] == "split_fixed") & (per_seed.method == "combined")]["Recall@K"]
    print(f"  desvio entre seeds: all_vary = {r.std():.3f} | split_fixed = {r2.std():.3f}")
    print(f"  vitórias sobre a baseline: {wins.get('combined', float('nan')):.0%} das seeds | Jaccard split fixo = {jac.iloc[:, 0].get('combined', float('nan')):.2f}")

    print("\n[3] Novos contextos")
    dnc = deltas_vs(D3[D3.split == "test_newctx"], ["split"], ["Recall@K"])
    print(line("3a Recall@K em contextos novos (combined vs baseline)", dnc[dnc["método"] == "combined"]))
    fn = DFF.pivot_table(index="method", values=["NLL", "Acc@1"])
    dn = pd.concat(dd, ignore_index=True)
    print(line("3b NLL da definição (combined − random; negativo = melhor)", dn[(dn["método"] == "combined") & (dn["referência"] == "random")]))
    print(f"  3b identificação Acc@1 combined = {fn.loc['combined', 'Acc@1']:.3f} (acaso {1 / ctx.N_T:.3f})")

    print(f"\n[4] Camadas: melhor = {BEST_LAYER}; indistinguíveis: {indist}")

    # --- integridade ---
    cur = ctx.student.get_input_embeddings().weight[:ctx.ORIG_VOCAB].detach().cpu()
    change = (cur - ctx.old_rows_snapshot).abs().max().item()
    print(f"\nIntegridade: maior alteração nas linhas antigas do embedding = {change}")
    assert change == 0.0, "Linhas antigas foram alteradas!"
    assert sum(p.requires_grad for p in ctx.student.parameters()) == 0 and sum(p.requires_grad for p in ctx.teacher.parameters()) == 0
    print("Parâmetros treináveis nos modelos: 0 (somente os vetores novos foram otimizados, fora do grafo do modelo).")

    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, "exp123_trials.csv"), index=False)
    DFL.to_csv(os.path.join(results_dir, "exp4_layers_trials.csv"), index=False)
    DFF.to_csv(os.path.join(results_dir, "exp3b_functional.csv"), index=False)
    s1.to_csv(os.path.join(results_dir, "exp1_summary.csv"), index=False)
    sl.to_csv(os.path.join(results_dir, "exp4_summary.csv"), index=False)
    with open(os.path.join(results_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                **asdict(ctx.cfg),
                "layer_main": sig_store.LAYER_MAIN,
                "best_layer": int(BEST_LAYER),
                "top_k": TOP_K,
                "targets": ctx.TARGETS,
                "anchors": ctx.ANCHORS,
            },
            f,
            indent=2,
        )
    print(f"Salvo em ./{results_dir}/")


# ---------------------------------------------------------------------------
# Seção Final — Teste qualitativo de geração com os vetores aprendidos
# ---------------------------------------------------------------------------
GENERATION_PROMPTS = {
    "photosynthesis": "Photosynthesis is the biological process in which",
    "bioluminescence": "Bioluminescence is the production of light by",
    "thermodynamics": "Thermodynamics studies the relationships among heat, work, temperature, and",
    "electroencephalography": "Electroencephalography is a method used to record",
}


def run_qualitative_generation(
    ctx: StudentContext,
    artifacts: TrialArtifacts,
) -> None:
    """
    Executa as duas células de geração qualitativa do notebook original:
    1) `gen`: insere explicitamente o ID do novo token (`NEW_IDS[t_idx]`) no meio do prompt.
    2) `generate`: tokeniza a string do prompt diretamente (dividindo a palavra nos subtokens originais).
    """
    ctx.set_new_rows(artifacts.VECTORS[("all_vary", 0)]["combined"].to(DEVICE))

    @torch.inference_mode()
    def gen(prompt_before: str, t_idx: int, prompt_after: str, n: int = 30) -> str:
        ids = (
            ctx.student_tok(prompt_before, add_special_tokens=False)["input_ids"]
            + [ctx.NEW_IDS[t_idx]]
            + ctx.student_tok(prompt_after, add_special_tokens=False)["input_ids"]
        )
        x = torch.tensor([ids], device=DEVICE)
        out = ctx.student.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            max_new_tokens=n,
            do_sample=False,
            pad_token_id=ctx.PAD_ID,
        )[0].tolist()
        new_map = {i: " " + w for i, w in zip(ctx.NEW_IDS, ctx.TARGETS)}
        parts, buf = [], []
        for i in out:
            if i in new_map:
                parts.append(ctx.student_tok.decode(buf))
                buf = []
                parts.append(new_map[i])
            else:
                buf.append(i)
        parts.append(ctx.student_tok.decode(buf))
        return "".join(parts)

    print("\n--- Amostra de geração com o token novo (NEW_IDS, combined, seed 0) ---")
    for t in range(min(4, ctx.N_T)):
        print("-", gen("In science,", t, " refers to"), "\n")

    @torch.inference_mode()
    def generate(prompt: str, max_new_tokens: int = 60) -> str:
        inputs = ctx.student_tok(prompt, return_tensors="pt").to(DEVICE)
        out = ctx.student.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=ctx.student_tok.eos_token_id,
            use_cache=True,
        )
        return ctx.student_tok.decode(out[0], skip_special_tokens=True)

    print("\n--- Geração qualitativa por prompt (tokenização padrão) ---")
    for w in ctx.TARGETS:
        prompt = GENERATION_PROMPTS.get(w, f"In science, {w} is a concept involving")
        print("\n###", w)
        print(generate(prompt, max_new_tokens=55))
