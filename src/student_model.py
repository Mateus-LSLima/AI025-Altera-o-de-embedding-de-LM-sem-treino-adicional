"""
Seções 3 e 4 de Metodo_A_Validacao_Escalavel.ipynb:
- Carregamento do modelo professor (Qwen2.5) e receptor (GPT-2)
- Seleção de palavras-alvo (multi-token no receptor) e âncoras + matriz semântica GOLD
- Embeddings de âncoras, ponto inicial (piece_mean), centralização (MU), escala (C_REF)
- Expansão cirúrgica do vocabulário (set_new_rows) e preservação de norma (finalize)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import (
    DEVICE,
    DTYPE,
    ScaleConfig,
    TARGET_POOL,
    ANCHOR_POOL,
    TRAIN_TEMPLATE_POOL,
    HELDOUT_TEMPLATES,
    CENTER_SPACES,
)


@dataclass
class StudentContext:
    cfg: ScaleConfig
    teacher_tok: AutoTokenizer
    student_tok: AutoTokenizer
    teacher: AutoModelForCausalLM
    student: AutoModelForCausalLM
    PAD_ID: int
    TARGETS: List[str]
    ANCHORS: List[str]
    GLOSS: Dict[str, str]
    N_T: int
    N_A: int
    ALL_WORDS: List[str]
    GOLD: np.ndarray
    ORIG_VOCAB: int
    HS: int
    E_ANCH: torch.Tensor
    E_INIT: torch.Tensor
    MU: torch.Tensor
    A_C: torch.Tensor
    C_REF: torch.Tensor
    old_rows_snapshot: torch.Tensor
    NEW_IDS: List[int]
    tied: bool

    @torch.no_grad()
    def set_new_rows(self, V: torch.Tensor) -> None:
        """Grava os vetores [N_T, Hs] nas linhas dos tokens novos (e na lm_head, se não houver tying)."""
        emb = self.student.get_input_embeddings().weight
        emb[self.NEW_IDS[0]:self.NEW_IDS[-1] + 1].copy_(V.to(emb.device, emb.dtype))
        head = self.student.get_output_embeddings().weight
        if head.data_ptr() != emb.data_ptr():
            head[self.NEW_IDS[0]:self.NEW_IDS[-1] + 1].copy_(V.to(head.device, head.dtype))

    def finalize(self, V: torch.Tensor) -> torch.Tensor:
        """Mantém a direção aprendida e restaura a escala da inicialização (evita norma arbitrária)."""
        return self.MU + self.C_REF.unsqueeze(1) * F.normalize(V - self.MU, dim=1)

    def student_scores(self, V: torch.Tensor, anchor_idx: Optional[np.ndarray] = None) -> torch.Tensor:
        A = self.A_C if anchor_idx is None else self.A_C[torch.as_tensor(anchor_idx, device=DEVICE)]
        return F.normalize(V - self.MU, dim=1) @ A.T


@torch.no_grad()
def piece_mean(word: str, W: torch.Tensor, student_tok: AutoTokenizer) -> torch.Tensor:
    ids = student_tok(" " + word, add_special_tokens=False)["input_ids"]
    return W[ids].float().mean(0)


def load_models_and_prepare_student(cfg: ScaleConfig) -> StudentContext:
    """
    Executa as Seções 3 e 4 do notebook:
    carrega modelos, filtra alvos multi-token, monta GOLD e expande o vocabulário do receptor.
    """
    teacher_tok = AutoTokenizer.from_pretrained(cfg.teacher_id)
    student_tok = AutoTokenizer.from_pretrained(cfg.student_id)
    if teacher_tok.pad_token is None:
        teacher_tok.pad_token = teacher_tok.eos_token
    teacher_tok.padding_side = "right"
    PAD_ID = student_tok.eos_token_id

    load_kw = dict(torch_dtype=DTYPE, low_cpu_mem_usage=True)
    if DEVICE.type == "cuda":
        load_kw["device_map"] = {"": 0}
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_id, **load_kw).eval()
    student = AutoModelForCausalLM.from_pretrained(cfg.student_id, **load_kw).eval()
    for m in (teacher, student):
        for p in m.parameters():
            p.requires_grad_(False)

    # --- alvos: multi-token no receptor; âncoras: não repetem alvos ---
    valid_targets = [
        (w, d, g) for (w, d, g) in TARGET_POOL
        if len(student_tok(" " + w, add_special_tokens=False)["input_ids"]) > 1
    ]
    dropped = [w for (w, _, _) in TARGET_POOL if w not in {v[0] for v in valid_targets}]
    if dropped:
        print("Descartados (já são token único no receptor):", dropped)
    TARGET_ROWS = valid_targets[:cfg.n_targets]
    if len(TARGET_ROWS) < cfg.n_targets:
        print(f"AVISO: só {len(TARGET_ROWS)} alvos válidos (pedidos {cfg.n_targets}).")
    TARGETS = [r[0] for r in TARGET_ROWS]
    anchor_rows = [(w, d) for (w, d) in ANCHOR_POOL if w not in TARGETS][:cfg.n_anchors]
    ANCHORS = [r[0] for r in anchor_rows]
    GLOSS = {r[0]: r[2] for r in TARGET_ROWS}
    N_T, N_A = len(TARGETS), len(ANCHORS)
    ALL_WORDS = TARGETS + ANCHORS

    # gold independente de modelo: nº de domínios compartilhados
    GOLD = np.array([[len(td & ad) for (_, ad) in anchor_rows] for (_, td, _) in TARGET_ROWS], dtype=float)
    print(
        f"{N_T} alvos | {N_A} âncoras | {cfg.n_train_templates}/{len(TRAIN_TEMPLATE_POOL)} templates de treino por seed | "
        f"{len(HELDOUT_TEMPLATES)} contextos novos | {cfg.n_seeds} seeds"
    )
    print("Gold: fração de pares com relevância > 0 =", round(float((GOLD > 0).mean()), 3))

    # --- Seção 4: Receptor: embeddings de âncoras, ponto inicial e expansão do vocabulário ---
    W0 = student.get_input_embeddings().weight.detach()
    ORIG_VOCAB, HS = W0.shape
    E_ANCH = torch.stack([piece_mean(a, W0, student_tok) for a in ANCHORS]).to(DEVICE)   # [A, Hs]
    E_INIT = torch.stack([piece_mean(t, W0, student_tok) for t in TARGETS]).to(DEVICE)   # [T, Hs]
    MU = torch.cat([E_ANCH, E_INIT]).mean(0) if CENTER_SPACES else torch.zeros(HS, device=DEVICE)
    A_C = F.normalize(E_ANCH - MU, dim=1)                                    # âncoras no espaço centralizado
    C_REF = (E_INIT - MU).norm(dim=1)                                        # escala de referência por alvo

    old_rows_snapshot = W0[:ORIG_VOCAB].detach().cpu().clone()
    try:
        student.resize_token_embeddings(ORIG_VOCAB + N_T, mean_resizing=False)
    except TypeError:
        student.resize_token_embeddings(ORIG_VOCAB + N_T)
    for p in student.parameters():
        p.requires_grad_(False)
    NEW_IDS = list(range(ORIG_VOCAB, ORIG_VOCAB + N_T))
    tied = student.get_output_embeddings().weight.data_ptr() == student.get_input_embeddings().weight.data_ptr()
    print(f"Vocab: {ORIG_VOCAB} -> {ORIG_VOCAB + N_T} | hidden receptor: {HS} | weight tying: {tied}")

    return StudentContext(
        cfg=cfg,
        teacher_tok=teacher_tok,
        student_tok=student_tok,
        teacher=teacher,
        student=student,
        PAD_ID=PAD_ID,
        TARGETS=TARGETS,
        ANCHORS=ANCHORS,
        GLOSS=GLOSS,
        N_T=N_T,
        N_A=N_A,
        ALL_WORDS=ALL_WORDS,
        GOLD=GOLD,
        ORIG_VOCAB=ORIG_VOCAB,
        HS=HS,
        E_ANCH=E_ANCH,
        E_INIT=E_INIT,
        MU=MU,
        A_C=A_C,
        C_REF=C_REF,
        old_rows_snapshot=old_rows_snapshot,
        NEW_IDS=NEW_IDS,
        tied=tied,
    )
