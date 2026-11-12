"""
Seção 5 de Metodo_A_Validacao_Escalavel.ipynb:
- Captura da assinatura de ativação de todas as camadas do MLP do professor via forward hooks
- Cálculo e cache das matrizes de similaridade relativa alvo × âncora (teacher_rel)
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, CENTER_SPACES, MAIN_LAYER, TRAIN_TEMPLATE_POOL, HELDOUT_TEMPLATES
from .student_model import StudentContext


def get_mlps(model: nn.Module) -> List[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [l.mlp for l in model.model.layers]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [b.mlp for b in model.transformer.h]
    raise RuntimeError("Estrutura desconhecida: adapte get_mlps() com model.named_modules().")


@torch.inference_mode()
def extract_signatures(
    ctx: StudentContext,
    teacher_mlps: Sequence[nn.Module],
    words: Sequence[str],
    templates: Sequence[str],
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Um único forward por lote captura a saída de cada MLP na posição do último subtoken da palavra.
    Retorna tensor [palavras, templates, camadas, hidden] em float32 na CPU.
    """
    jobs = []
    for wi, w in enumerate(words):
        for ti, tpl in enumerate(templates):
            pre, post = tpl.split("{word}")
            pos = len(ctx.teacher_tok(pre + w, add_special_tokens=False)["input_ids"]) - 1
            jobs.append((wi, ti, pre + w + post, pos))

    n_layers = len(teacher_mlps)
    H = ctx.teacher.config.hidden_size
    out = torch.empty(len(words), len(templates), n_layers, H, dtype=torch.float32)

    for s in range(0, len(jobs), batch_size):
        chunk = jobs[s:s + batch_size]
        enc = ctx.teacher_tok([j[2] for j in chunk], return_tensors="pt", padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        pos = torch.tensor([j[3] for j in chunk], device=DEVICE)
        ar = torch.arange(len(chunk), device=DEVICE)
        buf = [None] * n_layers

        def mk(i: int):
            def hook(mod, inp, o):
                o = o[0] if isinstance(o, tuple) else o
                buf[i] = o[ar, pos].float()
            return hook

        handles = [m.register_forward_hook(mk(i)) for i, m in enumerate(teacher_mlps)]
        try:
            ctx.teacher.base_model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        acts = torch.stack(buf, dim=1)                       # [B, L, H]
        if not torch.isfinite(acts).all():
            raise FloatingPointError(
                "Ativações não finitas no professor (overflow em fp16). "
                "Reduza MODEL_SCALE ou use uma GPU com bf16 (A100)."
            )
        acts = acts.cpu()
        for b, (wi, ti, _, _) in enumerate(chunk):
            out[wi, ti] = acts[b]
    return out


@dataclass
class TeacherSignatureStore:
    N_LAYERS: int
    LAYER_MAIN: int
    SIG_TRAIN: torch.Tensor
    SIG_HELD: torch.Tensor
    N_T: int
    rel_cache: Dict[Tuple[str, Tuple[int, ...], int], torch.Tensor] = field(default_factory=dict)

    def teacher_rel(self, sig: torch.Tensor, key: str, tmpl_idx: Sequence[int], layer: int) -> torch.Tensor:
        """Relações alvo×âncora [T, A] (cosseno entre assinaturas) para uma camada e um conjunto de templates."""
        ck = (key, tuple(tmpl_idx), layer)
        if ck not in self.rel_cache:
            x = sig[:, :, layer][:, list(tmpl_idx)].mean(1).to(DEVICE).float()   # [W, H]
            if CENTER_SPACES:
                x = x - x.mean(0, keepdim=True)
            x = F.normalize(x, dim=1)
            self.rel_cache[ck] = (x[:self.N_T] @ x[self.N_T:].T).cpu()
        return self.rel_cache[ck]


def build_teacher_signatures(ctx: StudentContext, batch_size: int = 32) -> TeacherSignatureStore:
    teacher_mlps = get_mlps(ctx.teacher)
    n_layers = len(teacher_mlps)
    print("Camadas MLP do professor:", n_layers)

    t0 = time.time()
    sig_train = extract_signatures(ctx, teacher_mlps, ctx.ALL_WORDS, TRAIN_TEMPLATE_POOL, batch_size=batch_size)
    sig_held = extract_signatures(ctx, teacher_mlps, ctx.ALL_WORDS, HELDOUT_TEMPLATES, batch_size=batch_size)
    print(
        f"Assinaturas: treino {tuple(sig_train.shape)} | contextos novos {tuple(sig_held.shape)} | "
        f"{time.time() - t0:.1f}s"
    )

    layer_main = MAIN_LAYER % n_layers
    print("Camada principal (Exp. 1-3):", layer_main)
    return TeacherSignatureStore(
        N_LAYERS=n_layers,
        LAYER_MAIN=layer_main,
        SIG_TRAIN=sig_train,
        SIG_HELD=sig_held,
        N_T=ctx.N_T,
    )
