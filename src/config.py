"""
Seções 0, 1 e 2 de Metodo_A_Validacao_Escalavel.ipynb:
- Ambiente e detecção de GPU/dtype
- Escala de modelagem composta estilo EfficientNet (MODEL_SCALE / φ)
- Configuração experimental: pools de alvos, âncoras, templates e hiperparâmetros
"""

import warnings
from dataclasses import dataclass, asdict
from typing import Dict, Any, Union

import pandas as pd
import torch

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ==============================================================================
# 0. Ambiente
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    major, _ = torch.cuda.get_device_capability(0)
    # T4 (sm_75) não tem bf16 nativo; is_bf16_supported() pode devolver True por emulação lenta.
    DTYPE = torch.bfloat16 if major >= 8 else torch.float16
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    DTYPE, GPU_NAME, GPU_GB = torch.float32, "CPU", 0.0


def print_environment() -> None:
    print(f"Device: {GPU_NAME} | VRAM: {GPU_GB:.1f} GB | dtype: {DTYPE} | torch {torch.__version__}")


# ==============================================================================
# 1. Variável de escala de modelagem (φ)
# ==============================================================================
MODEL_SCALE: Union[int, str] = 5  # inteiro 0..5 ou "auto" (maior φ que cabe na GPU, com 20% de folga)
OVERRIDES: Dict[str, Any] = {}    # ex.: {"n_seeds": 10, "n_targets": 12}

# (professor, receptor, params professor [B], params receptor [B])
LADDER = [
    ("Qwen/Qwen2.5-0.5B", "openai-community/gpt2",        0.494, 0.124),  # φ0  CPU / teste rápido
    ("Qwen/Qwen2.5-0.5B", "openai-community/gpt2-medium", 0.494, 0.355),  # φ1
    ("Qwen/Qwen2.5-1.5B", "openai-community/gpt2-medium", 1.54,  0.355),  # φ2
    ("Qwen/Qwen2.5-3B",   "openai-community/gpt2-large",  3.09,  0.774),  # φ3  teto da T4
    ("Qwen/Qwen2.5-7B",   "openai-community/gpt2-xl",     7.62,  1.558),  # φ4  A100 40 GB
    ("Qwen/Qwen2.5-14B",  "openai-community/gpt2-xl",     14.77, 1.558),  # φ5  A100 80 GB (40 GB apertado)
]
BASE = dict(n_targets=8, n_anchors=860, n_train_templates=4, n_seeds=3, epochs=120)
GROWTH = dict(n_targets=1.35, n_anchors=1.01, n_train_templates=1.12, n_seeds=1.25)
EPOCH_STEP = 20
OVERHEAD_GB = 2.0          # contexto CUDA + ativações + logits dos testes funcionais
VRAM_HEADROOM = 0.80       # só usa até 80% da VRAM total


@dataclass
class ScaleConfig:
    phi: int
    teacher_id: str
    student_id: str
    n_targets: int
    n_anchors: int
    n_train_templates: int
    n_seeds: int
    epochs: int
    est_vram_gb: float


def est_vram_gb(phi: int) -> float:
    _, _, pt, ps = LADDER[phi]
    bytes_per_param = 2 if DTYPE != torch.float32 else 4
    return bytes_per_param * (pt + ps) + OVERHEAD_GB


def make_config(phi: int, overrides: Dict[str, Any] = None) -> ScaleConfig:
    t_id, s_id, _, _ = LADDER[phi]
    v = {k: int(round(BASE[k] * GROWTH[k] ** phi)) for k in GROWTH}
    v["epochs"] = BASE["epochs"] + EPOCH_STEP * phi
    cfg = ScaleConfig(phi, t_id, s_id, est_vram_gb=round(est_vram_gb(phi), 1), **v)
    active_overrides = OVERRIDES if overrides is None else overrides
    for k, val in active_overrides.items():
        setattr(cfg, k, val)
    return cfg


def pick_phi(model_scale: Union[int, str] = MODEL_SCALE) -> int:
    if isinstance(model_scale, int):
        return max(0, min(model_scale, len(LADDER) - 1))
    if DEVICE.type != "cuda":
        return 0
    best = 0
    for p in range(len(LADDER)):
        if est_vram_gb(p) <= VRAM_HEADROOM * GPU_GB:
            best = p
    return best


def build_ladder_dataframe(selected_phi: int = None) -> pd.DataFrame:
    ladder_df = pd.DataFrame([asdict(make_config(p)) for p in range(len(LADDER))])
    for name, gb in [("T4", 15.8), ("A100-40", 42.5), ("A100-80", 85.0)]:
        ladder_df[f"cabe_{name}"] = ladder_df.est_vram_gb <= VRAM_HEADROOM * gb
    if selected_phi is not None:
        ladder_df["escolhida"] = ["*" if p == selected_phi else "" for p in range(len(LADDER))]
    return ladder_df


# ==============================================================================
# 2. Configuração experimental
# ==============================================================================
# (palavra, domínios, gloss) — alvos; os 8 primeiros são os do notebook original
TARGET_POOL = [
    ("photosynthesis", {"bio", "chem"}, "is the process by which plants use sunlight to turn water and carbon dioxide into sugar and oxygen"),
    ("bioluminescence", {"bio"}, "is the emission of light by living organisms such as fireflies and deep-sea fish"),
    ("thermodynamics", {"phys"}, "is the branch of physics that studies heat, energy, and work"),
    ("electroencephalography", {"neuro", "med"}, "is a method of recording the electrical activity of the brain using electrodes on the scalp"),
    ("quasar", {"astro"}, "is an extremely bright and distant galactic nucleus powered by a supermassive black hole"),
    ("mitochondrion", {"bio"}, "is the organelle in a cell that produces most of its chemical energy"),
    ("photosensitive", {"phys", "bio"}, "describes a material or organ that reacts to light"),
    ("geothermal", {"earth", "phys"}, "describes energy or heat that comes from inside the Earth"),
    ("spectroscopy", {"phys", "chem", "astro"}, "is the study of how matter interacts with light across different wavelengths"),
    ("chromatography", {"chem"}, "is a laboratory technique for separating the components of a mixture"),
    ("neurotransmitter", {"neuro", "bio", "med"}, "is a chemical messenger that carries signals between nerve cells"),
    ("photovoltaic", {"phys"}, "describes devices that convert sunlight directly into electricity"),
    ("seismology", {"earth", "phys"}, "is the scientific study of earthquakes and the waves they send through the Earth"),
    ("supernova", {"astro", "phys"}, "is the explosion of a massive star at the end of its life"),
    ("exoplanet", {"astro"}, "is a planet that orbits a star outside our solar system"),
    ("chlorophyll", {"bio", "chem"}, "is the green pigment in plants that absorbs light for photosynthesis"),
    ("endothermic", {"chem", "phys"}, "describes a reaction that absorbs heat from its surroundings"),
    ("electrolysis", {"chem", "phys"}, "is a process that uses electric current to drive a chemical reaction such as splitting water"),
    ("neuroplasticity", {"neuro", "bio"}, "is the ability of the brain to reorganize itself by forming new connections"),
    ("tectonics", {"earth"}, "is the study of the large-scale movement of the plates of the Earth's crust"),
    ("magnetosphere", {"astro", "earth", "phys"}, "is the region around a planet dominated by its magnetic field"),
    ("nucleotide", {"bio", "chem"}, "is the basic building block of DNA and RNA"),
    ("hemoglobin", {"bio", "med", "chem"}, "is the protein in red blood cells that carries oxygen"),
    ("superconductivity", {"phys"}, "is a state in which a material conducts electricity with zero resistance"),
    ("interferometer", {"phys", "astro"}, "is an instrument that combines waves to measure tiny differences in distance"),
    ("volcanology", {"earth"}, "is the scientific study of volcanoes and their eruptions"),
    ("ribosome", {"bio"}, "is the cellular machine that builds proteins from instructions in RNA"),
    ("pulsar", {"astro", "phys"}, "is a rapidly rotating neutron star that emits regular pulses of radiation"),
    ("electromagnetism", {"phys"}, "is the force between electrically charged particles and magnetic fields"),
    ("immunology", {"med", "bio"}, "is the study of how the body defends itself against disease"),
    ("hydrothermal", {"earth", "chem"}, "describes processes involving hot water circulating beneath the Earth's surface"),
    ("astrophysics", {"astro", "phys"}, "is the branch of astronomy that applies physics to stars, galaxies, and the universe"),
    ("oscilloscope", {"phys"}, "is an instrument that displays electrical signals as waves on a screen"),
    ("phytoplankton", {"bio", "earth"}, "is the community of tiny drifting plants that feed ocean life and produce oxygen"),
    ("radioactivity", {"phys", "chem"}, "is the spontaneous emission of radiation from unstable atomic nuclei"),
    ("cardiovascular", {"med", "bio"}, "describes the heart and blood vessels that circulate blood through the body"),
    # reservas: entram se algum acima já for token único no receptor
    ("catalysis", {"chem"}, "is the speeding up of a chemical reaction by a substance that is not consumed"),
    ("biodiversity", {"bio", "earth"}, "is the variety of living species found in a habitat or on the whole planet"),
    ("antibacterial", {"med", "bio", "chem"}, "describes a substance that kills bacteria or stops them from growing"),
    ("seismograph", {"earth", "phys"}, "is an instrument that records the vibrations of the ground during earthquakes"),
    ("radiotelescope", {"astro", "phys"}, "is an instrument that collects radio waves from distant objects in space"),
]

# (palavra, domínios) — âncoras; as 36 primeiras são as do notebook original
ANCHOR_POOL = [
    ("plant", {"bio"}), ("light", {"phys"}), ("energy", {"phys"}), ("chemical", {"chem"}), ("biology", {"bio"}), ("organism", {"bio"}),
    ("cell", {"bio"}), ("heat", {"phys"}), ("temperature", {"phys", "earth"}), ("physics", {"phys"}), ("electricity", {"phys"}),
    ("brain", {"neuro", "med", "bio"}), ("signal", {"neuro", "phys"}), ("measurement", {"phys"}), ("wave", {"phys"}),
    ("astronomy", {"astro"}), ("star", {"astro"}), ("galaxy", {"astro"}), ("space", {"astro"}), ("radiation", {"phys", "astro"}),
    ("machine", {"phys"}), ("water", {"chem", "earth", "bio"}), ("earth", {"earth", "astro"}), ("rock", {"earth"}), ("volcano", {"earth"}),
    ("reaction", {"chem"}), ("molecule", {"chem", "bio"}), ("oxygen", {"chem", "bio"}), ("growth", {"bio"}), ("process", set()),
    ("instrument", {"phys"}), ("medicine", {"med"}), ("animal", {"bio"}), ("environment", {"earth", "bio"}), ("power", {"phys"}), ("matter", {"phys", "chem"}),
    # extras para φ maiores
    ("sun", {"astro", "phys"}), ("planet", {"astro", "earth"}), ("telescope", {"astro", "phys"}), ("atom", {"chem", "phys"}),
    ("protein", {"bio", "chem"}), ("gene", {"bio"}), ("enzyme", {"bio", "chem"}), ("bacteria", {"bio", "med"}), ("virus", {"bio", "med"}),
    ("disease", {"med"}), ("blood", {"bio", "med"}), ("heart", {"med", "bio"}), ("nerve", {"neuro", "bio"}), ("neuron", {"neuro", "bio"}),
    ("memory", {"neuro"}), ("hospital", {"med"}), ("drug", {"med", "chem"}), ("acid", {"chem"}), ("metal", {"chem", "earth"}),
    ("crystal", {"chem", "earth"}), ("magnet", {"phys"}), ("current", {"phys"}), ("voltage", {"phys"}), ("laser", {"phys"}),
    ("particle", {"phys"}), ("gravity", {"phys", "astro"}), ("orbit", {"astro", "phys"}), ("comet", {"astro"}), ("moon", {"astro", "earth"}),
    ("earthquake", {"earth"}), ("mountain", {"earth"}), ("ocean", {"earth", "bio"}), ("climate", {"earth"}), ("atmosphere", {"earth", "astro"}),
    ("soil", {"earth", "bio"}), ("mineral", {"earth", "chem"}), ("fossil", {"earth", "bio"}), ("tree", {"bio"}), ("bird", {"bio"}),
    ("fish", {"bio"}), ("forest", {"bio", "earth"}), ("sugar", {"chem", "bio"}), ("carbon", {"chem", "earth"}), ("hydrogen", {"chem", "astro"}),
]

# Contextos de TREINO da assinatura (palavra no fim; posição extraída = último subtoken da palavra)
TRAIN_TEMPLATE_POOL = [
    "The scientific concept is {word}",
    "Researchers study {word}",
    "An example of this process is {word}",
    "The relevant scientific term is {word}",
    "A textbook chapter about {word}",
    "The professor explained {word}",
    "This article is about {word}",
    "Scientists often discuss {word}",
]

# Contextos NOVOS (nunca usados no treino; palavra no meio da frase)
HELDOUT_TEMPLATES = [
    "Yesterday the students read a long paper on {word} in the library.",
    "Few people outside the field understand {word} very well.",
    "It is hard to explain {word} to a child.",
    "The conference included a talk on {word} and its applications.",
    "Many questions remain open about {word} today.",
]

# Moldes do teste funcional: prefixo + palavra + " " + gloss + "."
FUNC_PREFIXES = [
    "In science class we learned that",
    "According to the encyclopedia,",
    "My teacher once explained that",
    "A short definition:",
]
QA_PROMPTS = ["Question: What {g}? Answer:", "Q: What {g}?\nA:"]

for _t in TRAIN_TEMPLATE_POOL + HELDOUT_TEMPLATES:
    assert _t.count("{word}") == 1, _t

# --- hiperparâmetros de treino/avaliação ---
TOP_K = 10
LR = 0.03
RANK_MARGIN = 0.10
LAMBDA_RANK = 1.0
INIT_NOISE = 0.10            # ruído relativo na inicialização (é o que dá sentido às seeds)
CENTER_SPACES = True         # centraliza assinaturas do professor e embeddings do receptor
STANDARDIZE_RELATIONS = True # z-score das similaridades na loss de cosseno
OBJECTIVES = ["cosine", "recall_surrogate", "combined"]
CONTROL_OBJECTIVE = "combined"   # controle negativo: treina com relações embaralhadas
FRACS = [0.3, 0.5, 0.7, 0.9]     # fração de âncoras usada no treino
MAIN_FRAC = 0.7
MAIN_LAYER = -1              # camada do MLP do professor usada nos Exp. 1-3 (-1 = última)
LAYER_OBJECTIVE = "combined"
N_BOOT = 2000
