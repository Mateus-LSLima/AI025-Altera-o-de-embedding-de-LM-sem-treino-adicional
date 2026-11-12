#!/usr/bin/env python3
"""
Ponto de entrada CLI para executar o pipeline completo de Metodo_A_Validacao_Escalavel.
Reproduz em sequência as Seções 0 a 13 do notebook original:
  - Escala automática ou manual (--scale auto | 1..5)
  - Extração de assinaturas do MLP do professor (Qwen2.5)
  - Otimização de representações relativas no receptor (GPT-2)
  - Experimentos 1, 2, 3a, 3b e 4
  - Resumo executivo, checagem de integridade, exportação em ./results/ e geração qualitativa.

Exemplo de uso:
  python run_metodo_a.py --scale auto
  python run_metodo_a.py --scale 1 --no-generation
"""

import argparse
from dataclasses import asdict

from src import (
    DEVICE,
    DTYPE,
    GPU_NAME,
    GPU_GB,
    make_config,
    pick_phi,
    build_ladder_dataframe,
    load_models_and_prepare_student,
    build_teacher_signatures,
    run_all_trials,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3a,
    run_experiment_3b,
    run_experiment_4,
    run_summary_and_export,
    run_qualitative_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa a bateria de validação escalável do Método A (Transferência de Vocabulário por Ativações)."
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="auto",
        help="Escala experimental φ ('auto' para detecção por VRAM ou inteiro de 1 a 5). Padrão: auto.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Diretório de saída para os arquivos CSV e JSON. Padrão: results.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="assets/figures",
        help="Diretório para salvar as figuras PNG dos Experimentos 1, 2 e 4.",
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="Pula a seção final de geração qualitativa.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_scale = "auto" if args.scale.lower() == "auto" else int(args.scale)

    print(f"Dispositivo: {DEVICE} ({GPU_NAME}, {GPU_GB:.1f} GB VRAM) | dtype dos modelos: {DTYPE}")
    phi = pick_phi(model_scale)
    cfg = make_config(phi)

    print("\nEscada de configurações (* = escolhida):")
    print(build_ladder_dataframe(phi).to_string(index=False))
    print("\nConfiguração ativa:", asdict(cfg))

    # Seções 3 e 4: Carregar modelos e preparar receptor
    ctx = load_models_and_prepare_student(cfg)

    # Seção 5: Extrair assinaturas de todas as camadas do MLP do professor
    sig_store = build_teacher_signatures(ctx)

    # Seção 9: Rodar todas as repetições (Exp. 1, 2 e 3a)
    df, artifacts, seeds = run_all_trials(ctx, sig_store)

    # Experimento 1: Separar âncoras de treino e teste
    exp1_res = run_experiment_1(
        ctx,
        sig_store,
        df,
        save_fig_path=f"{args.figures_dir}/exp1_curva_aprendizado.png",
    )

    # Experimento 2: Estabilidade entre seeds
    exp2_res = run_experiment_2(
        ctx,
        sig_store,
        df,
        artifacts,
        save_fig_path=f"{args.figures_dir}/exp2_estabilidade_seeds.png",
    )

    # Experimento 3a e 3b: Novos contextos e uso funcional no receptor
    exp3a_res = run_experiment_3a(exp1_res["D1m"])
    exp3b_res = run_experiment_3b(ctx, artifacts, seeds)

    # Experimento 4: Varredura das camadas do MLP do professor
    exp4_res = run_experiment_4(
        ctx,
        sig_store,
        artifacts,
        seeds,
        save_fig_path=f"{args.figures_dir}/exp4_varredura_camadas_mlp.png",
    )

    # Resumo executivo, checagens de integridade e exportação
    run_summary_and_export(
        ctx,
        sig_store,
        seeds,
        df,
        exp1_res,
        exp2_res,
        exp3a_res,
        exp3b_res,
        exp4_res,
        results_dir=args.results_dir,
    )

    # Geração qualitativa opcional
    if not args.no_generation:
        run_qualitative_generation(ctx, artifacts)


if __name__ == "__main__":
    main()
