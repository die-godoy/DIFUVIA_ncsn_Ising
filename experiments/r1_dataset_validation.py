"""
BLOCK 1: Dataset Validation

Loads the Monte Carlo Wolff dataset, computes thermodynamic observables,
compares against exact 2D Ising theory, and exports figures and a CSV table.

Usage
-----
python experiments/r1_dataset_validation.py \
    --data_dir  train_data \
    --L         64 \
    --fig_dir   figures \
    --table_dir tables
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from difuvia.physics import Tc, get_theory_curves
from difuvia.thermodynamics import load_all_temperatures, load_mc_ground_truth, compute_spin_correlation
from difuvia.viz import (
    UV_BLUE,
    plot_representative_lattices,
    plot_spin_correlation_comparison,
    plot_thermodynamic_observables,
)
import torch


def build_dataframe(mc_data: dict, temperatures: list) -> pd.DataFrame:
    """Convert the {T: obs_dict} ground truth into a tidy DataFrame."""
    rows = []
    for T in sorted(temperatures):
        if T not in mc_data:
            continue
        obs = mc_data[T]
        rows.append({
            "T":      T,
            "M_mean": obs["M_mean"],
            "M_std":  obs["M_std"],
            "E_mean": obs["E_mean"],
            "E_std":  obs["E_std"],
            "Cv":     obs["Cv"],
            "Chi":    obs["Chi"],
        })
    return pd.DataFrame(rows)


def export_results_table(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df_export = df.round(5)
    df_export.columns = ["T", "<|m|>", "std(m)", "<e>", "std(e)", "Cv", "Chi"]
    df_export.to_csv(filename, index=False)

    print("\n" + "=" * 68)
    print("TABLE 1: Thermodynamic Observables — Monte Carlo Ground Truth")
    print("=" * 68)
    print(f"{'T':>6} {'<|m|>':>10} {'std(m)':>10} {'<e>':>10} {'std(e)':>10} {'Cv':>10} {'Chi':>10}")
    print("-" * 68)
    for _, row in df_export.iterrows():
        print(
            f"{row['T']:>6.2f} {row['<|m|>']:>10.5f} {row['std(m)']:>10.5f} "
            f"{row['<e>']:>10.5f} {row['std(e)']:>10.5f} {row['Cv']:>10.5f} {row['Chi']:>10.5f}"
        )
    print("=" * 68)
    print(f"Table saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description="R1: Monte Carlo dataset validation")
    parser.add_argument("--data_dir",  default="train_data")
    parser.add_argument("--L",         type=int, default=64)
    parser.add_argument("--fig_dir",   default="figures")
    parser.add_argument("--table_dir", default="tables")
    args = parser.parse_args()

    TEMPERATURES = [round(1.77 + i * 0.1, 2) for i in range(11)]
    os.makedirs(args.fig_dir,   exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    print("=" * 60)
    print("BLOCK 1: DATASET VALIDATION")
    print("=" * 60)

    # 1. Load dataset (numpy arrays for representative lattice plot)
    print("\n[1/5] Loading Monte Carlo dataset...")
    dataset = load_all_temperatures(args.data_dir, TEMPERATURES, L=args.L)
    if not dataset:
        raise FileNotFoundError(f"No data files found in '{args.data_dir}'.")

    # 2. Compute thermodynamic observables (torch-based, with sample std)
    print("\n[2/5] Computing thermodynamic observables...")
    mc_data = load_mc_ground_truth(args.data_dir, TEMPERATURES, L=args.L)
    df = build_dataframe(mc_data, TEMPERATURES)
    for _, row in df.iterrows():
        print(
            f"  T={row['T']:.2f} | <|m|>={row['M_mean']:.4f} | "
            f"<e>={row['E_mean']:.4f} | Cv={row['Cv']:.4f} | χ={row['Chi']:.4f}"
        )

    # 3. Exact theory curves
    print("\n[3/5] Generating exact theoretical curves...")
    T_range = [min(TEMPERATURES) - 0.2, max(TEMPERATURES) + 0.2]
    theory = get_theory_curves(T_min=T_range[0], T_max=T_range[1])
    print(f"  Tc (exact) = {Tc:.6f}")

    # 4. Figures
    print("\n[4/5] Generating figures...")
    plot_representative_lattices(
        dataset,
        save_path=os.path.join(args.fig_dir, "Fig1_representative_lattices.pdf"),
    )
    plot_thermodynamic_observables(
        df,
        theory,
        save_path=os.path.join(args.fig_dir, "Fig2_thermodynamic_validation.pdf"),
    )

    # 5. Spin-spin correlation at Tc
    T_crit = min(dataset.keys(), key=lambda t: abs(t - Tc))
    print(f"\n  Computing G(r) at T={T_crit:.2f} (closest to Tc={Tc:.4f})...")
    configs_tc = torch.tensor(dataset[T_crit]).unsqueeze(1)
    r_vals, G_r = compute_spin_correlation(configs_tc)
    print(f"  G(r) computed over {len(r_vals)} radial bins.")
    plot_spin_correlation_comparison(
        {"Monte Carlo (Wolff)": {T_crit: configs_tc}},
        labels=["Monte Carlo (Wolff)"],
        colors=[UV_BLUE],
        markers=["o"],
        T_crit=T_crit,
        L=args.L,
        save_path=os.path.join(args.fig_dir, "Fig3_spin_correlation_mc.pdf"),
    )
    gr_path = os.path.join(args.table_dir, "Table_spin_correlation_mc.csv")
    pd.DataFrame({"r": r_vals, "G_r": G_r}).to_csv(gr_path, index=False)
    print(f"  G(r) table saved: {gr_path}")

    # 6. Export table
    print("\n[5/5] Exporting results table...")
    export_results_table(
        df,
        filename=os.path.join(args.table_dir, "Table1_validation_wolff.csv"),
    )

    print("\n" + "=" * 60)
    print("BLOCK 1 COMPLETE.")
    print(f"  Temperatures validated : {len(df)}")
    print(f"  Figures saved in       : {args.fig_dir}/")
    print(f"  Table saved in         : {args.table_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
