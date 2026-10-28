"""
BLOCK 4: NCSN vs DDPM Comparative Study

Loads pre-generated samples for NCSNv2-PIPC and DDPM (and Monte Carlo ground
truth), computes thermodynamic observables with bootstrap standard errors, prints
TABLE 4, and generates:

  - Fig4_thermodynamic_comparison.pdf   — 4-panel thermodynamics vs exact theory
  - Fig4.1_thermodynamic_comparison_bootstrap.pdf — same, with bootstrap SE on all four observables
  - Fig5_lattice_comparison.pdf         — visual lattice grid across models
  - Fig6_correlation_comparison.pdf     — log-log G(r) at Tc
  - Fig7_pareto_IPS.pdf                 — Pareto frontier (time vs IPS)
  - Fig8_pareto_W1.pdf                  — Pareto frontier (time vs W1)

Usage
-----
python experiments/r4_comparative.py \
    --ncsn_dir   gen_data/NCSN \
    --ddpm_dir   gen_data/DDPM \
    --data_dir   train_data \
    --config     ncsnv2/configs/ising.yml \
    --checkpoint networks/ckpt_epoch_300.pth \
    --fig_dir    figures \
    --table_dir  tables \
    --n_samples  1000 \
    --n_bootstrap 0

Notes
-----
- If --ncsn_dir is provided, samples are loaded from .pt or .npy files.
- If --checkpoint is provided instead (and --ncsn_dir is absent), the script
  generates NCSN samples on-the-fly before comparison.
- DDPM samples must be pre-generated externally and placed in --ddpm_dir.
"""

import os
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import torch

from difuvia.physics import get_theory_curves
from difuvia.thermodynamics import (
    bootstrap_errors,
    compute_observables,
    compute_spin_correlation,
    load_mc_ground_truth,
    load_mc_correlation_at_tc,
    load_all_temperatures,
)
from difuvia.analysis import (
    compute_nrmse,
    compute_ips,
    compute_wasserstein,
    print_comparison_table,
)
from difuvia.viz import (
    plot_thermodynamic_comparison,
    plot_spin_correlation_comparison,
    plot_lattice_comparison,
    plot_pareto,
)

TEMPERATURES = [round(1.77 + i * 0.1, 2) for i in range(11)]
IMAGE_SIZE   = 64


# ──────────────────────────────────────────────────────────────
# Sample loaders
# ──────────────────────────────────────────────────────────────

def _load_pt_samples(directory: str, temperatures: list) -> dict:
    """Load per-temperature .pt files from directory. Returns {T: Tensor(N,1,L,L)}."""
    dataset = {}
    for T in temperatures:
        candidates = [
            os.path.join(directory, f"T{T:.2f}.pt"),
            os.path.join(directory, f"samples_T{T:.2f}.pt"),
        ]
        for fpath in candidates:
            if os.path.exists(fpath):
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    data = data.get("samples", data.get("configs", next(iter(data.values()))))
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(data)
                if data.dim() == 3:
                    data = data.unsqueeze(1)
                dataset[T] = data.float()
                break
    return dataset


def _load_npy_samples(directory: str, temperatures: list) -> dict:
    """Load per-temperature .npy files from directory. Returns {T: Tensor(N,1,L,L)}."""
    dataset = {}
    for T in temperatures:
        candidates = [
            os.path.join(directory, f"T{T:.2f}.npy"),
            os.path.join(directory, f"samples_T{T:.2f}.npy"),
        ]
        for fpath in candidates:
            if os.path.exists(fpath):
                arr = np.load(fpath)
                t   = torch.from_numpy(arr).float()
                if t.dim() == 3:
                    t = t.unsqueeze(1)
                dataset[T] = t
                break
    return dataset


def load_samples(directory: str, temperatures: list) -> dict:
    """Try .pt first, then .npy. Returns {T: Tensor(N,1,L,L)}."""
    dataset = _load_pt_samples(directory, temperatures)
    if not dataset:
        dataset = _load_npy_samples(directory, temperatures)
    if not dataset:
        raise FileNotFoundError(
            f"No samples found in '{directory}'. Expected files named "
            f"T{{T:.2f}}.pt or T{{T:.2f}}.npy for each temperature."
        )
    found = sorted(dataset.keys())
    print(f"  Loaded {len(found)} temperatures from {directory}: {found}")
    return dataset


def generate_ncsn_samples(checkpoint: str, config_path: str, n_samples: int,
                           device: torch.device) -> dict:
    """Generate NCSN samples for all temperatures on-the-fly."""
    from difuvia.model_utils import load_model
    from difuvia.physics import SoftIsingEnergy
    from difuvia.sampling import pc_physics_guided_sampler

    score_model, sigmas, config = load_model(checkpoint, config_path, device)
    energy_fn = SoftIsingEnergy(J=1.0, double_well_strength=0.6)
    dataset   = {}

    print(f"  Generating {n_samples} NCSN samples per temperature...")
    for T in TEMPERATURES:
        configs = pc_physics_guided_sampler(
            score_model=score_model,
            energy_fn=energy_fn,
            sigmas=sigmas,
            target_temp=T,
            image_size=IMAGE_SIZE,
            batch_size=min(n_samples, 50),
            num_steps=200,
            corrector_steps=3,
            guidance_scale=0.3,
            snr=0.16,
            device=device,
        )
        n_needed = n_samples - len(configs)
        while n_needed > 0:
            extra = pc_physics_guided_sampler(
                score_model=score_model,
                energy_fn=energy_fn,
                sigmas=sigmas,
                target_temp=T,
                image_size=IMAGE_SIZE,
                batch_size=min(n_needed, 50),
                num_steps=200,
                corrector_steps=3,
                guidance_scale=0.3,
                snr=0.16,
                device=device,
            )
            configs = torch.cat([configs, extra], dim=0)
            n_needed = n_samples - len(configs)
        dataset[T] = configs[:n_samples]
        print(f"    T={T:.2f}  {dataset[T].shape}")

    return dataset


# ──────────────────────────────────────────────────────────────
# Observable computation across temperatures
# ──────────────────────────────────────────────────────────────

def compute_all_observables(dataset: dict, temperatures: list,
                             n_bootstrap: int = 0) -> dict:
    """
    Compute thermodynamic observables for all available temperatures.
    Returns {T: obs_dict}.
    """
    obs = {}
    for T in sorted(temperatures):
        if T not in dataset:
            continue
        obs[T] = compute_observables(dataset[T], T, IMAGE_SIZE, n_bootstrap=n_bootstrap)
    return obs


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NCSNv2-PIPC vs DDPM comparative study")
    parser.add_argument("--ncsn_dir",    default="gen_data/NCSN",
                        help="Directory with pre-generated NCSN .pt/.npy files (T{T:.2f}.pt)")
    parser.add_argument("--ddpm_dir",    default="gen_data/DDPM",
                        help="Directory with pre-generated DDPM .pt/.npy files")
    parser.add_argument("--data_dir",    default="train_data")
    parser.add_argument("--config",      default="ncsnv2/configs/ising.yml")
    parser.add_argument("--checkpoint",
                        default="networks/ckpt_epoch_300.pth",
                        help="Checkpoint path (used only if --ncsn_dir is not provided)")
    parser.add_argument("--n_samples",   type=int, default=1000)
    parser.add_argument("--n_bootstrap", type=int, default=0)
    parser.add_argument("--n_bootstrap_se", type=int, default=1000,
                        help="Bootstrap resamples for the Fig4.1 error bars")
    parser.add_argument("--lattice_temps", type=float, nargs="+",
                        default=[1.77, 1.97, 2.07],
                        help="Temperatures shown in the Fig5 lattice grid")
    parser.add_argument("--fig_dir",     default="figures")
    parser.add_argument("--table_dir",   default="tables")
    parser.add_argument("--experiment_name", default="E4",
                        help="Label used in thermodynamic comparison plot legend")
    args = parser.parse_args()

    os.makedirs(args.fig_dir,   exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    print("=" * 60)
    print("BLOCK 4: COMPARATIVE STUDY — NCSNv2 vs DDPM")
    print("=" * 60)

    # 1. MC ground truth
    print("\n[1/7] Loading Monte Carlo ground truth...")
    mc_dataset = load_all_temperatures(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)
    mc_gt      = load_mc_ground_truth(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)
    T_crit, mc_corr_tc = load_mc_correlation_at_tc(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)
    if T_crit:
        print(f"  MC G(r) at T={T_crit:.2f}")

    # 2. NCSN samples
    print("\n[2/7] Preparing NCSNv2-PIPC samples...")
    if args.ncsn_dir and os.path.isdir(args.ncsn_dir):
        ncsn_dataset = load_samples(args.ncsn_dir, TEMPERATURES)
    else:
        print("  --ncsn_dir not provided — generating NCSN samples on-the-fly...")
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
        ncsn_dataset = generate_ncsn_samples(args.checkpoint, args.config,
                                              args.n_samples, device)

    # 3. DDPM samples
    print("\n[3/7] Preparing DDPM samples...")
    if args.ddpm_dir and os.path.isdir(args.ddpm_dir):
        ddpm_dataset = load_samples(args.ddpm_dir, TEMPERATURES)
    else:
        print("  [Warning] --ddpm_dir not provided or does not exist. "
              "Using MC ground truth as DDPM placeholder (for testing only).")
        ddpm_dataset = {T: torch.from_numpy(mc_dataset[T]).unsqueeze(1).float()
                        for T in TEMPERATURES if T in mc_dataset}

    # 4. Compute observables
    print(f"\n[4/7] Computing observables (n_bootstrap={args.n_bootstrap})...")
    print("  NCSNv2-PIPC...")
    obs_ncsn = compute_all_observables(ncsn_dataset, TEMPERATURES, n_bootstrap=args.n_bootstrap)
    print("  DDPM...")
    obs_ddpm = compute_all_observables(ddpm_dataset, TEMPERATURES, n_bootstrap=args.n_bootstrap)

    # 5. Quantitative metrics
    print("\n[5/7] Computing quantitative metrics...")
    nrmse_ncsn = compute_nrmse(obs_ncsn, mc_gt, TEMPERATURES)
    nrmse_ddpm = compute_nrmse(obs_ddpm, mc_gt, TEMPERATURES)
    ips_ncsn   = compute_ips(nrmse_ncsn)
    ips_ddpm   = compute_ips(nrmse_ddpm)
    w1_ncsn    = compute_wasserstein(obs_ncsn, mc_gt, TEMPERATURES)
    w1_ddpm    = compute_wasserstein(obs_ddpm, mc_gt, TEMPERATURES)

    print(f"\n  NCSNv2-PIPC — IPS={ips_ncsn:.5f}, W1={w1_ncsn:.5f}")
    print(f"  DDPM        — IPS={ips_ddpm:.5f}, W1={w1_ddpm:.5f}")

    # Print + save TABLE 4
    print_comparison_table(mc_gt, obs_ncsn, obs_ddpm, TEMPERATURES, save_csv=True)

    # 6. Figures
    print("\n[6/7] Generating figures...")
    theory = get_theory_curves(T_min=min(TEMPERATURES) - 0.2, T_max=max(TEMPERATURES) + 0.2)

    # Fig 4: thermodynamic comparison
    plot_thermodynamic_comparison(
        mc_gt, obs_ncsn, obs_ddpm, theory, TEMPERATURES,
        experiment_name=args.experiment_name,
        save_path=os.path.join(args.fig_dir, "Fig4_thermodynamic_comparison.pdf"),
    )

    # Fig 4.1: same comparison, bootstrap SE on all four observables
    obs_bs = [
        {T: {**obs[T], **bootstrap_errors(obs[T], T, IMAGE_SIZE, args.n_bootstrap_se)}
         for T in obs}
        for obs in (mc_gt, obs_ncsn, obs_ddpm)
    ]
    plot_thermodynamic_comparison(
        *obs_bs, theory, TEMPERATURES,
        experiment_name=args.experiment_name,
        save_path=os.path.join(args.fig_dir, "Fig4.1_thermodynamic_comparison_bootstrap.pdf"),
    )

    # Fig 5: lattice grid
    temperatures_show = args.lattice_temps
    mc_tensor = {T: torch.from_numpy(mc_dataset[T]).unsqueeze(1).float()
                 for T in temperatures_show if T in mc_dataset}
    datasets_vis = {
        "Monte Carlo\n(Wolff)": mc_tensor,
        f"NCSNv2-PIPC\n({args.experiment_name})": ncsn_dataset,
        "DDPM": ddpm_dataset,
    }
    plot_lattice_comparison(
        {k: {T: v[T] for T in temperatures_show if T in v} for k, v in datasets_vis.items()},
        temperatures_show=temperatures_show,
        n_samples=3,
        L=IMAGE_SIZE,
        save_path=os.path.join(args.fig_dir, "Fig5_lattice_comparison.pdf"),
    )

    # Fig 6: G(r) comparison
    tc_datasets = {
        "Monte Carlo (Wolff)": {T_crit: mc_tensor.get(T_crit)} if T_crit in mc_tensor else {},
        f"NCSNv2-PIPC ({args.experiment_name})": ncsn_dataset,
        "DDPM": ddpm_dataset,
    }
    tc_datasets = {k: v for k, v in tc_datasets.items() if v}
    plot_spin_correlation_comparison(
        tc_datasets,
        labels=list(tc_datasets.keys()),
        colors=[(0, 0, 0), (0.094, 0.322, 0.616), (0.157, 0.678, 0.337)],
        markers=["o", "s", "^"],
        T_crit=T_crit,
        L=IMAGE_SIZE,
        save_path=os.path.join(args.fig_dir, "Fig6_correlation_comparison.pdf"),
    )

    # 7. Pareto plots
    print("\n[7/7] Generating Pareto plots...")
    import pandas as pd
    ablation_table_path = os.path.join(args.table_dir, "Table_ablation_normalized.csv")
    if os.path.exists(ablation_table_path):
        df_ab = pd.read_csv(ablation_table_path)

        for metric in ["IPS", "W1_Dist"]:
            if metric in df_ab.columns and "Time(ms)" in df_ab.columns:
                df_plot = df_ab[["Exp", metric, "Time(ms)"]].copy()
                df_plot[metric]     = df_plot[metric].astype(float)
                df_plot["Time(ms)"] = df_plot["Time(ms)"].astype(float)
                plot_pareto(
                    df_plot, metric=metric,
                    save_path=os.path.join(args.fig_dir, f"Fig_{7 + ['IPS','W1_Dist'].index(metric)}_pareto_{metric}.pdf"),
                )
    else:
        print(f"  [Info] Ablation table not found at {ablation_table_path}. "
              "Run r3_offline_analysis.py first to generate Pareto plots.")

    print("\n" + "=" * 60)
    print("BLOCK 4 COMPLETE.")
    print(f"  NCSNv2-PIPC: IPS={ips_ncsn:.5f}, W1={w1_ncsn:.5f}")
    print(f"  DDPM:        IPS={ips_ddpm:.5f}, W1={w1_ddpm:.5f}")
    print(f"  Figures saved in  : {args.fig_dir}/")
    print(f"  Table saved in    : {args.table_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
