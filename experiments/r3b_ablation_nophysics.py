"""
BLOCK 3b: Physics-FREE ablation (λ₀ = 0).

Isolates the contribution of the physical guidance by running the SAME PC
sampler and the SAME trained weights with the guidance turned off (λ₀ = 0),
at M = 150 and K ∈ {1, 2, 3}. This is the "how much does the physics add?"
comparison a committee will ask for: A1–A3 vs the physics-guided E1–E12.

These physics-free samples are kept SEPARATE from the physics-guided ones:
  - physics-guided E1–E12 :  ablation_samples/         (Table_ablation_normalized.csv)
  - physics-free   A1–A3  :  ablation_samples_nophysics/ (Table_ablation_nophysics.csv)

Data source
-----------
--source download : fetch pre-generated A1–A3 from HF if available, else error
                    telling you to generate (they are not shipped by default).
--source generate : run the score model to create A1–A3 (needs the NCSN
                    checkpoint; no retraining, same weights as E1–E12).

Usage
-----
python experiments/r3b_ablation_nophysics.py --source generate
python experiments/r3b_ablation_nophysics.py --source download   # if uploaded
"""

import os
import argparse
import warnings

warnings.filterwarnings("ignore")

from difuvia.thermodynamics import load_mc_ground_truth, load_mc_correlation_at_tc
from difuvia.sampling import load_offline_samples_and_compute
from difuvia.analysis import print_ablation_table, fit_time_vs_nfe
from difuvia.viz import plot_all_observables, plot_correlation_at_tc
from difuvia.data_access import ensure_data


# Physics-free grid: M=150 fixed, K swept, λ₀ = 0 (no physical guidance).
PHYSFREE_GRID = [
    {"id": "A1", "M": 150, "K": 1, "lambda0": 0.0}, # pure langevin   
    {"id": "A2", "M": 150, "K": 2, "lambda0": 0.0},
    {"id": "A3", "M": 150, "K": 3, "lambda0": 0.0},
]

NOPHYS_DIR = "ablation_samples_nophysics"

# Physics-guided reference from the main ablation (r3_ablation.py), overlaid
# on the observables figure only; it is not part of the A1–A3 table.
E4_CONFIG = {"id": "E4", "M": 150, "K": 3, "lambda0": 0.2}


def main():
    parser = argparse.ArgumentParser(description="R3b: physics-free (λ₀=0) ablation")
    parser.add_argument("--source", choices=["download", "generate"], default="generate",
                        help="download pre-generated A1–A3 from HF, or generate from checkpoint")
    parser.add_argument("--checkpoint", default="networks/ckpt_epoch_300.pth")
    parser.add_argument("--config",     default="ncsnv2/configs/ising.yml")
    parser.add_argument("--data_dir",   default="train_data")
    parser.add_argument("--save_dir",   default=NOPHYS_DIR)
    parser.add_argument("--ablation_dir", default="ablation_samples",
                        help="Physics-guided E1–E12 samples; E4 is overlaid on the figure.")
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--fig_dir",    default="figures")
    parser.add_argument("--table_dir",  default="tables")
    args = parser.parse_args()

    TEMPERATURES = [round(1.77 + i * 0.1, 2) for i in range(11)]
    IMAGE_SIZE = 64
    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)

    print("=" * 60)
    print("BLOCK 3b: PHYSICS-FREE ABLATION (λ₀ = 0)")
    print(f"  Configs      : {[p['id'] for p in PHYSFREE_GRID]}  (M=150, K∈{{1,2,3}})")
    print(f"  Samples dir  : {args.save_dir}/")
    print(f"  Source       : {args.source}")
    print("=" * 60)

    # MC ground truth (needed for metrics; fetched from HF if absent).
    ensure_data([args.data_dir])
    print("\n[1/4] Loading Monte Carlo ground truth...")
    mc_gt = load_mc_ground_truth(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)
    T_crit, mc_corr_tc = load_mc_correlation_at_tc(args.data_dir, TEMPERATURES, L=IMAGE_SIZE)

    # Obtain the A1–A3 samples.
    print(f"\n[2/4] Preparing physics-free samples ({args.source})...")
    if args.source == "generate":
        from difuvia.physics import SoftIsingEnergy
        from difuvia.model_utils import load_model, get_device
        from difuvia.sampling import run_experiment

        ensure_data(["networks"])
        device = get_device()
        score_model, sigmas, _ = load_model(args.checkpoint, args.config, device)
        energy_fn = SoftIsingEnergy(J=1.0, double_well_strength=0.6)
        results = run_experiment(
            score_model=score_model, sigmas=sigmas, energy_fn=energy_fn,
            mc_ground_truth=mc_gt, temperatures=TEMPERATURES,
            hyperparameter_grid=PHYSFREE_GRID, n_samples_per_temp=args.n_samples,
            image_size=IMAGE_SIZE, device=device, save_dir=args.save_dir,
            save_samples=True,
        )
        hardware = str(device)
    else:  # download
        ensure_data([args.save_dir])
        if not (os.path.isdir(args.save_dir) and os.listdir(args.save_dir)):
            raise SystemExit(
                f"No physics-free samples found in {args.save_dir}/ and none on HF. "
                f"Run with --source generate to create them.")
        results = load_offline_samples_and_compute(
            hyperparameter_grid=PHYSFREE_GRID, temperatures=TEMPERATURES,
            mc_ground_truth=mc_gt, image_size=IMAGE_SIZE, save_dir=args.save_dir,
        )
        hardware = None

    if not results:
        print("[ERROR] No physics-free results available.")
        return

    # Figures + table (separate CSV so E1–E12 output is untouched).
    print("\n[3/4] Generating figures...")
    ensure_data([args.ablation_dir])
    e4_results = load_offline_samples_and_compute(
        hyperparameter_grid=[E4_CONFIG], temperatures=TEMPERATURES,
        mc_ground_truth=mc_gt, image_size=IMAGE_SIZE, save_dir=args.ablation_dir,
    )
    plot_all_observables(
        results + e4_results, mc_gt, TEMPERATURES,
        highlight="E4", fade_others=False,
        save_path=os.path.join(args.fig_dir, "Fig3b_nophysics_observables.pdf"),
    )
    if T_crit:
        plot_correlation_at_tc(
            results, mc_corr_tc, T_crit, L=IMAGE_SIZE,
            save_path=os.path.join(args.fig_dir, "Fig3b_nophysics_correlation.pdf"),
        )

    print("\n[4/4] Results table + time-vs-NFE fit...")
    print_ablation_table(
        results, mc_gt, TEMPERATURES, save_csv=True,
        batch_size=50, hardware=hardware,
        csv_path=os.path.join(args.table_dir, "Table_ablation_nophysics.csv"),
    )
    fit_time_vs_nfe(results)

    print("\n" + "=" * 60)
    print("BLOCK 3b COMPLETE (physics-free λ₀=0).")
    print("=" * 60)


if __name__ == "__main__":
    main()
