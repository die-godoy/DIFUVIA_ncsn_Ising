"""
Measure FLOPs per sample — hardware-independent computational cost.

Complements the NFE metric. NFE counts network evaluations; FLOPs weights each
evaluation by its actual arithmetic cost, which is what makes the NCSNv2-vs-DDPM
comparison fair (600 NFE through NCSNv2 != 999 NFE through the DDPM U-Net).

Cheap: no sample generation. Only a handful of forward passes plus one tiny
verification trajectory, so this runs in seconds. Measured on CPU by default —
FLOP counts are device-independent, and CPU avoids MPS dispatch quirks.

Outputs tables/Table_flops.csv with per-config GFLOPs and a hardware-independent
Speedup_FLOPs column vs the DDPM baseline.

Usage
-----
python experiments/measure_flops.py
python experiments/measure_flops.py --skip_ddpm
"""

import os
import argparse
import sys

import pandas as pd
import torch

from difuvia.data_access import ensure_data
from difuvia.nfe import nfe_pipc, nfe_ddpm
from difuvia.flops import (
    score_model_flops_per_eval,
    ddpm_flops_per_eval,
    force_flops_per_eval,
    flops_pipc,
    flops_ddpm,
    verify_pipc_flops,
    print_flops_verification,
)


# Same combined grid as experiments/verify_nfe.py: E1-E12 physics-guided + A1-A3 physics-free.
PIPC_GRID = [
    {"id": "E1",  "M": 100, "K": 2, "lambda0": 0.2},
    {"id": "E2",  "M": 100, "K": 4, "lambda0": 0.2},
    {"id": "E3",  "M": 150, "K": 2, "lambda0": 0.2},
    {"id": "E4",  "M": 150, "K": 3, "lambda0": 0.2},
    {"id": "E5",  "M": 150, "K": 2, "lambda0": 0.4},
    {"id": "E6",  "M": 170, "K": 1, "lambda0": 0.3},
    {"id": "E7",  "M": 200, "K": 4, "lambda0": 0.2},
    {"id": "E8",  "M": 200, "K": 3, "lambda0": 0.3},
    {"id": "E9",  "M": 200, "K": 1, "lambda0": 0.2},
    {"id": "E10", "M": 250, "K": 2, "lambda0": 0.2},
    {"id": "E11", "M": 250, "K": 3, "lambda0": 0.2},
    {"id": "E12", "M": 250, "K": 3, "lambda0": 0.4},
    {"id": "A1",  "M": 150, "K": 1, "lambda0": 0.0},
    {"id": "A2",  "M": 150, "K": 2, "lambda0": 0.0},
    {"id": "A3",  "M": 150, "K": 3, "lambda0": 0.0},
]

GIGA = 1e9


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs per sample (PIPC + DDPM)")
    parser.add_argument("--ncsn_ckpt",  default="networks/ckpt_epoch_300.pth")
    parser.add_argument("--ddpm_ckpt",  default="networks/ising_ddpm_11t.pt")
    parser.add_argument("--config",     default="ncsnv2/configs/ising.yml")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--table_dir",  default="tables")
    parser.add_argument("--skip_ddpm",   action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.table_dir, exist_ok=True)

    # FLOP counts are device-independent; CPU keeps the dispatch path predictable.
    device = torch.device("cpu")
    ensure_data(["networks"])

    from difuvia.model_utils import load_model
    from difuvia.physics import SoftIsingEnergy

    print("=" * 60)
    print("FLOPs MEASUREMENT — computational cost per sample")
    print(f"  Device (measurement): {device}   image_size={args.image_size}")
    print("=" * 60)

    score_model, sigmas, _ = load_model(args.ncsn_ckpt, args.config, device)
    energy_fn = SoftIsingEnergy(J=1.0, double_well_strength=0.6)

    # 1. Per-evaluation costs.
    print("\n[1/4] Measuring per-evaluation cost...")
    f_net = score_model_flops_per_eval(score_model, image_size=args.image_size)
    f_force = force_flops_per_eval(energy_fn, image_size=args.image_size, device=device)

    f_ddpm = None
    if not args.skip_ddpm:
        from difuvia.ddpm import Diffusion, UNet_conditional

        ddpm_model = UNet_conditional(device=device).to(device)
        ddpm_model.load_state_dict(torch.load(args.ddpm_ckpt, map_location=device))
        ddpm_model.eval()
        diffusion = Diffusion(img_size=args.image_size, device=device)
        f_ddpm = ddpm_flops_per_eval(ddpm_model, img_size=args.image_size)

    print("\n" + "=" * 60)
    print("PER-EVALUATION COST (batch=1)")
    print("=" * 60)
    print(f"  NCSNv2 score network : {f_net / GIGA:10.4f} GFLOPs/eval")
    print(f"  Physics force (∇E)   : {f_force / GIGA:10.4f} GFLOPs/eval"
          f"   ({100 * f_force / f_net:.2f}% of one network eval)")
    if f_ddpm is not None:
        print(f"  DDPM U-Net           : {f_ddpm / GIGA:10.4f} GFLOPs/eval"
              f"   ({f_ddpm / f_net:.2f}x the NCSNv2 cost)")
    print("=" * 60)
    print("  Note: counts matmul/conv-class ops only; elementwise work (activations,")
    print("        the double-well term) is not counted, so the force is an underestimate.")

    # 2. Per-config totals.
    print("\n[2/4] Computing per-config totals...")
    ddpm_total = flops_ddpm(1000, f_ddpm) if f_ddpm is not None else None

    rows = []
    for p in PIPC_GRID:
        M, K = p["M"], p["K"]
        nfe = nfe_pipc(M, K)
        g_total = flops_pipc(M, K, f_net, f_force)
        row = {
            "Exp": p["id"],
            "M": M,
            "K": K,
            "λ₀": p["lambda0"],
            "NFE": nfe,
            "GFLOPs": f"{g_total / GIGA:.2f}",
            "_total": g_total,
        }
        if ddpm_total is not None:
            row["FLOPs_ratio_vs_DDPM"] = f"{ddpm_total / g_total:.2f}"
        rows.append(row)

    if ddpm_total is not None:
        rows.append({
            "Exp": "DDPM", "M": "-", "K": "-", "λ₀": "-",
            "NFE": nfe_ddpm(1000),
            "GFLOPs": f"{ddpm_total / GIGA:.2f}",
            "_total": ddpm_total,
            "FLOPs_ratio_vs_DDPM": "1.00",
        })

    df = pd.DataFrame(rows).sort_values(by="_total", ascending=True)
    display_cols = ["Exp", "M", "K", "λ₀", "NFE", "GFLOPs"]
    if ddpm_total is not None:
        display_cols.append("FLOPs_ratio_vs_DDPM")

    print("\n" + "=" * 95)
    print("TABLE: Computational Cost in FLOPs (per generated sample)")
    print("  GFLOPs = NFE * (network + physics-force cost); the force adds <0.001%")
    if ddpm_total is not None:
        print(f"  FLOPs_ratio_vs_DDPM = DDPM / config  (DDPM baseline = {ddpm_total / GIGA:.2f} GFLOPs/sample)")
        print("  >1 = less arithmetic than DDPM. This is a WORK ratio, NOT a speed ratio —")
        print("  it does not predict wall-clock time (see difuvia/flops.py limitations).")
    print("=" * 95)
    print(df[display_cols].to_string(index=False))
    print("=" * 95)

    # 3. Save.
    print("\n[3/4] Saving table...")
    csv_path = os.path.join(args.table_dir, "Table_flops.csv")
    df.drop(columns=["_total"]).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # 4. Verify the closed form against a real trajectory.
    ok = True
    if not args.skip_verify:
        print("\n[4/4] Verifying closed form on one short trajectory...")
        info = verify_pipc_flops(score_model, energy_fn, sigmas, f_net, f_force,
                                 device=device, image_size=args.image_size)
        ok = print_flops_verification(info)
    else:
        print("\n[4/4] Verification skipped (--skip_verify).")

    if not ok:
        print("\n[STOP] FLOPs closed form does not match the measured trajectory.")
        sys.exit(1)
    print("\nFLOPs measurement complete.")


if __name__ == "__main__":
    main()
