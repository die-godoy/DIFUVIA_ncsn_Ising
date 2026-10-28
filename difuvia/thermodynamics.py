"""
Thermodynamic observables and Monte Carlo data loaders.

All functions operate on torch.Tensor inputs (N, 1, L, L) unless noted.
compute_observables is the canonical unified function used by all experiments.
"""

import os
import json
import numpy as np
import torch

from difuvia.physics import Tc


def compute_observables(
    configs_tensor: torch.Tensor,
    T: float,
    L: int = 64,
    n_bootstrap: int = 0,
) -> dict:
    """
    Compute all thermodynamic observables from a batch of spin configurations.

    Parameters
    ----------
    configs_tensor : torch.Tensor (N, 1, L, L), values in {-1, +1}
    T              : temperature
    L              : linear lattice size
    n_bootstrap    : bootstrap resamples for standard errors
                     0 → use sample std (fast, for ablation loops)
                     >0 → bootstrap SE (accurate, for final figures)

    Returns
    -------
    dict with M_mean, M_std, M_se, E_mean, E_std, E_se, Cv, Chi,
    m_vals, m_signed, e_vals.
    M_se / E_se are aliases for M_std / E_std for plot_thermodynamic_comparison.
    """
    x = configs_tensor.float().squeeze(1)  # (N, L, L)
    N_spins = L * L

    m = x.mean(dim=(1, 2))           # signed magnetization (N,)
    m_abs = m.abs()
    M_mean = m_abs.mean().item()

    x4 = x.unsqueeze(1)
    neighbors = (
        torch.roll(x4, -1, 3) + torch.roll(x4, 1, 3)
        + torch.roll(x4, -1, 2) + torch.roll(x4, 1, 2)
    ).squeeze(1)
    e = (-0.5 * x * neighbors).mean(dim=(1, 2))
    E_mean = e.mean().item()

    e_np  = e.cpu().numpy()
    m_np  = m.cpu().numpy()
    ma_np = m_abs.cpu().numpy()

    Cv  = (N_spins / T ** 2) * np.var(e_np, ddof=1)
    Chi = max((N_spins / T) * (np.mean(m_np ** 2) - M_mean ** 2), 0.0)

    if n_bootstrap > 0:
        def _bs(arr: np.ndarray) -> float:
            return float(np.std(
                [np.mean(np.random.choice(arr, len(arr), replace=True))
                 for _ in range(n_bootstrap)]
            ))
        M_std = _bs(ma_np)
        E_std = _bs(e_np)
    else:
        M_std = float(m_abs.std(correction=1).item())
        E_std = float(e.std(correction=1).item())

    return {
        "M_mean": M_mean, "M_std": M_std, "M_se": M_std,
        "E_mean": E_mean, "E_std": E_std, "E_se": E_std,
        "Cv": Cv, "Chi": Chi,
        "m_vals": ma_np, "m_signed": m_np, "e_vals": e_np,
    }


def bootstrap_errors(obs: dict, T: float, L: int = 64,
                     n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """
    Bootstrap standard errors of <|m|>, <e>, Cv and Chi.

    Each resample draws N configurations with replacement from the per-sample
    values stored in `obs` (from compute_observables) and recomputes all four
    observables on it; the SE is the std of each over the resamples.

    Returns dict with M_se, E_se, Cv_se, Chi_se.
    """
    N_spins = L * L
    m, e = obs["m_signed"], obs["e_vals"]
    n = len(m)
    idx = np.random.default_rng(seed).integers(0, n, (n_bootstrap, n))
    m_b, e_b = m[idx], e[idx]

    M_b   = np.abs(m_b).mean(axis=1)
    E_b   = e_b.mean(axis=1)
    Cv_b  = (N_spins / T ** 2) * e_b.var(axis=1, ddof=1)
    Chi_b = np.maximum((N_spins / T) * ((m_b ** 2).mean(axis=1) - M_b ** 2), 0.0)

    return {
        "M_se": float(M_b.std()), "E_se": float(E_b.std()),
        "Cv_se": float(Cv_b.std()), "Chi_se": float(Chi_b.std()),
    }


def compute_spin_correlation(configs_tensor: torch.Tensor):
    """
    Isotropic connected spin-spin correlation G(r) via FFT (Wiener-Khinchin).

    configs_tensor : torch.Tensor (N, 1, L, L)
    Returns: r_vals (np.ndarray), G_r (np.ndarray)
    """
    x = configs_tensor.float().squeeze(1).cpu().numpy()
    B, L, _ = x.shape
    x = x - x.mean(axis=(1, 2), keepdims=True)

    f = np.fft.fft2(x)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft2(power).real / (L * L)
    autocorr = np.fft.fftshift(autocorr, axes=(1, 2)).mean(axis=0)

    Y, X = np.ogrid[:L, :L]
    center = L // 2
    r_grid = np.sqrt((X - center) ** 2 + (Y - center) ** 2).flatten()
    c_vals = autocorr.flatten()

    bins = np.arange(0, center + 1)
    indices = np.digitize(r_grid, bins)
    G_r, r_out = [], []
    for i in range(1, len(bins)):
        mask = indices == i
        if np.any(mask):
            G_r.append(c_vals[mask].mean())
            r_out.append(bins[i - 1])

    return np.array(r_out, dtype=float), np.array(G_r)


def load_all_temperatures(data_dir: str, temperatures: list, L: int = 64) -> dict:
    """
    Load all MC JSON files and return {T: np.ndarray(N, L, L)}.

    Used in R1 (dataset validation) where pure-NumPy arrays are needed.
    """
    dataset = {}
    for T in temperatures:
        candidates = [
            os.path.join(data_dir, f"isingW_{L}_{T:.2f}.json"),
            os.path.join(data_dir, f"ising_wolff_{L}_{T:.2f}.json"),
            os.path.join(data_dir, f"ising_wolff_{L}_{T}.json"),
        ]
        filepath = next((c for c in candidates if os.path.exists(c)), None)
        if filepath is None:
            print(f"[WARNING] File not found for T={T:.2f} — skipping.")
            continue
        with open(filepath, "r") as f:
            data = json.load(f)
        configs = np.array([entry["config"] for entry in data], dtype=np.float32)
        dataset[T] = configs
        print(f"  T={T:.2f} → {configs.shape[0]} samples from {os.path.basename(filepath)}")
    print(f"Total temperatures loaded: {len(dataset)}/{len(temperatures)}")
    return dataset


def load_mc_ground_truth(data_dir: str, temperatures: list, L: int = 64) -> dict:
    """
    Load MC observables from JSON files.
    Returns {T: obs_dict} computed via compute_observables (n_bootstrap=0).
    """
    mc_data = {}
    for T in temperatures:
        for fname in [f"isingW_{L}_{T:.2f}.json", f"ising_wolff_{L}_{T:.2f}.json"]:
            path = os.path.join(data_dir, fname)
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                configs = torch.tensor(
                    [entry["config"] for entry in data], dtype=torch.float32
                ).unsqueeze(1)
                mc_data[T] = compute_observables(configs, T, L)
                break
    print(f"MC ground truth loaded for {len(mc_data)} temperatures.")
    return mc_data


def load_mc_correlation_at_tc(data_dir: str, temperatures: list, L: int = 64):
    """
    Compute MC G(r) at the temperature closest to Tc.
    Returns (T_crit, (r_vals, G_r)) or (None, None).
    """
    T_crit = min(temperatures, key=lambda t: abs(t - Tc))
    for fname in [f"isingW_{L}_{T_crit:.2f}.json", f"ising_wolff_{L}_{T_crit:.2f}.json"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            configs = torch.tensor(
                [entry["config"] for entry in data], dtype=torch.float32
            ).unsqueeze(1)
            r, G = compute_spin_correlation(configs)
            return T_crit, (r, G)
    return None, None
