"""
All matplotlib figure functions for the DIFUVIA project.

Sections:
- R1: Dataset validation — lattice images and thermodynamic observables vs exact theory
- R2: Marchenko-Pastur — eigenvalue spectrum comparison
- R3: Ablation study — observables and G(r) across PC sampler configurations
- R4: Comparative study — NCSNv2 vs DDPM with error bars and lattice grids
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D

from difuvia.physics import Tc

UV_GREEN = (0.157, 0.678, 0.337)
UV_BLUE  = (0.094, 0.322, 0.616)
UV_RED   = (0.800, 0.180, 0.180)


def _label_panels(axes, fontsize=16):
    for ax, letter in zip(np.ravel(axes), "abcdefgh"):
        ax.text(0.0, 1.02, f"{letter})", transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", va="bottom", ha="left")


# ──────────────────────────────────────────────────────────────
# R1: Dataset Validation
# ──────────────────────────────────────────────────────────────

def plot_representative_lattices(dataset: dict, temperatures_to_show=None, save_path: str = None):
    """
    Three representative lattice configurations: ordered / critical / disordered.

    dataset: {T: np.array(N, L, L)} or {T: torch.Tensor(N, 1, L, L)}
    """
    if temperatures_to_show is None:
        all_T = sorted(dataset.keys())
        temperatures_to_show = [
            all_T[0],
            min(all_T, key=lambda t: abs(t - Tc)),
            all_T[-1],
        ]

    fig, axes = plt.subplots(1, len(temperatures_to_show),
                             figsize=(5 * len(temperatures_to_show), 5))
    if len(temperatures_to_show) == 1:
        axes = [axes]

    labels = {
        temperatures_to_show[0]:  "Ordered Phase",
        temperatures_to_show[-1]: "Disordered Phase",
        min(temperatures_to_show, key=lambda t: abs(t - Tc)): "Critical Region",
    }

    for ax, T in zip(axes, temperatures_to_show):
        sample = dataset[T][0]
        if hasattr(sample, "numpy"):
            sample = sample.numpy()
        if sample.ndim == 3:
            sample = sample.squeeze(0)
        ax.imshow(sample, cmap="gray", interpolation="nearest", vmin=-1, vmax=1)
        ax.set_title(f"T = {T:.2f}\n{labels.get(T, '')}", fontsize=13)
        ax.axis("off")

    plt.suptitle("Representative Ising Configurations (Monte Carlo, Wolff Algorithm)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


def plot_thermodynamic_observables(df, theory: dict, save_path: str = None):
    """
    Four-panel: MC observables (DataFrame) vs exact theory curves.

    df     : pandas DataFrame with columns T, M_mean, M_std, E_mean, E_std, Cv, Chi
    theory : dict from get_theory_curves() with keys T, M, E, Cv, Chi
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    T_mc = df["T"].values

    panels = [
        ("M_mean", "M_std", theory["M"],            UV_BLUE,  "o",
         r"$\langle |m| \rangle$", "Mean Absolute Magnetization", "(Yang 1952)"),
        ("E_mean", "E_std", theory["E"],            UV_GREEN, "s",
         r"$\langle e \rangle$",    "Mean Energy per Spin",       "(Onsager 1944)"),
        ("Cv",     None,    np.clip(theory["Cv"], -50, 50), UV_BLUE, "^",
         r"$C_v$",                  "Specific Heat per Spin",     "(numerical deriv.)"),
        ("Chi",    None,    np.clip(theory["Chi"], 0, 100), UV_GREEN, "D",
         r"$\chi$",                  "Magnetic Susceptibility",    "$\\gamma=7/4$"),
    ]

    for ax, (key, err_key, th_y, mc_col, marker, ylabel, title, legend_label) in zip(axes, panels):
        ax.plot(theory["T"], th_y, color="gray", lw=2, zorder=1, label=legend_label)
        if err_key:
            ax.errorbar(T_mc, df[key], yerr=df[err_key],
                        fmt=marker, color=mc_col, capsize=4, ms=6, lw=1.5, zorder=2)
        else:
            ax.scatter(T_mc, df[key], marker=marker, color=mc_col, s=60, zorder=2)
        ax.axvline(Tc, color="crimson", ls="--", alpha=0.7)
        ax.set_xlabel("$T^*$", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid(True, alpha=0.3)

    _label_panels(axes)

    legend_handles = [
        mlines.Line2D([], [], color="gray", lw=2, label="Exact solution"),
        mlines.Line2D([], [], color="crimson", lw=1.5, ls="--",
                      label=rf"$T_c = {Tc:.4f}$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=11, frameon=True, framealpha=0.9,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# R2: Marchenko-Pastur
# ──────────────────────────────────────────────────────────────

def plot_eigenvalue_spectra(clean_data, sigma_star: float, save_path: str = None):
    """
    Two-panel eigenvalue spectrum: clean data vs after σ₁* noise injection.

    Visually demonstrates why σ₁* is the correct upper bound for the noise schedule.
    """
    X_clean = clean_data.view(clean_data.shape[0], -1).cpu().numpy()
    N, P = X_clean.shape
    Q = N / P
    lambda_plus = (1 + np.sqrt(1 / Q)) ** 2

    def get_eigvals(X):
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        _, s, _ = np.linalg.svd(X_norm, full_matrices=False)
        return (s ** 2) / (N - 1)

    eigvals_clean = get_eigvals(X_clean)
    eigvals_noisy = get_eigvals(X_clean + sigma_star * np.random.normal(0, 1, X_clean.shape))

    ratio_clean = np.max(eigvals_clean) / lambda_plus
    ratio_noisy = np.max(eigvals_noisy) / lambda_plus

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(eigvals_clean, bins=60, density=True, alpha=0.75,
                 color=UV_BLUE, edgecolor="white", linewidth=0.3, label="Empirical spectrum")
    axes[0].axvline(lambda_plus, color="red", linestyle="--", linewidth=2,
                    label=f"$\\lambda_+ = {lambda_plus:.3f}$")
    axes[0].axvline(np.max(eigvals_clean), color="orange", linestyle="-", linewidth=2,
                    label=f"$\\lambda_{{\\max}} = {np.max(eigvals_clean):.2f}$  "
                          f"($\\rho = {ratio_clean:.2f}$)")
    axes[0].set_xlabel("$\\lambda$", fontsize=16)
    axes[0].set_ylabel("Density", fontsize=16)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(eigvals_noisy, bins=60, density=True, alpha=0.75,
                 color=UV_GREEN, edgecolor="white", linewidth=0.3, label="Empirical spectrum")
    axes[1].axvline(lambda_plus, color="red", linestyle="--", linewidth=2,
                    label=f"$\\lambda_+ = {lambda_plus:.3f}$")
    axes[1].axvline(np.max(eigvals_noisy), color="orange", linestyle="-", linewidth=2,
                    label=f"$\\lambda_{{\\max}} = {np.max(eigvals_noisy):.3f}$  "
                          f"($\\rho = {ratio_noisy:.3f}$)")
    axes[1].set_xlabel("$\\lambda$", fontsize=16)
    axes[1].set_ylabel("Density", fontsize=16)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    _label_panels(axes)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# R3: Ablation Study
# ──────────────────────────────────────────────────────────────

def plot_all_observables(
    results: list,
    mc_ground_truth: dict,
    temperatures: list,
    save_path: str = None,
):
    """
    Four-panel: thermodynamic observables vs T for each ablation experiment.

    results: list of dicts from run_experiment or load_offline_samples_and_compute.
    """
    T_arr  = sorted(temperatures)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    panels = [
        ("M_mean", r"$\langle |m| \rangle$", "Mean Absolute Magnetization"),
        ("E_mean", r"$\langle e \rangle$",    "Mean Energy per Spin"),
        ("Cv",     r"$C_v$",                  "Specific Heat per Spin"),
        ("Chi",    r"$\chi$",                  "Magnetic Susceptibility"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        T_mc    = [T for T in T_arr if T in mc_ground_truth]
        mc_vals = [mc_ground_truth[T][key] for T in T_mc]
        ax.plot(T_mc, mc_vals, "k-", linewidth=2.5, zorder=10,
                label="Monte Carlo (ground truth)")

        for res, color in zip(results, colors):
            T_gen  = sorted(res["obs_per_temp"].keys())
            y_vals = [res["obs_per_temp"][T][key] for T in T_gen]
            label  = (f"{res['exp_id']}: M={res['M']}, "
                      f"K={res['K']}, $\\lambda_0$={res['lambda0']}")
            ax.plot(T_gen, y_vals, "o--", color=color, linewidth=1.2,
                    markersize=5, alpha=0.85, label=label)

        ax.axvline(Tc, color="red", linestyle=":", alpha=0.5, linewidth=1,
                   label="$T_c$" if key == "M_mean" else "")
        ax.set_xlabel(" $T^*$", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.grid(True, alpha=0.3)
        if key == "M_mean":
            ax.legend(fontsize=8, ncol=2, loc="upper right")

    _label_panels(axes)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


def plot_correlation_at_tc(
    results: list,
    mc_corr_at_tc,
    T_crit: float,
    L: int = 64,
    save_path: str = None,
):
    """
    Log-log G(r) at T~Tc for all ablation experiments vs MC ground truth.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    if mc_corr_at_tc is not None:
        r_mc, G_mc = mc_corr_at_tc
        mask = (r_mc >= 1) & (G_mc > 0)
        ax.plot(r_mc[mask], G_mc[mask], "k-", linewidth=2.5, zorder=10,
                label="Monte Carlo (ground truth)")
        r_ref = r_mc[mask]
        A_ref = G_mc[mask][0] * (r_ref[0] ** 0.25)
        ax.plot(r_ref, A_ref * r_ref ** (-0.25), "r:", linewidth=1.5,
                label=r"Theory: $\eta=0.25$ (2D Ising)")

    for res, color in zip(results, colors):
        if res["corr_at_tc"] is None:
            continue
        r_vals, G_r = res["corr_at_tc"]
        mask = (r_vals >= 1) & (G_r > 0)
        label = (f"{res['exp_id']}: M={res['M']}, "
                 f"K={res['K']}, $\\lambda_0$={res['lambda0']}")
        ax.plot(r_vals[mask], G_r[mask], "o--", color=color,
                markersize=4, linewidth=1.2, alpha=0.85, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance $r$", fontsize=12)
    ax.set_ylabel("$G(r)$", fontsize=12)
    ax.set_title(f"Spin-Spin Correlation at $T = {T_crit:.2f} \\approx T_c$ — Ablation Study",
                 fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# R4: Comparative Study
# ──────────────────────────────────────────────────────────────

def plot_thermodynamic_comparison(
    obs_mc: dict,
    obs_ncsn: dict,
    obs_ddpm: dict,
    theory: dict,
    temperatures: list,
    experiment_name: str = "",
    save_path: str = None,
):
    """
    Four-panel thermodynamic comparison: MC, NCSNv2, DDPM vs exact theory.
    Includes error bars (M_se, E_se) when available.
    """
    T_arr = sorted(temperatures)

    def _get_vals(obs, key):
        T_vals = [T for T in T_arr if T in obs]
        means  = [obs[T][key] for T in T_vals]
        se_key = key.replace("mean", "se").replace("Cv", "").replace("Chi", "")
        ses    = [obs[T].get(se_key, 0) for T in T_vals]
        return np.array(T_vals), np.array(means), np.array(ses)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    panels = [
        ("M_mean", r"$\langle |m| \rangle$", "Mean Absolute Magnetization", "M"),
        ("E_mean", r"$\langle e \rangle$",    "Mean Energy per Spin",        "E"),
        ("Cv",     r"$C_v$",                  "Specific Heat per Spin",      "Cv"),
        ("Chi",    r"$\chi$",                  "Magnetic Susceptibility",     "Chi"),
    ]

    for ax, (key, ylabel, title, th_key) in zip(axes, panels):
        th_vals = np.clip(theory[th_key], -200, 200)
        ax.plot(theory["T"], th_vals, "-", color="gray",
                linewidth=2, alpha=0.7, label="Exact theory", zorder=1)

        for (obs, color, marker, label) in [
            (obs_mc,   "black",   "o", "MC Wolff (ground truth)"),
            (obs_ncsn, UV_BLUE,   "s", f"NCSNv2-PIPC ({experiment_name})"),
            (obs_ddpm, UV_GREEN,  "^", "DDPM"),
        ]:
            T_v, means, ses = _get_vals(obs, key)
            if ses.any():
                ax.errorbar(T_v, means, yerr=ses, fmt=marker,
                            color=color, capsize=4, markersize=6,
                            linewidth=1.5, label=label, zorder=3)
            else:
                ax.scatter(T_v, means, color=color, marker=marker, s=50,
                           zorder=3, label=label)

        ax.axvline(Tc, color="red", linestyle=":", alpha=0.6, linewidth=1.2,
                   label=f"$T_c = {Tc:.4f}$" if key == "M_mean" else "")
        ax.set_xlabel("$T^*$", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        if key == "M_mean":
            ax.legend(fontsize=9, loc="upper right")

    plt.suptitle("Thermodynamic Fidelity Comparison", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


def plot_spin_correlation_comparison(
    datasets: dict,
    labels: list,
    colors: list,
    markers: list,
    T_crit: float = None,
    L: int = 64,
    save_path: str = None,
):
    """
    Log-log G(r) comparison at T~Tc for multiple models.

    datasets: dict {model_name: {T: torch.Tensor(N, 1, L, L)}}
    """
    from difuvia.thermodynamics import compute_spin_correlation

    if T_crit is None:
        all_T = list(next(iter(datasets.values())).keys())
        T_crit = min(all_T, key=lambda t: abs(t - Tc))

    fig, ax = plt.subplots(figsize=(9, 6))
    model_names = list(datasets.keys())

    for idx, (name, dataset) in enumerate(datasets.items()):
        if T_crit not in dataset:
            continue
        r_vals, G_r = compute_spin_correlation(dataset[T_crit])
        mask = (r_vals >= 1) & (G_r > 0)
        ls   = "-" if idx == 0 else "--"
        ax.plot(r_vals[mask], G_r[mask], f"o{ls}",
                color=colors[idx], markersize=5, linewidth=1.8,
                alpha=0.9, label=name)

    # Theoretical reference η = 0.25
    r_ref = np.linspace(1, L // 2, 200)
    first_name = model_names[0]
    if T_crit in datasets[first_name]:
        r_mc, G_mc = compute_spin_correlation(datasets[first_name][T_crit])
        mask_mc = (r_mc >= 1) & (G_mc > 0)
        A_ref   = G_mc[mask_mc][0] * (r_mc[mask_mc][0] ** 0.25)
    else:
        A_ref = 0.3
    ax.plot(r_ref, A_ref * r_ref ** (-0.25), "r:",
            linewidth=2, label=r"Theory: $\eta = 0.25$ (2D Ising)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance $r$", fontsize=12)
    ax.set_ylabel("$G(r)$", fontsize=12)
    ax.set_title(f"Spin-to-Spin Correlation at $T^* = {T_crit:.2f} \\approx T_c$", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved: {save_path}")
    plt.show()


def plot_lattice_comparison(
    datasets: dict,
    temperatures_show: list,
    n_samples: int = 3,
    L: int = 64,
    save_path: str = None,
):
    """
    Grid of representative lattice configurations across models and temperatures.

    datasets: dict {model_name: {T: torch.Tensor(N, 1, L, L)}}
    """
    model_names = list(datasets.keys())
    n_models    = len(model_names)
    n_T         = len(temperatures_show)

    fig, axes = plt.subplots(n_models, n_T * n_samples,
                             figsize=(n_T * n_samples * 2.2, n_models * 2.5))

    for row, name in enumerate(model_names):
        dataset = datasets[name]
        for col_group, T in enumerate(temperatures_show):
            if T not in dataset:
                continue
            configs = dataset[T]
            for s in range(n_samples):
                col = col_group * n_samples + s
                ax  = axes[row, col]
                sample = configs[s, 0]
                if hasattr(sample, "numpy"):
                    sample = sample.numpy()
                ax.imshow(sample, cmap="gray", vmin=-1, vmax=1,
                          interpolation="nearest")
                ax.axis("off")
                if row == 0 and s == n_samples // 2:
                    ax.set_title(f"$T^*$={T:.2f}", fontsize=15)
                if col == 0:
                    ax.text(-0.15, 0.5, name, transform=ax.transAxes,
                            fontsize=12, rotation=90, ha="right", va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved: {save_path}")
    plt.show()


def plot_pareto(
    df_analysis,
    metric: str = "IPS",
    ddpm_reference: dict = None,
    save_path: str = None,
):
    """
    Pareto frontier plot: inference time vs quality metric (IPS or W1_Dist).

    df_analysis     : DataFrame with columns 'Time(ms)', metric, 'Exp'
    ddpm_reference  : optional dict with keys 'time_ms' and the metric value
    """
    from difuvia.analysis import pareto_frontier

    times    = df_analysis["Time(ms)"].values.astype(float)
    scores   = df_analysis[metric].values.astype(float)
    exp_ids  = df_analysis["Exp"].values

    frontier     = pareto_frontier(times, scores, exp_ids)
    frontier_ids = {p[2] for p in frontier}

    fig, ax = plt.subplots(figsize=(11, 7))

    for t, ip, eid in zip(times, scores, exp_ids):
        on_frontier = eid in frontier_ids
        ax.scatter(t, ip,
                   color=UV_RED if on_frontier else UV_BLUE,
                   marker="D" if on_frontier else "o",
                   s=120 if on_frontier else 80,
                   zorder=4, alpha=0.9)
        ax.annotate(eid, xy=(t, ip), textcoords="offset points", xytext=(6, 5),
                    ha="left", fontsize=9,
                    color=UV_RED if on_frontier else UV_BLUE,
                    fontweight="bold" if on_frontier else "normal")

    if len(frontier) > 1:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.step(fx, fy, where="post", color=UV_RED,
                linewidth=2, linestyle="--", alpha=0.7, label="Pareto frontier", zorder=3)

    if ddpm_reference is not None:
        ddpm_t = ddpm_reference["time_ms"]
        ddpm_s = ddpm_reference[metric]
        ax.scatter(ddpm_t, ddpm_s, color=UV_GREEN, s=150,
                   marker="v", zorder=5, label="DDPM (baseline)")
        ax.annotate("DDPM", (ddpm_t, ddpm_s),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=9, color=UV_GREEN, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=UV_RED,
               markersize=10, label="Pareto-optimal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=UV_BLUE,
               markersize=9, label="PC Sampler configurations"),
        Line2D([0], [0], linestyle="--", color=UV_RED, linewidth=2, label="Pareto frontier"),
    ]
    if ddpm_reference is not None:
        legend_elements.append(
            Line2D([0], [0], marker="v", color="w", markerfacecolor=UV_GREEN,
                   markersize=10, label="DDPM (baseline)")
        )
    ax.legend(handles=legend_elements, fontsize=14, loc="upper right")
    ax.set_xlabel("Inference time (ms)", fontsize=15)
    ax.set_ylabel(metric, fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved: {save_path}")
    plt.show()
