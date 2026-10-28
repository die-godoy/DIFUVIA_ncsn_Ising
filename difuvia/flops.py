"""
FLOPs instrumentation: hardware-independent computational cost.

Complements NFE (see difuvia.nfe). NFE counts *how many* times the learned
network is evaluated; FLOPs counts *how much arithmetic* those evaluations
cost. The distinction matters for the NCSNv2-vs-DDPM comparison: E4's 600 NFE
and DDPM's 999 NFE are counts through two different architectures (NCSNv2 vs
UNet_conditional), so their ratio is not a cost ratio. FLOPs puts both on a
common currency, and unlike Time(ms) it does not depend on MPS vs CUDA vs CPU.

Closed forms
------------
- PIPC : FLOPs = M*(K+1) * (f_net + f_force)
         The force multiplier is the SAME M*(K+1) as the network's, because
         `pc_physics_guided_sampler` calls `energy_fn.get_grad(x)` once per
         corrector step and once per predictor step.
- DDPM : FLOPs = (noise_steps - 1) * f_net

Note on the physics-free ablation
---------------------------------
`pc_physics_guided_sampler` calls `energy_fn.get_grad(x)` unconditionally — it
does not check `guidance_scale > 0`, it only scales the result by λ₀. So the
physics-free configs A1–A3 (λ₀ = 0) still pay the full force cost, and A3
(M=150, K=3) has IDENTICAL FLOPs to E4 (M=150, K=3). This is a property of the
implementation being measured, reported as-is rather than silently patched.

Limitations (state these alongside any reported number)
-------------------------------------------------------
1. `FlopCounterMode` counts only matmul/conv-class operations. Elementwise work
   (activations, normalizations, and the double-well term (x²-1)² in
   `SoftIsingEnergy.energy`) is NOT counted. `f_force` therefore captures mainly
   its F.conv2d and is an UNDERESTIMATE of the true force cost.
2. FLOPs ignores memory bandwidth and assumes every evaluation costs the same,
   so it does not perfectly predict wall-clock time. It complements Time(ms).
3. Within a single architecture FLOPs is exactly NFE times a constant, so it
   adds no new ranking information across E1–E12/A1–A3. Its value is the
   cross-architecture comparison and exposing the force cost NFE cannot see.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode

from difuvia.nfe import nfe_pipc, nfe_ddpm


def _count_flops(fn, *args, **kwargs) -> int:
    """Total matmul/conv FLOPs performed by `fn(*args, **kwargs)`."""
    counter = FlopCounterMode(display=False)
    with counter:
        fn(*args, **kwargs)
    return counter.get_total_flops()


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


# ──────────────────────────────────────────────────────────────
# Per-evaluation measurements (normalized to batch = 1)
# ──────────────────────────────────────────────────────────────

def score_model_flops_per_eval(score_model, image_size: int = 64,
                               probe_batch: int = 1, target_temp: float = 2.27) -> float:
    """
    FLOPs for ONE NCSNv2 evaluation, per sample.

    Inputs mirror the real call site in `pc_physics_guided_sampler`:
    `score_model(x, labels, y_temp=temp_tensor)`.
    """
    device = _model_device(score_model)
    x = torch.randn(probe_batch, 1, image_size, image_size, device=device)
    labels = torch.zeros(probe_batch, device=device, dtype=torch.long)
    temp = torch.full((probe_batch,), target_temp, device=device, dtype=torch.float)

    def _fwd():
        with torch.no_grad():
            score_model(x, labels, y_temp=temp)

    return _count_flops(_fwd) / probe_batch


def ddpm_flops_per_eval(model, img_size: int = 64, probe_batch: int = 1,
                        target_temp: float = 2.27) -> float:
    """
    FLOPs for ONE DDPM UNet_conditional evaluation, per sample.

    Inputs mirror the real call site in `Diffusion.sample`: `model(x, t, labels)`.
    """
    device = _model_device(model)
    x = torch.randn(probe_batch, 1, img_size, img_size, device=device)
    t = torch.ones(probe_batch, device=device).long()
    labels = torch.full((probe_batch,), target_temp, device=device, dtype=torch.float)

    def _fwd():
        with torch.no_grad():
            model(x, t, labels)

    return _count_flops(_fwd) / probe_batch


def force_flops_per_eval(energy_fn, image_size: int = 64, probe_batch: int = 1,
                         device=None) -> float:
    """
    FLOPs for ONE analytical physics-force evaluation (`get_grad`), per sample,
    including its autograd backward.

    Underestimates the true cost — the double-well term is elementwise and so is
    not counted by FlopCounterMode (see module docstring).
    """
    device = device or torch.device("cpu")
    x = torch.randn(probe_batch, 1, image_size, image_size, device=device)
    return _count_flops(energy_fn.get_grad, x) / probe_batch


# ──────────────────────────────────────────────────────────────
# Closed forms
# ──────────────────────────────────────────────────────────────

def flops_pipc(M: int, K: int, f_net: float, f_force: float = 0.0) -> float:
    """Closed-form PIPC FLOPs per sample: NFE network evals + NFE force evals."""
    return nfe_pipc(M, K) * (f_net + f_force)


def flops_ddpm(noise_steps: int, f_net: float) -> float:
    """Closed-form DDPM FLOPs per sample."""
    return nfe_ddpm(noise_steps) * f_net


# ──────────────────────────────────────────────────────────────
# Empirical verification
# ──────────────────────────────────────────────────────────────

def verify_pipc_flops(score_model, energy_fn, sigmas, f_net: float, f_force: float,
                      device=None, image_size: int = 64, M: int = 5, K: int = 1,
                      lambda0: float = 0.2, target_temp: float = 2.27,
                      rel_tol: float = 1e-6) -> dict:
    """
    Wrap ONE short full PIPC trajectory in FlopCounterMode and compare the
    measured total against the closed form.

    Does NOT raise on mismatch — the caller decides (per project rule: report
    discrepancies rather than silently "fixing" them).

    Returns {M, K, flops_formula, flops_empirical, rel_err, match}.
    """
    from difuvia.sampling import pc_physics_guided_sampler

    def _traj():
        pc_physics_guided_sampler(
            score_model=score_model,
            energy_fn=energy_fn,
            sigmas=sigmas,
            target_temp=target_temp,
            image_size=image_size,
            batch_size=1,
            num_steps=M,
            corrector_steps=K,
            guidance_scale=lambda0,
            device=device,
        )

    empirical = _count_flops(_traj)
    formula = flops_pipc(M, K, f_net, f_force)
    rel_err = abs(empirical - formula) / formula if formula else float("nan")
    return {
        "M": M, "K": K,
        "flops_formula": formula,
        "flops_empirical": empirical,
        "rel_err": rel_err,
        "match": rel_err <= rel_tol,
    }


def print_flops_verification(info: dict) -> bool:
    """Pretty-print the trajectory verification. Returns True iff it matches."""
    print("\n" + "=" * 60)
    print("FLOPs VERIFICATION — one trajectory vs M*(K+1)*(f_net + f_force)")
    print("=" * 60)
    print(f"  probe config : M={info['M']}, K={info['K']}")
    print(f"  formula      : {info['flops_formula']:,.0f} FLOPs")
    print(f"  empirical    : {info['flops_empirical']:,.0f} FLOPs")
    print(f"  rel. error   : {info['rel_err']:.3e}")
    print(f"  match        : {info['match']}")
    print("=" * 60)
    if not info["match"]:
        print("  [STOP] Empirical FLOPs do not match the closed form.")
        print("         Reporting the discrepancy — NOT auto-fixing.")
    return info["match"]
