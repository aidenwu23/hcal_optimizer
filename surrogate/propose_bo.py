#!/usr/bin/env python3
"""
propose_bo.py 

Reads a BO spec (bounds/constraints + sweep_base), samples candidate geometries,
scores them using a trained LightGBM surrogate (point predictions), and writes
a sweep YAML compatible with hcal_optimizer/geometries/sweep_geometries.py.

- Geometry knobs may be split between:
    * bounds        -> optimized/proposed variables
    * fixed_features -> fixed surrogate inputs
    * sweep_base.constants -> fixed geometry written into output YAML
- Candidate geometries are scored across an energy_spectrum using exact
  kinetic-energy points and user-provided weights.
- Surrogate expects features:
    kinetic_energy_GeV,
    seg1_layers, seg2_layers, seg3_layers,
    t_absorber_seg1/2/3, t_scin_seg1/2/3
  
- Variants written to YAML contain ONLY valid geometry keys (no extra bookkeeping keys),
  so hcal_optimizer/geometries/generate_hcal.py won’t see unknown --set parameters.

Example:
python3 surrogate/propose_bo.py \
  --model surrogate/model/lgbm_surrogate_NK_0.joblib \
  --spec  geometries/sweeps/bo_spec.yaml \
  --out   geometries/sweeps/proposed/proposed_0_test.yaml \
  --pool  20000 \
  --k     5 \
  --seed  0
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import yaml

# Check if PyTorch is available for Sobol sampling
_TORCH_OK = False
try:
    import torch  # type: ignore
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

SURROGATE_FEATURES = [
    "kinetic_energy_GeV",
    "seg1_layers",
    "seg2_layers",
    "seg3_layers",
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
]

SURROGATE_TARGETS = [
    "kaon0L_efficiency",
    "kaon0L_energy_resolution",
    "neutron_efficiency",
    "neutron_energy_resolution",
]

GEOM_VARS = [
    "seg1_layers",
    "seg2_layers",
    "seg3_layers",
    "t_absorber_seg1",
    "t_absorber_seg2",
    "t_absorber_seg3",
    "t_scin_seg1",
    "t_scin_seg2",
    "t_scin_seg3",
]


# -------------------------------------------------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------------------------------------------------
def sobol_u01(n: int, d: int, seed: int) -> np.ndarray:
    if _TORCH_OK:
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(n).cpu().numpy()
    rng = np.random.default_rng(seed)
    return rng.random((n, d))

def load_model_bundle(model_path: str) -> Tuple[Any, List[str], List[str]]:
    loaded_payload = joblib.load(model_path)
    if isinstance(loaded_payload, dict) and "model" in loaded_payload:
        model = loaded_payload["model"]
        feature_columns = list(loaded_payload.get("feature_columns", []))
        target_columns = list(loaded_payload.get("target_columns", []))
    else:
        model = loaded_payload
        feature_columns = SURROGATE_FEATURES
        target_columns = SURROGATE_TARGETS

    if not feature_columns:
        raise SystemExit("Loaded model bundle is missing feature_columns.")
    if not target_columns:
        raise SystemExit("Loaded model bundle is missing target_columns.")

    return model, feature_columns, target_columns


def parse_bounds(bounds_spec: Dict[str, Any], var_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    lows, highs, meta = [], [], []
    for name in var_names:
        if name not in bounds_spec:
            raise SystemExit(f"Missing bounds for '{name}' in bo_spec.yaml")
        
        spec = bounds_spec[name]
        if isinstance(spec, dict):
             low = float(spec["low"])
             high = float(spec["high"])
             meta.append(spec)
        else:
             low = float(spec[0])
             high = float(spec[1])
             meta.append({"type": "float"})
        if high <= low:
            raise SystemExit(f"Invalid bounds for '{name}': low={low} high={high}")
        
        lows.append(low)
        highs.append(high)

    return np.array(lows, dtype=float), np.array(highs, dtype=float), meta


def apply_discreteness(X: np.ndarray, var_names: List[str], meta: List[Dict[str, Any]]) -> np.ndarray:
     X2 = X.copy()
     for j, (name, m) in enumerate(zip(var_names, meta)):
         t = str(m.get("type", "float")).lower()
         step = float(m.get("step", 1.0))
         if t in ("int", "integer"):
             X2[:, j] = np.round(X2[:, j] / step) * step
             X2[:, j] = np.round(X2[:, j]).astype(int)
         elif t == "discrete":
             X2[:, j] = np.round(X2[:, j] / step) * step
         # else float, do nothing
     return X2 


def normalize01(X: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    denom = np.maximum(highs - lows, 1e-12)
    return (X - lows) / denom


def diverse_topk(X: np.ndarray, scores: np.ndarray, k: int, min_dist: float, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    idx_sorted = np.argsort(-scores)
    Xn = normalize01(X, lows, highs)
    chosen: List[int] = []
    for idx in idx_sorted:
        if len(chosen) >= k:
            break
        if min_dist <= 0 or not chosen:
            chosen.append(int(idx))
            continue
        d = np.linalg.norm(Xn[idx] - Xn[chosen], axis=1)
        if np.all(d >= min_dist):
            chosen.append(int(idx))

    # pad if needed
    if len(chosen) < k:
        for idx in idx_sorted:
            ii = int(idx)
            if ii not in chosen:
                chosen.append(ii)
            if len(chosen) >= k:
                break
    return np.array(chosen[:k], dtype=int)


# -------------------------------------------------------------------------------------------------
# Constraint + scoring helpers
# -------------------------------------------------------------------------------------------------
def safe_eval_expr(expr: str, local_vars: Dict[str, Any]) -> bool:
    return bool(eval(expr, {"__builtins__": {}}, local_vars))


def filter_design_constraints(X: np.ndarray, var_names: List[str], exprs: List[str]) -> np.ndarray:
    if not exprs:
        return np.ones((X.shape[0],), dtype=bool)
    ok = np.ones((X.shape[0],), dtype=bool)
    for i in range(X.shape[0]):
        loc = {var_names[j]: float(X[i, j]) for j in range(len(var_names))}
        # also expose ints nicely for layer vars
        for k, v in list(loc.items()):
            if "layers" in k:
                loc[k] = int(round(v))
        for e in exprs:
            try:
                if not safe_eval_expr(e, loc):
                    ok[i] = False
                    break
            except Exception:
                ok[i] = False
                break
    return ok


def filter_predicted_constraints(pred: Dict[str, np.ndarray], exprs: List[str]) -> np.ndarray:
    if not exprs:
        return np.ones((len(next(iter(pred.values()))),), dtype=bool)
    ok = np.ones((len(next(iter(pred.values()))),), dtype=bool)
    for i in range(ok.shape[0]):
        loc = {k: float(v[i]) for k, v in pred.items()}
        for e in exprs:
            try:
                if not safe_eval_expr(e, loc):
                    ok[i] = False
                    break
            except Exception:
                ok[i] = False
                break
    return ok


def score_candidates(pred: Dict[str, np.ndarray], scoring: Dict[str, Any]) -> np.ndarray:
    mode = str(scoring.get("mode", "metric")).lower()
    maximize = bool(scoring.get("maximize", True))

    if mode == "tradeoff":
        expr = scoring.get("expr")
        if not expr:
            raise SystemExit("scoring.mode=tradeoff requires scoring.expr")
        scores = np.zeros((len(next(iter(pred.values()))),), dtype=float)
        for i in range(scores.shape[0]):
            loc = {k: float(v[i]) for k, v in pred.items()}
            scores[i] = float(eval(expr, {"__builtins__": {}}, loc))
        if not maximize:
            scores = -scores
        return scores

    metric = str(scoring.get("metric", next(iter(pred.keys()))))
    if metric not in pred:
        raise SystemExit(f"Unknown scoring.metric '{metric}'. Available: {list(pred.keys())}")
    scores = pred[metric].astype(float).copy()
    if not maximize:
        scores = -scores
    return scores


def parse_energy_spectrum(spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    # Read the exact kinetic-energy points and their weights from the BO spec.
    energy_spectrum = spec.get("energy_spectrum", {}) or {}
    if not isinstance(energy_spectrum, dict):
        raise SystemExit("'energy_spectrum' must be a mapping if provided.")

    energy_values = energy_spectrum.get("kinetic_energy_GeV")
    weight_values = energy_spectrum.get("weights")

    if energy_values is None and weight_values is None:
        return np.array([1.0], dtype=float), np.array([1.0], dtype=float)
    if energy_values is None or weight_values is None:
        raise SystemExit("'energy_spectrum' must contain both 'kinetic_energy_GeV' and 'weights'.")
    if not isinstance(energy_values, list) or not isinstance(weight_values, list):
        raise SystemExit("'energy_spectrum.kinetic_energy_GeV' and 'energy_spectrum.weights' must be lists.")
    if not energy_values:
        raise SystemExit("'energy_spectrum.kinetic_energy_GeV' must not be empty.")
    if len(energy_values) != len(weight_values):
        raise SystemExit("'energy_spectrum.kinetic_energy_GeV' and 'energy_spectrum.weights' must have the same length.")

    try:
        kinetic_energies = np.array([float(value) for value in energy_values], dtype=float)
        weights = np.array([float(value) for value in weight_values], dtype=float)
    except Exception as exc:
        raise SystemExit(f"Invalid 'energy_spectrum' values: {exc}") from exc

    return kinetic_energies, weights


# -------------------------------------------------------------------------------------------------
# Mapping geometry -> surrogate features
# -------------------------------------------------------------------------------------------------
def geom_to_surrogate_features(
    X_geom: np.ndarray,
    geom_var_names: List[str],
    feature_columns: List[str],
    fixed_features: Dict[str, Any],
) -> np.ndarray:
    """
    Build the surrogate feature matrix in the saved model feature order.
    """
    n = X_geom.shape[0]
    Xs = np.zeros((n, len(feature_columns)), dtype=float)
    feat_idx = {f: i for i, f in enumerate(feature_columns)}
    geom_idx = {g: i for i, g in enumerate(geom_var_names)}

    # Fixed features
    for k, v in (fixed_features or {}).items():
        if k not in feat_idx:
            raise SystemExit(f"fixed_features contains unknown surrogate feature '{k}'")
        Xs[:, feat_idx[k]] = float(v)


    # Fill any proposed geometry vars that are also surrogate features
    for g in geom_var_names:
        if g in feat_idx:
            Xs[:, feat_idx[g]] = X_geom[:, geom_idx[g]].astype(float)

    return Xs


# -------------------------------------------------------------------------------------------------
# YAML emission
# -------------------------------------------------------------------------------------------------
def build_sweep_yaml(sweep_base: Dict[str, Any], X_geom: np.ndarray, geom_var_names: List[str]) -> Dict[str, Any]:
    """
    Output sweep YAML using sweep_base plus generated variants.

    Each variant gets:
      - tag: <tag_prefix><zero-padded index>
      - only proposed geometry keys from geom_var_names

    Fixed geometry values should live in sweep_base.constants.
    """
    out = dict(sweep_base)
    out["variants"] = []

    tag_prefix = str(out.get("tag_prefix", "var"))
    index_base = int(out.get("index_base", 0))

    for i in range(X_geom.shape[0]):
        v: Dict[str, Any] = {
            "tag": f"{tag_prefix}{index_base + i:03d}"
        }
        for j, key in enumerate(geom_var_names):
            val = X_geom[i, j]
            if "layers" in key:
                v[key] = int(round(float(val)))
            else:
                v[key] = round(float(val), 4)
        out["variants"].append(v)

    return out


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Propose next geometries (Option A) and write a sweep YAML.")
    ap.add_argument("--model", required=True, help="Path to trained surrogate .joblib bundle.")
    ap.add_argument("--spec", required=True, help="Path to bo_spec.yaml.")
    ap.add_argument("--out", required=True, help="Output sweep YAML path.")
    ap.add_argument("--pool", type=int, default=20000, help="Number of random/Sobol candidates to sample.")
    ap.add_argument("--k", type=int, default=32, help="Number of variants to output.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load spec.
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    sweep_base = spec.get("sweep_base")
    if not isinstance(sweep_base, dict):
        raise SystemExit("bo_spec.yaml must contain a 'sweep_base' mapping.")

    bounds_spec = spec.get("bounds")
    if not isinstance(bounds_spec, dict):
        raise SystemExit("bo_spec.yaml must contain a 'bounds' mapping.")

    fixed_features = spec.get("fixed_features", {}) or {}
    if not isinstance(fixed_features, dict):
        raise SystemExit("'fixed_features' must be a mapping if provided.")
    kinetic_energies, energy_weights = parse_energy_spectrum(spec)

    constraints = spec.get("constraints", {}) or {}
    design_exprs = constraints.get("design", []) or []
    pred_exprs = constraints.get("predicted", []) or []
    if not isinstance(design_exprs, list) or not isinstance(pred_exprs, list):
        raise SystemExit("'constraints.design' and 'constraints.predicted' must be lists of expressions.")

    scoring = spec.get("scoring", {}) or {}
    diversity = spec.get("diversity", {}) or {}
    min_dist = float(diversity.get("min_l2_norm", 0.05))
    model, feature_columns, target_columns = load_model_bundle(args.model)

    # Determine which geometry variables we propose.
    # Allow only a subset of GEOM_VARS to be optimized; the rest may come from fixed_features.
    geom_var_names = [k for k in GEOM_VARS if k in bounds_spec]

    unknown_bounds = [k for k in bounds_spec.keys() if k not in GEOM_VARS]
    if unknown_bounds:
        raise SystemExit(
            f"Unknown bounds keys not recognized as geometry variables: {unknown_bounds}"
        )

    missing_for_surrogate = [
        k for k in feature_columns
        if k not in geom_var_names and k not in fixed_features and k != "kinetic_energy_GeV"
    ]

    if missing_for_surrogate:
        raise SystemExit(
            "Some surrogate features are neither proposed in bounds nor provided in fixed_features: "
            f"{missing_for_surrogate}"
        )
    
    # Parse bounds.
    lows, highs, meta = parse_bounds(bounds_spec, geom_var_names)

    # Sample pool in geometry space.
    u = sobol_u01(args.pool, len(geom_var_names), seed=args.seed)
    X_geom = lows + (highs - lows) * u
    X_geom = apply_discreteness(X_geom, geom_var_names, meta)

    # Apply design constraints first (cheap).
    ok_design = filter_design_constraints(X_geom, geom_var_names, design_exprs)
    X_geom_d = X_geom[ok_design]
    if X_geom_d.shape[0] == 0:
        raise SystemExit("No candidates satisfy design constraints. Loosen constraints or increase bounds/pool.")

    import pandas as pd

    # Build one surrogate row per geometry-and-energy pair.
    feature_rows: List[Dict[str, float]] = []
    for geometry_values in X_geom_d:
        geometry_feature_values = {
            geom_var_names[j]: float(geometry_values[j]) for j in range(len(geom_var_names))
        }
        for kinetic_energy_gev in kinetic_energies:
            feature_row: Dict[str, float] = {}
            for feature_name in feature_columns:
                if feature_name == "kinetic_energy_GeV":
                    feature_row[feature_name] = float(kinetic_energy_gev)
                elif feature_name in geometry_feature_values:
                    feature_row[feature_name] = geometry_feature_values[feature_name]
                elif feature_name in fixed_features:
                    feature_row[feature_name] = float(fixed_features[feature_name])
                else:
                    raise SystemExit(
                        f"Cannot build surrogate feature rows because '{feature_name}' is missing."
                    )
            feature_rows.append(feature_row)

    Xs_df = pd.DataFrame(feature_rows, columns=feature_columns)

    Y = np.asarray(model.predict(Xs_df))

    n_candidates = X_geom_d.shape[0]
    n_energies = len(kinetic_energies)
    expected_rows = n_candidates * n_energies
    if Y.ndim != 2 or Y.shape[0] != expected_rows or Y.shape[1] != len(target_columns):
        raise SystemExit(f"Unexpected surrogate output shape {Y.shape}; expected (N,{len(target_columns)}).")

    pred_flat = {name: Y[:, i] for i, name in enumerate(target_columns)}

    # Every predicted constraint must hold at every evaluated energy for a candidate.
    ok_pred_flat = filter_predicted_constraints(pred_flat, pred_exprs)
    ok_pred = ok_pred_flat.reshape(n_candidates, n_energies).all(axis=1)

    energy_scores_flat = score_candidates(pred_flat, scoring)
    energy_scores = energy_scores_flat.reshape(n_candidates, n_energies)
    aggregated_scores = energy_scores @ energy_weights

    X_final = X_geom_d[ok_pred]
    scores_final = aggregated_scores[ok_pred]

    if X_final.shape[0] == 0:
        raise SystemExit("No candidates satisfy predicted constraints. Loosen constraints or increase pool.")

    # Score + choose top-k with diversity
    k = min(args.k, X_final.shape[0])

    chosen_idx = diverse_topk(
        X_final, scores_final, k=k, min_dist=min_dist,
        lows=lows, highs=highs
    )
    X_out = X_final[chosen_idx]

    # Emit sweep YAML
    sweep_yaml = build_sweep_yaml(sweep_base, X_out, geom_var_names)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(sweep_yaml, f, sort_keys=False)

    print(f"Wrote {args.out} with {len(sweep_yaml['variants'])} variants "
          f"(sampled={X_geom.shape[0]}, design_ok={X_geom_d.shape[0]}, feasible={X_final.shape[0]}).")


if __name__ == "__main__":
    main()
