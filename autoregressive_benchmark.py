"""Colab-friendly evaluation of Chronos-2 on synthetic VAR yield levels.

Pass an already loaded Chronos2Pipeline to run_benchmark. No model download
or GPU allocation happens when this module is imported.
"""
from itertools import product

import numpy as np
import pandas as pd

from synthetic_yield_curve_var import simulate_yield_curve_var, var_coefficient_matrices


def parameter_grid():
    """One null control plus a crossed persistence/order/decay experiment."""
    configs = [dict(name="null", n=1, persistence=0.0, lag_decay=0.7)]
    for n, persistence, decay in product((1, 5, 20), (0.2, 0.5, 0.8), (0.3, 0.9)):
        if n == 1 and decay == 0.9:  # Decay has no effect with one lag.
            continue
        configs.append(dict(name=f"n{n}_p{persistence}_d{decay}", n=n,
                            persistence=persistence, lag_decay=decay))
    return configs


def oracle_forecast(context, coefficients, horizon):
    """Exact conditional mean of future levels, using only observed changes."""
    n = len(coefficients)
    changes = list(np.diff(np.asarray(context, dtype=float), axis=0)[-n:])
    if len(changes) < n:
        raise ValueError("Oracle requires at least n + 1 context observations")
    current = np.asarray(context[-1], dtype=float).copy()
    forecasts = []
    for _ in range(horizon):
        change = sum(coefficients[k] @ changes[-1-k] for k in range(n))
        changes.append(change)
        current = current + change
        forecasts.append(current.copy())
    return np.asarray(forecasts)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value)


def run_benchmark(pipeline, configs=None, *, seeds=(11, 22, 33),
                  context_length=4096, horizon=60, origins_per_seed=4,
                  origin_stride=None, curves_per_batch=4,
                  volatility_match="one_minute", factor_vols_bp=(70., 45., 30.),
                  noise_vol_bp=10., end="2026-01-01 00:00:00+00:00"):
    """Return metrics by config/seed/origin/tenor/exact forecast lead.

    Each curve is an independent multivariate task. cross_learning=False is
    essential: later rolling contexts must not inform earlier forecasts.
    Forecast windows do not overlap by default; contexts still overlap, so
    origins and tenors must not be treated as independent replications.
    Use independent seeds to assess variability. Levels are the model target.
    """
    configs = parameter_grid() if configs is None else list(configs)
    seeds = tuple(seeds)
    for name, value in dict(context_length=context_length, horizon=horizon,
                            origins_per_seed=origins_per_seed,
                            curves_per_batch=curves_per_batch).items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    origin_stride = horizon if origin_stride is None else origin_stride
    if not isinstance(origin_stride, int) or origin_stride < 1:
        raise ValueError("origin_stride must be a positive integer")
    if not configs or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide configurations and distinct seeds")
    if len({c['name'] for c in configs}) != len(configs):
        raise ValueError("Configuration names must be unique")
    if context_length <= max(c['n'] for c in configs):
        raise ValueError("context_length must exceed every VAR order")
    quantiles = [0.1, 0.5, 0.9]
    rows = context_length + horizon + (origins_per_seed - 1) * origin_stride
    records = []
    common = dict(factor_vols_bp=factor_vols_bp, noise_vol_bp=noise_vol_bp,
                  weekdays_only=False)
    for number, config in enumerate(configs, 1):
        params = {key: config[key] for key in ('n', 'persistence', 'lag_decay')}
        matrices = var_coefficient_matrices(**params, **common)
        for seed in seeds:
            frame = simulate_yield_curve_var(
                f"{rows-1}min", **params, **common, end=end, seed=seed,
                volatility_match=volatility_match)
            values = frame.to_numpy()
            origins = [context_length + i * origin_stride for i in range(origins_per_seed)]
            for start in range(0, len(origins), curves_per_batch):
                batch_origins = origins[start:start + curves_per_batch]
                contexts = [values[o-context_length:o] for o in batch_origins]
                inputs = np.stack([c.T for c in contexts]).astype(np.float32)
                q_outputs, _ = pipeline.predict_quantiles(
                    inputs, prediction_length=horizon, quantile_levels=quantiles,
                    context_length=context_length, batch_size=10 * curves_per_batch,
                    cross_learning=False)
                if len(q_outputs) != len(contexts):
                    raise ValueError("Unexpected number of Chronos forecast tasks")
                for origin, context, output in zip(batch_origins, contexts, q_outputs):
                    q = _numpy(output).transpose(1, 0, 2)
                    if q.shape != (horizon, 10, 3) or not np.isfinite(q).all():
                        raise ValueError(f"Invalid Chronos quantiles: {q.shape}")
                    actual = values[origin:origin+horizon]
                    predictions = {
                        'chronos': q[:, :, 1],
                        'naive': np.broadcast_to(context[-1], actual.shape),
                        'oracle': oracle_forecast(context, matrices, horizon),
                    }
                    base = dict(config=config['name'], **params, seed=seed,
                                origin=origin, volatility_match=volatility_match,
                                context_length=context_length)
                    for model, predicted in predictions.items():
                        error = (predicted - actual) * 1e4
                        result = pd.DataFrame({
                            **base, 'model': model,
                            'lead': np.repeat(np.arange(1, horizon+1), 10),
                            'tenor': np.tile(frame.columns, horizon),
                            'squared_error_bp2': (error**2).ravel(),
                            'absolute_error_bp': np.abs(error).ravel(),
                        })
                        if model == 'chronos':
                            residual = (actual[:, :, None] - q) * 1e4
                            loss = np.maximum(np.asarray(quantiles)*residual,
                                              (np.asarray(quantiles)-1)*residual)
                            result['pinball_bp'] = loss.mean(axis=-1).ravel()
                            result['coverage_80'] = ((actual >= q[:, :, 0]) &
                                                      (actual <= q[:, :, 2])).ravel()
                            result['width_80_bp'] = ((q[:, :, 2]-q[:, :, 0])*1e4).ravel()
                        records.append(result)
        print(f"[{number}/{len(configs)}] completed {config['name']}", flush=True)
    return pd.concat(records, ignore_index=True)


def summarize(results, by=('config', 'lead')):
    """Pool squared errors before taking roots; positive MSE skill is better."""
    by = list(by)
    metrics = results.groupby(by + ['model'], observed=True).agg(
        mse_bp2=('squared_error_bp2', 'mean'), mae_bp=('absolute_error_bp', 'mean'),
        pinball_bp=('pinball_bp', 'mean'), coverage_80=('coverage_80', 'mean'),
        width_80_bp=('width_80_bp', 'mean')).reset_index()
    metrics['rmse_bp'] = np.sqrt(metrics.mse_bp2)
    baseline = metrics.loc[metrics.model == 'naive', by + ['mse_bp2']].rename(
        columns={'mse_bp2': 'naive_mse_bp2'})
    metrics = metrics.merge(baseline, on=by, validate='many_to_one')
    metrics['mse_skill_vs_naive'] = 1 - metrics.mse_bp2 / metrics.naive_mse_bp2.replace(0, np.nan)
    return metrics
