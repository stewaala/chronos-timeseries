import unittest
import numpy as np
from autoregressive_benchmark import oracle_forecast, parameter_grid, run_benchmark, summarize
from synthetic_yield_curve_var import simulate_yield_curve_var


class BenchmarkTests(unittest.TestCase):
    def test_oracle_ar1_closed_form(self):
        context = np.array([[1., 2.], [1.2, 2.4]])
        forecast = oracle_forecast(context, np.array([np.eye(2)*0.5]), 4)
        expected = context[-1] + np.array([1-0.5**h for h in range(1, 5)])[:, None] * np.array([0.2, 0.4])
        np.testing.assert_allclose(forecast, expected)

    def test_oracle_ar2_lag_order(self):
        forecast = oracle_forecast(np.array([[0.], [1.], [3.]]),
                                   np.array([[[0.4]], [[0.2]]]), 2)
        np.testing.assert_allclose(forecast[:, 0], [4., 4.8])

    def test_evaluation_and_no_future_input(self):
        class FakePipeline:
            def __init__(self):
                self.contexts = []
            def predict_quantiles(self, inputs, **kwargs):
                assert kwargs['cross_learning'] is False
                self.contexts.extend(inputs)
                point = np.repeat(inputs[:, :, -1:], kwargs['prediction_length'], axis=2)
                q = np.stack([point-0.001, point, point+0.001], axis=-1)
                return list(q), list(point)
        pipeline = FakePipeline()
        config = dict(name='null', n=1, persistence=0., lag_decay=0.7)
        result = run_benchmark(pipeline, [config], seeds=[11], context_length=8,
                               horizon=3, origins_per_seed=2, curves_per_batch=2)
        full = simulate_yield_curve_var('13min', n=1, persistence=0., seed=11,
                                       end='2026-01-01 00:00:00+00:00')
        for supplied, origin in zip(pipeline.contexts, [8, 11]):
            np.testing.assert_array_equal(supplied, full.iloc[origin-8:origin].to_numpy().T.astype(np.float32))
        self.assertEqual(len(result), 2*3*10*3)
        summary = summarize(result)
        np.testing.assert_allclose(summary.query("model == 'oracle'").mse_skill_vs_naive, 0.)
        np.testing.assert_allclose(summary.query("model == 'chronos'").mse_skill_vs_naive, 0., atol=0.003)
        self.assertTrue((summary.query("model == 'chronos'").coverage_80 == 1).all())

    def test_grid(self):
        self.assertEqual(len(parameter_grid()), 16)


if __name__ == '__main__':
    unittest.main()
