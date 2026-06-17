"""Tests for the JumpDiffusionEngine public API.

All stochastic tests use a fixed seed (42) or rely on statistical invariants
that hold with very high probability, to avoid brittle flakiness.
"""
import numpy as np
import pytest

from jump_diffusion_engine import JumpDiffusionEngine


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Standard engine with a constant lambda and deterministic seed."""
    return JumpDiffusionEngine(
        lambda_func=lambda t: 0.5,
        sigma=0.3,
        jump_rate=0.0,   # no jumps — keeps tests deterministic
        dt=0.01,
        seed=42,
        k=0.8,
        g=0.5,
        K=2.0,
    )


def test_public_import_uses_package_layout():
    """The public import should resolve through the dedicated package."""
    assert JumpDiffusionEngine.__module__ == "jump_diffusion_engine.engine"


# ---------------------------------------------------------------------------
# find_fixed_points
# ---------------------------------------------------------------------------

class TestFindFixedPoints:
    def test_returns_at_least_one_stable_point(self, engine):
        fps = engine.find_fixed_points(lambda_val=0.5)
        assert len(fps) > 0, "Expected at least one fixed point"

    def test_stable_points_have_positive_f_prime(self, engine):
        fps = engine.find_fixed_points(lambda_val=0.5)
        stable = [fp for fp in fps if fp['stable']]
        assert len(stable) > 0, "Expected at least one stable fixed point"
        for fp in stable:
            assert fp['f_prime'] > 0, "Stable fixed point must have f' > 0"

    def test_fixed_point_satisfies_f_equals_lambda(self, engine):
        lambda_val = 0.5
        fps = engine.find_fixed_points(lambda_val=lambda_val)
        for fp in fps:
            residual = abs(engine.f_func(fp['x_star']) - lambda_val)
            assert residual < 1e-4, f"Fixed point residual too large: {residual}"

    def test_result_structure(self, engine):
        fps = engine.find_fixed_points(lambda_val=0.5)
        for fp in fps:
            assert 'x_star' in fp
            assert 'stable' in fp
            assert 'f_prime' in fp
            assert 'lambda_val' in fp

    def test_no_fixed_points_returns_empty(self, engine):
        # With a very large lambda_val that exceeds the sink's maximum,
        # there should be no fixed points.
        fps = engine.find_fixed_points(lambda_val=1e6)
        assert fps == []

    def test_different_lambda_shifts_fixed_point(self, engine):
        fps_low = engine.find_fixed_points(lambda_val=0.1)
        fps_high = engine.find_fixed_points(lambda_val=0.9)
        x_low = sorted(fp['x_star'] for fp in fps_low if fp['stable'])
        x_high = sorted(fp['x_star'] for fp in fps_high if fp['stable'])
        # Higher lambda → fixed point shifts to higher x
        if x_low and x_high:
            assert x_high[0] > x_low[0]


# ---------------------------------------------------------------------------
# escape_probability
# ---------------------------------------------------------------------------

class TestEscapeProbability:
    def test_returns_value_in_unit_interval(self, engine):
        p = engine.escape_probability(threshold=5.0, t_max=10.0, n_trials=20)
        assert 0.0 <= p <= 1.0

    def test_large_threshold_gives_low_escape(self, engine):
        # With a very large threshold, very few (if any) trials should escape.
        p = engine.escape_probability(threshold=50.0, t_max=5.0, n_trials=30)
        assert p < 0.5, f"Expected low escape probability with large threshold, got {p}"

    def test_tiny_threshold_gives_high_escape(self, engine):
        # With a near-zero threshold, almost every trial should "escape".
        p = engine.escape_probability(threshold=1e-6, t_max=5.0, n_trials=30)
        assert p > 0.5, f"Expected high escape probability with tiny threshold, got {p}"

    def test_explicit_x0_and_x_star(self, engine):
        fps = engine.find_fixed_points(0.5)
        x_star = fps[0]['x_star'] if fps else 0.0
        p = engine.escape_probability(
            threshold=5.0, t_max=5.0, x0=x_star, x_star=x_star, n_trials=20
        )
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# stationary_density
# ---------------------------------------------------------------------------

class TestStationaryDensity:
    def test_returns_arrays_of_matching_length(self, engine):
        x, p = engine.stationary_density(lambda_val=0.5)
        assert len(x) == len(p)
        assert len(x) > 0

    def test_density_is_normalised(self, engine):
        x, p = engine.stationary_density(lambda_val=0.5)
        # Numerical integral should be approximately 1.
        integral = (np.trapezoid(p, x) if hasattr(np, 'trapezoid') else
                    np.trapz(p, x))
        assert abs(integral - 1.0) < 0.05, f"Density not normalised: integral={integral}"

    def test_density_is_non_negative(self, engine):
        x, p = engine.stationary_density(lambda_val=0.5)
        assert np.all(p >= 0), "Density contains negative values"

    def test_peak_near_stable_fixed_point(self, engine):
        lambda_val = 0.5
        fps = engine.find_fixed_points(lambda_val)
        stable = [fp for fp in fps if fp['stable']]
        if not stable:
            pytest.skip("No stable fixed point to compare against")
        x_star = stable[0]['x_star']
        x, p = engine.stationary_density(lambda_val=lambda_val)
        peak_x = x[np.argmax(p)]
        assert abs(peak_x - x_star) < 1.5, (
            f"Density peak ({peak_x:.3f}) far from stable equilibrium ({x_star:.3f})"
        )


# ---------------------------------------------------------------------------
# seat_and_release
# ---------------------------------------------------------------------------

class TestSeatAndRelease:
    def test_returns_expected_keys(self, engine):
        result = engine.seat_and_release(t_max=5.0, x0=3.0, lambda_val=0.5)
        for key in ('t', 'x', 'control', 'boundary', 'x_star', 'settle_tol',
                    'release_idx', 'released'):
            assert key in result, f"Missing key: {key}"

    def test_trajectory_length_matches_time(self, engine):
        t_max = 5.0
        result = engine.seat_and_release(t_max=t_max, x0=3.0, lambda_val=0.5)
        expected_steps = int(t_max / engine.dt) + 1
        assert len(result['t']) == expected_steps
        assert len(result['x']) == expected_steps

    def test_control_is_zero_after_release(self, engine):
        result = engine.seat_and_release(t_max=5.0, x0=3.0, lambda_val=0.5)
        if result['released'] and result['release_idx'] is not None:
            post_control = result['control'][result['release_idx']:]
            assert np.allclose(post_control, 0.0), "Control was non-zero after release"

    def test_raises_without_stable_bowl(self):
        """seat_and_release should raise when no stable bowl exists."""
        eng = JumpDiffusionEngine(
            lambda_func=lambda t: 1e6,  # extreme lambda → no fixed points
            sigma=0.3, jump_rate=0.0, dt=0.01, seed=42,
        )
        with pytest.raises(ValueError, match="No stable bowl"):
            eng.seat_and_release(t_max=1.0)

    def test_trajectory_stays_bounded(self, engine):
        # After seating, x should not wander to extreme values.
        result = engine.seat_and_release(t_max=10.0, x0=1.0, lambda_val=0.5,
                                         dwell=50)
        assert np.all(np.abs(result['x']) < 30), "Trajectory left reasonable bounds"


# ---------------------------------------------------------------------------
# simulate (smoke / integration test)
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_single_realization_structure(self, engine):
        results = engine.simulate(t_max=1.0, x0=0.0, n_realizations=1)
        assert len(results) == 1
        r = results[0]
        assert 't' in r and 'x' in r
        assert len(r['t']) == len(r['x'])

    def test_multiple_realizations(self, engine):
        n = 5
        results = engine.simulate(t_max=1.0, x0=0.0, n_realizations=n)
        assert len(results) == n

    def test_seeded_reproducibility(self):
        """Same seed → identical trajectory."""
        def lf(t):
            return 0.5

        eng1 = JumpDiffusionEngine(lf, sigma=0.3, jump_rate=0.1, dt=0.01, seed=7)
        eng2 = JumpDiffusionEngine(lf, sigma=0.3, jump_rate=0.1, dt=0.01, seed=7)
        r1 = eng1.simulate(t_max=1.0, x0=0.0, n_realizations=1)[0]
        r2 = eng2.simulate(t_max=1.0, x0=0.0, n_realizations=1)[0]
        np.testing.assert_array_equal(r1['x'], r2['x'])

    def test_trajectory_mean_reverting(self, engine):
        """With no jumps and strong mean reversion, x should stay bounded."""
        results = engine.simulate(t_max=20.0, x0=5.0, n_realizations=3,
                                  record_energy=False)
        for r in results:
            assert np.all(np.abs(r['x']) < 50), "Trajectory diverged unexpectedly"

    def test_energy_recorded_when_requested(self, engine):
        results = engine.simulate(t_max=1.0, x0=0.0, n_realizations=1,
                                  record_energy=True)
        assert 'energy' in results[0]
        assert len(results[0]['energy']) == len(results[0]['x'])

    def test_energy_not_recorded_when_skipped(self, engine):
        results = engine.simulate(t_max=1.0, x0=0.0, n_realizations=1,
                                  record_energy=False)
        assert 'energy' not in results[0]


# ---------------------------------------------------------------------------
# plot_trajectories (non-display smoke test)
# ---------------------------------------------------------------------------

class TestPlotTrajectories:
    def test_returns_figure(self, engine, monkeypatch):
        import matplotlib.pyplot as plt
        # Suppress display in CI
        monkeypatch.setattr(plt, "show", lambda: None)
        results = engine.simulate(t_max=1.0, x0=0.0, n_realizations=2,
                                  record_energy=True)
        fig = engine.plot_trajectories(results, show_energy=True)
        assert fig is not None

    def test_accepts_single_realization(self, engine, monkeypatch):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        results = engine.simulate(t_max=1.0, x0=0.0, n_realizations=1,
                                  record_energy=False)
        fig = engine.plot_trajectories(results)
        assert fig is not None
