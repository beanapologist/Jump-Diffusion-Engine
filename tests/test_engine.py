"""Tests for the JumpDiffusionEngine public API.

All stochastic tests use a fixed seed (42) or rely on statistical invariants
that hold with very high probability, to avoid brittle flakiness.
"""
import numpy as np
import pytest

from jump_diffusion_engine import JumpDiffusionEngine, reduce_ring, ring_generator


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
# jump_operator
# ---------------------------------------------------------------------------

class TestJumpOperator:
    """
    Rigorous tests for 𝒥f(x) = λ ∫ [f(x+z) − f(x)] ν(dz|x).

    All MC tests use seed=0 and n_samples=100_000 for tight tolerances.
    """

    # --- helpers shared across tests ---

    def _engine_with_jumps(self, jump_rate=0.5):
        """Engine with Gaussian jump kernel N(0,1), fixed seed."""
        return JumpDiffusionEngine(
            lambda_func=lambda t: 0.5,
            sigma=0.3,
            jump_rate=jump_rate,
            dt=0.01,
            seed=42,
        )

    N = 100_000   # MC samples — enough for 3-decimal accuracy on smooth f

    # 1. Annihilates constants -------------------------------------------------

    def test_constant_function_gives_zero(self):
        eng = self._engine_with_jumps()
        f = lambda x: 7.0
        for x in [-2.0, 0.0, 1.5, 3.0]:
            result = eng.jump_operator(f, x, n_samples=self.N, seed=0)
            assert abs(result) < 1e-10, (
                f"Jf should be 0 for constant f, got {result} at x={x}"
            )

    # 2. Zero when jump_rate = 0 -----------------------------------------------

    def test_zero_jump_rate_gives_zero(self):
        eng = self._engine_with_jumps(jump_rate=0.0)
        f = lambda x: x**2 + np.sin(x)
        for x in [-1.0, 0.0, 2.5]:
            result = eng.jump_operator(f, x, n_samples=self.N, seed=0)
            assert result == 0.0, (
                f"Jf must be exactly 0 when jump_rate=0, got {result}"
            )

    # 3. Linearity: 𝒥(αf + βg) = α𝒥f + β𝒥g ------------------------------------

    def test_linearity_in_f(self):
        eng = self._engine_with_jumps()
        f = lambda x: x**2
        g = lambda x: np.cos(x)
        alpha, beta = 3.0, -2.0
        h = lambda x: alpha * f(x) + beta * g(x)

        for x in [0.5, 1.0, 2.0]:
            Jf = eng.jump_operator(f, x, n_samples=self.N, seed=0)
            Jg = eng.jump_operator(g, x, n_samples=self.N, seed=0)
            Jh = eng.jump_operator(h, x, n_samples=self.N, seed=0)
            expected = alpha * Jf + beta * Jg
            assert abs(Jh - expected) < 0.02, (
                f"Linearity failed at x={x}: J(αf+βg)={Jh:.6f}, "
                f"αJf+βJg={expected:.6f}"
            )

    # 4. Quadratic identity (analytical): f=x², N(0,1) kernel → 𝒥f(x) = λ -----

    def test_quadratic_identity(self):
        """
        For f(x)=x² and Z~N(0,1):
          𝒥f(x) = λ · E[(x+Z)² − x²]
                = λ · E[2xZ + Z²]
                = λ · (2x · 0 + 1) = λ
        Result is independent of x.
        """
        lam = 0.5
        eng = self._engine_with_jumps(jump_rate=lam)
        f = lambda x: x**2
        for x in [-3.0, 0.0, 1.0, 4.0]:
            result = eng.jump_operator(f, x, n_samples=self.N, seed=0)
            assert abs(result - lam) < 0.02, (
                f"Quadratic identity: expected {lam}, got {result:.5f} at x={x}"
            )

    # 5. Linear f has zero jump operator under symmetric kernel ----------------

    def test_linear_function_zero_under_symmetric_kernel(self):
        """
        For f(x)=x and Z~N(0,1) (symmetric, mean 0):
          𝒥f(x) = λ · E[(x+Z) − x] = λ · E[Z] = 0
        """
        eng = self._engine_with_jumps()
        f = lambda x: x
        for x in [-1.0, 0.0, 2.0]:
            result = eng.jump_operator(f, x, n_samples=self.N, seed=0)
            assert abs(result) < 0.02, (
                f"Linear f under symmetric kernel: expected 0, got {result:.5f} at x={x}"
            )

    # 6. Flat obstruction: 𝒥f(0⁺) ≠ 0 even though f'(0)=0 (all orders) --------

    def test_flat_obstruction_nonzero(self):
        """
        f(x) = e^{-1/x}  (x>0),  0 otherwise — a flat obstruction.

        All derivatives vanish at x=0 → drift and diffusion terms go to zero.
        Yet 𝒥f(0⁺) > 0 because the jump kernel puts weight on x>0 where f>0.
        This is the fundamental non-locality of 𝒥.
        """
        eng = self._engine_with_jumps(jump_rate=1.0)

        def flat_f(x):
            return float(np.exp(-1.0 / x)) if x > 1e-10 else 0.0

        # At x→0⁺ with Gaussian jumps, about half the destinations are positive.
        # E[f(0+Z)] = E[e^{-1/Z} · 1_{Z>0}] > 0  → 𝒥f(0⁺) > 0.
        result = eng.jump_operator(flat_f, 0.0, n_samples=self.N, seed=0)
        assert result > 0.0, (
            f"Flat obstruction: 𝒥f(0⁺) must be > 0, got {result:.6f}"
        )

        # Confirm that local terms (drift b·f' and diffusion ½σ²f'') are 0 at x=0.
        # All derivatives of e^{-1/x} vanish as x→0⁺.
        h = 1e-6
        f_prime_approx = (flat_f(h) - flat_f(0.0)) / h
        assert abs(f_prime_approx) < 1e-100, (
            f"f'(0⁺) should be 0 (flat point), got {f_prime_approx}"
        )

    # 7. Crossing condition: cliff, not slope ----------------------------------

    def test_crossing_condition_cliff_not_slope(self):
        """
        Demonstrates the crossing condition J > J_c = Δ_edge − Δ*.

        Uses a fixed-size jump distribution: all jumps equal J (deterministic).
        Below J_c:  escape_probability stays low (≤ 0.15).
        Above J_c:  escape_probability jumps sharply (≥ 0.50).

        The cliff shape — not a gradual slope — reflects that the nonlocal
        operator either reaches over the basin wall or it does not.

        Uses a bistable parameter regime (k=0.5, g=2.0, K=1.0) so the basin
        walls are well-defined unstable fixed points.
        """
        bistable_kwargs = dict(k=0.5, g=2.0, K=1.0)
        eng_base = JumpDiffusionEngine(
            lambda_func=lambda t: 0.5,
            sigma=0.05,
            jump_rate=0.2,
            dt=0.01,
            seed=42,
            **bistable_kwargs,
        )
        boundary = eng_base.identify_boundary(lambda_val=0.5)
        x_star = boundary['x_star']
        half_width = boundary['half_width']

        if half_width is None or x_star is None:
            pytest.skip("No well-defined basin found for crossing test")

        J_c = half_width   # minimum jump to clear the basin edge

        results = {}
        for scale in [0.4, 1.8]:
            J = scale * J_c

            def _make_jump(size):
                return lambda: size

            eng = JumpDiffusionEngine(
                lambda_func=lambda t: 0.5,
                sigma=0.05,
                jump_rate=0.2,
                jump_size_dist=_make_jump(J),   # deterministic jump size
                dt=0.01,
                seed=42,
                **bistable_kwargs,
            )
            results[scale] = eng.escape_probability(
                threshold=J_c * 0.9,
                t_max=20.0,
                x0=x_star,
                x_star=x_star,
                n_trials=80,
            )

        p_below = results[0.4]
        p_above = results[1.8]
        assert p_below < 0.15, (
            f"Crossing condition (below J_c): expected P < 0.15, got {p_below:.3f}"
        )
        assert p_above > 0.50, (
            f"Crossing condition (above J_c): expected P > 0.50, got {p_above:.3f}"
        )
        assert p_above > p_below + 0.35, (
            f"Cliff: gap P_above−P_below should be > 0.35, got {p_above - p_below:.3f}"
        )

    # 8. Scaling with jump_rate ------------------------------------------------

    def test_scales_linearly_with_jump_rate(self):
        """𝒥f(x) is proportional to λ (rate doubles → operator doubles)."""
        f = lambda x: x**3
        x = 1.5
        lam1, lam2 = 0.5, 1.0
        eng1 = self._engine_with_jumps(jump_rate=lam1)
        eng2 = self._engine_with_jumps(jump_rate=lam2)
        J1 = eng1.jump_operator(f, x, n_samples=self.N, seed=0)
        J2 = eng2.jump_operator(f, x, n_samples=self.N, seed=0)
        # Both use the same seed → same z draws → ratio should equal lam2/lam1
        ratio = J2 / J1 if abs(J1) > 1e-12 else None
        if ratio is not None:
            assert abs(ratio - (lam2 / lam1)) < 0.05, (
                f"Expected ratio {lam2/lam1}, got {ratio:.4f}"
            )

    # 9. Custom kernel is respected -------------------------------------------

    def test_custom_kernel_respected(self):
        """With a custom jump_size_dist, 𝒥 uses that distribution."""
        # Positive-only jumps: Z ~ Exp(1).  For f(x)=x:
        #   𝒥f(x) = λ · E[Z] = λ · 1 = λ
        lam = 0.6
        rng_inner = np.random.default_rng(99)

        def exp_kernel():
            return rng_inner.exponential(1.0)

        eng = JumpDiffusionEngine(
            lambda_func=lambda t: 0.5,
            sigma=0.3,
            jump_rate=lam,
            jump_size_dist=exp_kernel,
            dt=0.01,
            seed=42,
        )
        f = lambda x: x
        x = 2.0
        result = eng.jump_operator(f, x, n_samples=self.N, seed=None)
        # E[Z] = 1 for Exp(1) → 𝒥f(x)=λ
        assert abs(result - lam) < 0.05, (
            f"Custom Exp(1) kernel: expected 𝒥f ≈ {lam}, got {result:.4f}"
        )


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


# ---------------------------------------------------------------------------
# reduce_ring
# ---------------------------------------------------------------------------

class TestReduceRing:
    def test_gcd_uses_only_active_channels(self):
        out = reduce_ring(12, {2: 1.0, 3: -5.0, 6: 0.0})
        assert out['g'] == 2
        assert out['N_core'] == 6
        assert out['channels_core'] == {1: 1.0}

    def test_effectively_empty_channels_reduce_to_trivial_core(self):
        out = reduce_ring(12, {1: 0.0, -1: -2.0, 12: 4.0})
        assert out['g'] == 12
        assert out['N_core'] == 1
        assert out['channels_core'] == {}
        np.testing.assert_allclose(out['L_core'], np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# entropy production
# ---------------------------------------------------------------------------

class TestEntropyProduction:
    def test_entropy_production_equals_current_times_affinity(self):
        N = 7
        rate_fwd = 1.7
        rate_bwd = 0.4
        L = ring_generator(N, {1: rate_fwd, -1: rate_bwd})
        p = np.full(N, 1.0 / N)

        ep = 0.0
        for i in range(N):
            j = (i + 1) % N
            f_ij = L[j, i] * p[i]
            f_ji = L[i, j] * p[j]
            assert f_ij > 0.0 and f_ji > 0.0
            J = f_ij - f_ji
            A = np.log(f_ij / f_ji)
            ep += J * A

        current = (rate_fwd - rate_bwd) / N
        affinity = N * np.log(rate_fwd / rate_bwd)
        np.testing.assert_allclose(ep, current * affinity, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# markov_generator
# ---------------------------------------------------------------------------

class TestMarkovGenerator:
    """Tests for the safer column-generator approach to building L."""

    N = 50   # small grid for fast tests

    def test_returns_expected_keys(self, engine):
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        for key in ('L', 'x', 'dx', 'rate_up', 'rate_down'):
            assert key in out, f"Missing key: {key}"

    def test_matrix_shape(self, engine):
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        L = out['L']
        assert L.shape == (self.N, self.N), f"Expected ({self.N},{self.N}), got {L.shape}"

    def test_column_sums_are_zero(self, engine):
        """Safer column-generator: each column must sum to exactly zero."""
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        col_sums = out['L'].sum(axis=0)
        np.testing.assert_allclose(
            col_sums, 0.0, atol=1e-12,
            err_msg="Column sums of generator matrix are not zero"
        )

    def test_diagonal_equals_negative_rate_sum(self, engine):
        """L[n, n] == -(rate_up[n] + rate_down[n]) for every n."""
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        L = out['L']
        diag = np.diag(L)
        expected = -(out['rate_up'] + out['rate_down'])
        np.testing.assert_allclose(
            diag, expected, atol=1e-14,
            err_msg="Diagonal entries do not equal -(rate_up + rate_down)"
        )

    def test_off_diagonal_non_negative(self, engine):
        """All off-diagonal entries must be ≥ 0 (valid transition rates)."""
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        L = out['L']
        off_diag = L - np.diag(np.diag(L))
        assert np.all(off_diag >= -1e-14), "Off-diagonal entries contain negative values"

    def test_tridiagonal_structure(self, engine):
        """Generator matrix for a diffusion should be tridiagonal."""
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        L = out['L']
        # All entries more than one step from the diagonal must be zero.
        for i in range(self.N):
            for j in range(self.N):
                if abs(i - j) > 1:
                    assert L[i, j] == 0.0, (
                        f"Non-zero entry at L[{i},{j}]={L[i,j]} (not tridiagonal)"
                    )

    def test_boundary_rates_are_zero(self, engine):
        """Absorbing boundaries: rate_up[-1] and rate_down[0] must be zero."""
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        assert out['rate_up'][-1] == 0.0, "rate_up at last node should be zero"
        assert out['rate_down'][0] == 0.0, "rate_down at first node should be zero"

    def test_grid_length_matches_n_points(self, engine):
        n = 30
        out = engine.markov_generator(lambda_val=0.5, n_points=n)
        assert len(out['x']) == n
        assert len(out['rate_up']) == n
        assert len(out['rate_down']) == n

    def test_dx_matches_grid(self, engine):
        out = engine.markov_generator(lambda_val=0.5, n_points=self.N)
        expected_dx = (out['x'][-1] - out['x'][0]) / (self.N - 1)
        np.testing.assert_allclose(out['dx'], expected_dx, atol=1e-12)
