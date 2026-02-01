import numpy as np
import pytest
from activation_functions import (
    sigmoid, swish, relu, glu, gelu, silu, swiglu,
    leaky_relu, elu, prelu, mish, geglu, telu
)


class TestSigmoid:
    def test_zero(self):
        assert sigmoid(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert sigmoid(100) == pytest.approx(1.0)

    def test_large_negative(self):
        assert sigmoid(-100) == pytest.approx(0.0)

    def test_symmetry(self):
        x = np.array([-2, -1, 0, 1, 2])
        assert np.allclose(sigmoid(x) + sigmoid(-x), 1.0)

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert sigmoid(x).shape == x.shape


class TestSwish:
    def test_zero(self):
        assert swish(0) == pytest.approx(0.0)

    def test_positive(self):
        # swish(x) = x * sigmoid(x), for large positive x ~ x
        assert swish(10) == pytest.approx(10.0, abs=0.01)

    def test_negative(self):
        # swish has a small negative dip
        assert swish(-1) < 0

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert swish(x).shape == x.shape


class TestReLU:
    def test_positive(self):
        assert relu(5) == 5

    def test_negative(self):
        assert relu(-5) == 0

    def test_zero(self):
        assert relu(0) == 0

    def test_array(self):
        x = np.array([-2, -1, 0, 1, 2])
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(relu(x), expected)

    def test_all_negative(self):
        x = np.array([-5, -3, -1])
        np.testing.assert_array_equal(relu(x), np.zeros(3))


class TestLeakyReLU:
    def test_positive(self):
        assert leaky_relu(5) == 5

    def test_negative_default(self):
        # default alpha=0.01
        assert leaky_relu(-5) == pytest.approx(-0.05)

    def test_negative_custom_alpha(self):
        assert leaky_relu(-5, alpha=0.1) == pytest.approx(-0.5)

    def test_zero(self):
        assert leaky_relu(0) == pytest.approx(0.0)

    def test_never_fully_zero_for_negative(self):
        x = np.array([-5, -3, -1])
        result = leaky_relu(x)
        assert np.all(result < 0)

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert leaky_relu(x).shape == x.shape


class TestELU:
    def test_positive(self):
        assert elu(5) == 5

    def test_zero(self):
        assert elu(0) == pytest.approx(0.0, abs=1e-10)

    def test_negative_approaches_minus_alpha(self):
        # for very negative x, elu -> -alpha
        assert elu(-100, alpha=1.0) == pytest.approx(-1.0)

    def test_custom_alpha(self):
        assert elu(-100, alpha=2.0) == pytest.approx(-2.0)

    def test_smooth_at_zero(self):
        # values near zero should be near zero
        assert elu(-0.001) == pytest.approx(-0.001, abs=0.01)

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert elu(x).shape == x.shape


class TestPReLU:
    def test_positive(self):
        assert prelu(5) == 5

    def test_negative_default(self):
        # default alpha=0.25
        assert prelu(-4) == pytest.approx(-1.0)

    def test_negative_custom_alpha(self):
        assert prelu(-4, alpha=0.5) == pytest.approx(-2.0)

    def test_zero(self):
        assert prelu(0) == pytest.approx(0.0)

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert prelu(x).shape == x.shape


class TestGLU:
    def test_zero(self):
        # glu(0) = 0 * sigmoid(0) = 0
        assert glu(0) == pytest.approx(0.0)

    def test_positive(self):
        # glu(x) = x * sigmoid(x), positive for positive x
        assert glu(2) > 0

    def test_negative(self):
        # glu(x) for negative x: x is negative, sigmoid(x) is positive -> negative
        assert glu(-2) < 0

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert glu(x).shape == x.shape


class TestGELU:
    def test_zero(self):
        assert gelu(0) == pytest.approx(0.0, abs=1e-6)

    def test_large_positive(self):
        # gelu(x) ~ x for large positive x
        assert gelu(5) == pytest.approx(5.0, abs=0.01)

    def test_large_negative(self):
        # gelu(x) ~ 0 for large negative x
        assert gelu(-5) == pytest.approx(0.0, abs=0.01)

    def test_small_negative_dip(self):
        # GELU has a slight negative dip around x ~ -0.5
        assert gelu(-0.5) < 0

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert gelu(x).shape == x.shape


class TestSiLU:
    def test_zero(self):
        assert silu(0) == pytest.approx(0.0)

    def test_equivalent_to_swish(self):
        x = np.linspace(-5, 5, 50)
        np.testing.assert_array_almost_equal(silu(x), swish(x))

    def test_positive(self):
        assert silu(5) > 0

    def test_small_negative_dip(self):
        assert silu(-1) < 0

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert silu(x).shape == x.shape


class TestSwiGLU:
    def test_zero(self):
        assert swiglu(0) == pytest.approx(0.0)

    def test_positive(self):
        # swiglu(x) = x * swish(x), positive for positive x
        assert swiglu(3) > 0

    def test_negative_produces_small_positive(self):
        # x * swish(x): x is negative, swish(x) is negative -> positive
        assert swiglu(-2) > 0

    def test_large_positive(self):
        # swiglu(x) ~ x^2 for large positive x (since swish(x) ~ x)
        assert swiglu(5) > 20

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert swiglu(x).shape == x.shape


class TestGeGLU:
    def test_zero(self):
        assert geglu(0) == pytest.approx(0.0, abs=1e-6)

    def test_positive(self):
        assert geglu(3) > 0

    def test_large_positive(self):
        # geglu(x) = x * gelu(x) ~ x^2 for large positive x
        assert geglu(5) > 20

    def test_negative_produces_small_positive(self):
        # x * gelu(x): x negative, gelu(x) near 0 but slightly negative -> small positive
        assert geglu(-2) > 0

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert geglu(x).shape == x.shape


class TestMish:
    def test_zero(self):
        # mish(0) = 0 * tanh(ln(2)) = 0
        assert mish(0) == pytest.approx(0.0)

    def test_positive(self):
        assert mish(5) > 0

    def test_large_positive_approaches_x(self):
        # mish(x) ~ x for large positive x
        assert mish(10) == pytest.approx(10.0, abs=0.01)

    def test_small_negative_dip(self):
        assert mish(-1) < 0

    def test_bounded_below(self):
        # mish is bounded below around -0.31
        x = np.linspace(-10, 0, 1000)
        assert np.min(mish(x)) > -0.4

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert mish(x).shape == x.shape


class TestTeLU:
    def test_zero(self):
        # telu(0) = 0 * tanh(exp(0)) = 0 * tanh(1) = 0
        assert telu(0) == pytest.approx(0.0)

    def test_positive(self):
        assert telu(2) > 0

    def test_positive_approaches_x(self):
        # tanh(exp(x)) -> 1 for moderate positive x, so telu(x) ~ x
        assert telu(3) == pytest.approx(3.0, abs=0.01)

    def test_negative_suppressed(self):
        # telu(x) ~ 0 for large negative x since tanh(exp(x)) -> tanh(0) = 0
        assert telu(-10) == pytest.approx(0.0, abs=0.01)

    def test_array_shape(self):
        x = np.linspace(-5, 5, 50)
        assert telu(x).shape == x.shape


class TestGeneralProperties:
    """Cross-cutting tests for properties shared by multiple functions."""

    def test_all_functions_handle_arrays(self):
        x = np.linspace(-3, 3, 20)
        functions = [relu, leaky_relu, elu, prelu, glu, gelu, silu, swiglu, geglu, mish, telu]
        for fn in functions:
            result = fn(x)
            assert result.shape == x.shape, f"{fn.__name__} failed shape test"

    def test_all_functions_handle_scalar(self):
        functions = [relu, leaky_relu, elu, prelu, glu, gelu, silu, swiglu, geglu, mish, telu]
        for fn in functions:
            result = fn(1.0)
            assert np.isfinite(result), f"{fn.__name__} not finite for scalar input"

    def test_relu_family_positive_identity(self):
        """ReLU, Leaky ReLU, ELU, PReLU should all equal x for positive x."""
        x = np.array([1.0, 2.0, 5.0, 10.0])
        np.testing.assert_array_almost_equal(relu(x), x)
        np.testing.assert_array_almost_equal(leaky_relu(x), x)
        np.testing.assert_array_almost_equal(elu(x), x)
        np.testing.assert_array_almost_equal(prelu(x), x)

    def test_smooth_functions_near_identity_for_large_positive(self):
        """SiLU, GELU, Mish, TeLU should approach x for large positive values."""
        x = 10.0
        for fn in [silu, gelu, mish, telu]:
            assert fn(x) == pytest.approx(x, abs=0.1), f"{fn.__name__} not near identity at x=10"
