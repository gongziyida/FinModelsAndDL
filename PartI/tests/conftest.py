import tensorflow as tf
import pytest

@pytest.fixture(scope="module")
def I_k():
    delta = 0.15
    k = tf.pow(1 / (1 - delta), tf.range(40, dtype=tf.float32))
    k, next_k = tf.meshgrid(k, k)
    I = next_k - (1 - delta) * k
    I, k = tf.reshape(I, [-1]), tf.reshape(k, [-1])
    return I, k

@pytest.fixture(scope="module")
def k_z():
    delta = 0.15
    k = tf.pow(1 / (1 - delta), tf.range(40, dtype=tf.float32))
    z = tf.exp(tf.linspace(-2.0, 2.0, 80))
    k, z = tf.meshgrid(k, z)
    k, z = tf.reshape(k, [-1]), tf.reshape(z, [-1])
    return k, z

@pytest.fixture
def common_assertions():
    def _assertions(tensor, nan=False, inf=False, sign=None, shape=None):
        if nan:
            assert not tf.reduce_any(tf.math.is_nan(tensor)), "NaN values encountered!"
        if inf:
            assert not tf.reduce_any(tf.math.is_inf(tensor)), "Inf values encountered!"
        if sign == 'positive':
            assert tf.reduce_all(tensor > 0), "Sign mismatch encountered!"
        elif sign == 'negative':
            assert tf.reduce_all(tensor < 0), "Sign mismatch encountered!"
        elif sign == 'nonnegative':
            assert tf.reduce_all(tensor >= 0), "Sign mismatch encountered!"
        elif sign == 'nonpositive':
            assert tf.reduce_all(tensor <= 0), "Sign mismatch encountered!"
        elif sign == 'zero':
            assert tf.reduce_all(tf.abs(tensor) < 1e-5), "Sign mismatch encountered!"
        if shape is not None:
            assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}!"
    return _assertions