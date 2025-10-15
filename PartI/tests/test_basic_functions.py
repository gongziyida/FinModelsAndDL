''' Test the basic functions are in their expected ranges and shapes.'''
import pytest
import tensorflow as tf
from PartI.functions import *

def test_step(common_assertions, I_k):
    delta = 0.15
    I, k = I_k
    next_k = step(I, k, delta)
    common_assertions(next_k, nan=True, inf=True, sign='nonnegative', shape=k.shape)

def test_profit_function(common_assertions, k_z):
    theta = 0.7
    k, z = k_z
    pi = profit_function(k, z, theta)
    common_assertions(pi, nan=True, inf=True, sign='nonnegative', shape=k.shape)

def test_dpi_dk(common_assertions, k_z):
    theta = 0.7
    k, z = k_z
    dpi = marginal_product_of_capital(k, z, theta)
    common_assertions(dpi, nan=True, inf=True, sign='nonnegative', shape=k.shape)

@pytest.mark.parametrize('psi0, psi1', [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
def test_adjustment_cost(common_assertions, I_k, psi0, psi1):
    I, k = I_k
    psi = adjustment_cost(I, k, psi0, psi1)
    common_assertions(psi, nan=True, inf=True, sign='nonnegative', shape=k.shape)

@pytest.mark.parametrize('psi0, psi1', [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
def test_dpsi_dk(common_assertions, I_k, psi0, psi1):
    I, k = I_k
    dpsi_dk = marginal_cost_of_capital(I, k, psi0, psi1)
    common_assertions(dpsi_dk, nan=True, inf=True, sign=None, shape=k.shape)

@pytest.mark.parametrize('psi0, psi1', [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
def test_dpsi_dI(common_assertions, I_k, psi0, psi1):
    I, k = I_k
    dpsi_dI = marginal_cost_of_capital(I, k, psi0, psi1)
    common_assertions(dpsi_dI, nan=True, inf=True, sign=None, shape=k.shape)

@pytest.mark.parametrize('psi0, psi1', [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)])
def test_optimal_condition(common_assertions, I_k, psi0, psi1):
    delta = 0.15
    r = 0.04
    theta = 0.7

    I, k = I_k
    next_k = step(I, k, delta)

    sample_size = 100
    I_k_ratio = tf.random.uniform(shape=(sample_size,), minval=-(1-delta), maxval=5, dtype=tf.float32)
    next_I = I_k_ratio[None,:] * k[:,None]
    z = tf.exp(tf.random.uniform(shape=(sample_size,), minval=-2.0, maxval=2.0, dtype=tf.float32))
    z = tf.tile(z[None,:], [k.shape[0], 1])

    cond = optimal_condition(k, next_k, I, next_I, z, psi0, psi1, theta, r, delta)
    common_assertions(cond, nan=True, inf=True, sign=None, shape=(k.shape[0], 1))

@pytest.mark.parametrize('rho, sigma', [(0.7, 0.15), (0.9, 0.1), (0.5, 0.2)])
def test_AR1(common_assertions, rho, sigma):
    batch_size = 1000
    T = 500
    lnz = AR1(batch_size, T, rho, sigma)
    common_assertions(lnz, nan=True, inf=True, shape=(batch_size, T))

    dW = lnz[:,1:] - rho * lnz[:,:-1]
    mean, std = tf.math.reduce_mean(dW), tf.math.reduce_std(dW)
    assert mean < 1e-3, f"mean of Wiener increments = {mean}, not close to 0!"
    assert tf.abs(std - sigma) < 1e-3, f"std of Wiener increments = {std}, not close to sigma = {sigma}!"
    