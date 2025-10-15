import pytest

import tensorflow as tf
from PartI.models import *
from PartI.functions import step

@pytest.mark.parametrize('NN, hidden_dim', 
    [(SharedLayerNN, 8), (SharedLayerNN, 16), (SharedLayerNN, 32), 
     (SepLayerNN, 8), (SepLayerNN, 16), (SepLayerNN, 32)])
def test_nn(common_assertions, k_z, NN, hidden_dim):
    delta = 0.15
    k, z = k_z
    model = NN(hidden_dim=hidden_dim, delta=delta)
    I, v = model(z, k)
    common_assertions(I, nan=True, inf=True, shape=k.shape)
    common_assertions(v, nan=True, inf=True, shape=k.shape)

    next_k = step(I, k, delta)
    common_assertions(next_k, nan=True, inf=True, sign='nonnegative', shape=k.shape)