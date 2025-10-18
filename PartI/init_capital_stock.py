import tensorflow as tf

@tf.function
def init_k_grid(n_k, pow_bound, delta, theta, r):
    ''' Initializes the discrete grid for capital stock k
    
    Parameters
    ----------
    n_k : int
        Number of discrete grids for capital stock k
    pow_bound : float
        Maximum b for (1 - delta)^{-b} <= k / k_center <= (1 - delta)^{b}
        with k_center := ((r + delta) / theta)^{1 / (theta - 1)} 
        For details, see Strebulaev and Whited (2012), Section 3.1.2 
    delta, theta, r : float
        Rate of capital depreciation, profit function curvature, 
        and risk-free interest rate respectively

    Returns
    -------
    k_vals : tf.Tensor
        Discretized values for capital stock k, of shape (n_k,)
    '''
    k_center = tf.pow((r + delta) / theta, 1 / (theta - 1))
    pow = tf.cast(tf.linspace(-pow_bound, pow_bound, n_k), tf.float32)
    k_vals = k_center * tf.pow(1 / (1 - delta), pow)  # (n_k,)
    return k_vals

@tf.function
def rand_k(shape, std_pow, delta, theta, r):
    ''' Generates random capital stock k samples uniformly in log-space

    Parameters
    ----------
    shape : tuple of int
        Shape of the output tensor
    std_pow : float
        Standard deviation of the normal distribution of b for
        k = k_center * (1 / (1 - delta))^b
    delta, theta, r : float
        Rate of capital depreciation, profit function curvature, 
        and risk-free interest rate respectively

    Returns
    -------
    k : tf.Tensor
        Randomly generated capital stock samples of shape `shape`
    '''
    k_center = tf.pow((r + delta) / theta, 1 / (theta - 1))
    pow = tf.random.normal(shape=shape, mean=0.0, stddev=std_pow, dtype=tf.float32)
    k = k_center * tf.pow(1 / (1 - delta), pow)
    return k