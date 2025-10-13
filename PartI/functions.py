import tensorflow as tf

@tf.function
def AR1(num_samples, T, rho, sigma):
    ''' Generates an AR(1) process.

    Parameters
    ----------
    num_samples : int
        Number of sample paths to generate
    T : int
        Length of each sample path
    rho : float, optional
        Autoregressive coefficient
    sigma : float, optional
        Standard deviation of the noise term

    Returns
    -------
    y : tf.Tensor
        AR(1) process of shape (num_samples, T)
    '''
    y = tf.TensorArray(dtype=tf.float32, size=T, element_shape=(num_samples,))
    y_t = tf.random.normal(shape=(num_samples,), mean=0.0, stddev=sigma)
    y = y.write(0, y_t)
    for t in range(T):
        eps = tf.random.normal(shape=(num_samples,), mean=0.0, stddev=sigma)
        y_t = rho * y_t + eps
        y = y.write(t, y_t)
    return tf.transpose(y.stack())
    
@tf.function
def profit_function(k, z, theta=0.7):
    ''' Computes the profit function π function in Strebulaev and Whited (2012), Section 3.1.3
    k and z need to have the same shape.

    Parameters
    ----------
    k : tf.Tensor
        Capital stock
    z : tf.Tensor
        Productivity / demand shock
    theta : float, optional
        Profit function curvature, by default 0.7

    Returns
    -------
    profit : tf.Tensor
        Computed profit tensor of shape (batch_size,)
    '''
    
    return z * tf.pow(k, theta)

@tf.function
def adjustment_cost(I, k, psi0=0.01, psi1=0.0):
    ''' Computes the adjustment cost function ψ in Strebulaev and Whited (2012), Section 3.1.3
    I and k need to have the same shape.

    Parameters
    ----------
    I : tf.Tensor
        Investment
    k : tf.Tensor
        Capital stock
    psi0 : float, optional
        Coefficient to the quardratic component, by default 0.01
    psi1 : float, optional
        Coefficient to the constant component, by default 0.0

    Returns
    -------
    adjusted_cost : tf.Tensor
        Computed adjustment cost tensor of shape (batch_size,)
    '''
    return psi0 * tf.square(I) / k / 2 + tf.where(tf.abs(I) < 1e-5, 0.0, psi1 * k)