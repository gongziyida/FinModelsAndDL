''' Contains basic functions used in the models in Strebulaev and Whited (2012). '''
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
    rho : float
        Autoregressive coefficient
    sigma : float
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
    
#@tf.function
def step(k, I, delta):
    ''' Computes the next period capital stock k[t+1] given current capital stock k[t] and investment I[t].
    k and I need to have the same shape.

    Parameters
    ----------
    k : tf.Tensor
        Current capital stock
    I : tf.Tensor
        Current investment
    delta : float
        Depreciation rate

    Returns
    -------
    next_k : tf.Tensor
        Next period capital stock, of the same shape as k and I
    '''
    new_k = (1 - delta) * k + I
    # assert tf.reduce_all(tf.logical_or(new_k > 0, tf.abs(new_k) < 1e-5)), \
    #     "Negative capital stock encountered!"
    # clip to avoid numerical issues
    return tf.where(new_k < 1e-5, 1e-5, new_k)

@tf.function
def profit_function(k, z, theta):
    ''' Computes the profit function π function in Strebulaev and Whited (2012), Section 3.1.3
    k and z need to have the same shape.

    Parameters
    ----------
    k : tf.Tensor
        Capital stock
    z : tf.Tensor
        Productivity / demand shock
    theta : float
        Profit function curvature

    Returns
    -------
    profit : tf.Tensor
        Computed profit tensor of the same shape as k and z
    '''
    
    return z * tf.pow(k, theta)

@tf.function
def marginal_product_of_capital(k, z, theta):
    ''' Computes the marginal product of capital, a.k.a. the partial derivative of 
    the profit π w.r.t. capital k in Strebulaev and Whited (2012), Section 3.1.3
    k and z need to have the same shape.
    '''
    return z * tf.pow(k + 1e-5, theta - 1.0) * theta

@tf.function
def adjustment_cost(I, k, psi0, psi1):
    ''' Computes the adjustment cost function ψ in Strebulaev and Whited (2012), Section 3.1.3
    I and k need to have the same shape.

    Parameters
    ----------
    I : tf.Tensor
        Investment
    k : tf.Tensor
        Capital stock
    psi0, psi1 : float
        Coefficients to the quardratic component and the constant component, respectively

    Returns
    -------
    adjusted_cost : tf.Tensor
        Computed adjustment cost tensor of the same shape as I and k
    '''
    return psi0 * tf.square(I) / k / 2 + tf.where(tf.abs(I) < 1e-5, 0.0, psi1 * k)

@tf.function
def marginal_cost_of_capital(I, k, psi0, psi1):
    ''' Computes the marginal cost of capital, a.k.a. the partial derivative of 
    the adjustment cost ψ w.r.t. capital k in Strebulaev and Whited (2012), Section 3.1.3
    I and k need to have the same shape.
    '''
    return -psi0 * tf.square(I / k) / 2 + tf.where(tf.abs(I) < 1e-5, 0.0, psi1)

@tf.function
def marginal_cost_of_investment(I, k, psi0):
    ''' Computes the marginal cost of investment, a.k.a. the partial derivative of 
    the adjustment cost ψ w.r.t. investment I in Strebulaev and Whited (2012), Section 3.1.3
    I and k need to have the same shape.
    '''
    return psi0 * I / k

@tf.function
def cash_flow(k, I, z, theta, psi0, psi1):
    ''' Computes the cash flow in Strebulaev and Whited (2012), Section 3.1.3

    Parameters
    ----------
    k : tf.Tensor
        Current capital stock
    I : tf.Tensor
        Current investment
    z : tf.Tensor
        Current productivity / demand shock
    theta : float
        Profit function curvature
    psi0, psi1 : float
        Coefficients to the quardratic component and the constant component, respectively

    Returns
    -------
    cash_flow : tf.Tensor
        Computed cash flow tensor of the same shape as k, I, and z
    '''
    return profit_function(k, z, theta) - adjustment_cost(I, k, psi0, psi1) - I

#@tf.function
def optimal_condition(k, next_k, I, z, psi0, psi1, theta, r, delta):
    ''' Computes the optimality condition for the model
    in Strebulaev and Whited (2012), Section 3.1.3, Eq. 3.8

    Parameters
    ----------
    k : tf.Tensor
        Current capital stock
    next_k : tf.Tensor
        Next period capital stock, of shape (*k.shape, rand_sample_size)
    I : tf.Tensor
        Current investment, of the same shape as k
    z : tf.Tensor
        Current exogenous shock, of the same shape as next_k
    psi0, psi1 : float
        Coefficients to the quardratic component and the constant component, respectively
    theta : float
        Profit function curvature
    r : float
        Interest rate
    delta : float
        Depreciation rate

    Returns
    -------
    condition : tf.Tensor
        Computed optimality condition tensor of shape (*dpsi_dI_prev.shape,)
    '''
    dpi_dk = marginal_product_of_capital(next_k, z, theta)
    dpsi_dk = marginal_cost_of_capital(I, next_k, psi0, psi1)
    dpsi_dI = marginal_cost_of_investment(I, next_k, psi0)
    dpsi_dI_prev = marginal_cost_of_investment(I, k, psi0)
    
    shadow_value = dpi_dk - dpsi_dk + (1 - delta) * (1 + dpsi_dI_prev)
    shadow_value = tf.reduce_mean(shadow_value, axis=-1, keepdims=True) # average over next_k randomness
    marginal_cost = (1 + dpsi_dI) * (1 + r)
    return shadow_value - marginal_cost