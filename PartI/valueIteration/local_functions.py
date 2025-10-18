import tensorflow as tf
import tensorflow_probability as tfp
from PartI.functions import profit_function, adjustment_cost
from PartI.init_capital_stock import init_k_grid

@tf.function
def value_function(k, lnz, k_vals, V_table, lnz_transition_prob, psi0, psi1, delta, r, theta):
    ''' Computes the tabular value function V given in Strebulaev and Whited (2012), Eq. 3.10.

    Parameters
    ----------
    k : tf.Tensor
        Capital stock tensor of shape (batch_size,)
    lnz : tf.Tensor
        Productivity / demand shock tensor of shape (batch_size,)
    k_vals : tf.Tensor
        All possible discrete values of capital stock, of shape (num_k,)
    V_table : tf.Tensor
        Values for (z, k) pairs, of shape (num_z, num_k)
    lnz_transition_prob : tf.Tensor
        Transition probability matrix for ln(z') given ln(z), of shape (batch_size, num_lnz)
    psi0 : float
        Coefficient to the quardratic component in the adjusted cost
    psi1 : float
        Coefficient to the constant component in the adjusted cost
    delta : float
        Constant rate of capital depreciation
    r : float
        Risk-free interest rate
    theta : float
        Profit function curvature in the profit function

    Returns
    -------
    V_max : tf.Tensor
        Value of (z, k) under the optimal policy, of shape (batch_size,)
    argmax_k : tf.Tensor
        Optimal k' given (z, k), of shape (batch_size,)
    '''
    expected_V_given_k = lnz_transition_prob @ V_table # shape (batch_size, num_k)
    pi = profit_function(k, tf.exp(lnz), theta) # shape (batch_size,)
    I = k_vals[None,:] - (1 - delta) * k[:,None] # shape (batch_size, num_k)
    psi = adjustment_cost(I, k[:,None], psi0, psi1) # shape (batch_size, num_k)
    V = pi[:,None] - psi - I + expected_V_given_k / (1 + r) # shape (batch_size, num_k)
    V_max = tf.reduce_max(V, axis=-1) # shape (batch_size,)
    argmax_k = tf.gather(k_vals, tf.argmax(V, axis=-1)) # shape (batch_size,)
    return V_max, argmax_k

@tf.function
def AR1_transition_matrix(n_grids, rho=0.7, sigma=0.15, m=6):
    ''' Computes the transition probability matrix for 
    the discretized y~AR(1) using Tauchen (1986).

    Parameters
    ----------
    n_grids : int
        Number of discrete grids for y
    rho : float, optional
        Autoregressive coefficient, by default 0.7
    sigma : float, optional
        Standard deviation of the noise term in AR(1), by default 0.15
    m : int, optional
        Maximum absolute value of y as a multiple of the unconditional std, 
        i.e. |y| < m * sigma / sqrt(1 - rho**2), by default 5

    Returns
    -------
    P: tf.Tensor
        Transition probability matrix of shape (n_grids, n_grids)
        P[i,j] = P(y' = y_j | y = y_i)
    y: tf.Tensor
        Discretized y values of shape (n_grids,)
    '''
    y_bound = m * sigma / tf.sqrt(1 - rho**2)
    y = tf.linspace(-y_bound, y_bound, n_grids)
    bw = (y[1] - y[0]) / 2 / sigma # half bin width scaled by sigma
    
    quantiles = (y[None,:] - rho * y[:,None]) / sigma
    
    dist = tfp.distributions.Normal(loc=0, scale=1)
    # get the cumulative density at the left and right edges of each bin
    p_r = dist.cdf(quantiles + bw)
    p_l = dist.cdf(quantiles - bw)

    # transition probability matrix, of shape (n_grids, n_grids)
    P = tf.concat([p_r[:,0:1], p_r[:,1:-1] - p_l[:,1:-1], 1 - p_l[:,-1:]], axis=-1)
    return tf.clip_by_value(P, 0.0, 1.0), y


def value_iteration(max_iter, n_lnz, n_k, pow_bound, psi0, psi1, 
                    rho, sigma, m, delta, r, theta):
    ''' Solves the value function iteration in Strebulaev and Whited (2012), Section 3.1.3

    Parameters
    ----------
    max_iter : int
        Maximum number of value function iterations
    n_lnz, n_k : int
        Number of discrete grids for log shock ln(z) and capital stock k respectively
    Other parameters see z_transition_matrix() and value_function()

    Returns
    -------
    V_tables : tf.Tensor
        Value function for (ln(z), k) pairs over iterations, of shape (T+1, n_lnz, n_k)
    lnz_vals, k_vals : tf.Tensor
        Discretized values for ln(z) and k, of shape (n_lnz,) and (n_k,)
    '''
    k_vals = init_k_grid(n_k, pow_bound, delta, theta, r) # (n_k,)
    Pz, lnz_vals = AR1_transition_matrix(n_lnz, rho, sigma, m) # (n_lnz, n_lnz) and (n_lnz,)
    V_table = tf.zeros((n_lnz, n_k), dtype=tf.float32) # initial V_table

    # z varies over rows, k varies over columns
    lnz_grid, k_grid = tf.meshgrid(lnz_vals, k_vals, indexing='ij') 
    lnz_grid, k_grid = tf.reshape(lnz_grid, [-1]), tf.reshape(k_grid, [-1]) # (n_lnz*n_k,)

    Pz_grid = tf.repeat(Pz, n_k, axis=0) # (n_lnz*n_k, n_lnz)

    max_diff, rel_diff = [], [] # for monitoring convergence
    for _ in range(max_iter): # value function iteration
        new_V_table = value_function(k_grid, lnz_grid, k_vals, V_table, Pz_grid, 
                                     psi0=psi0, psi1=psi1, 
                                     delta=delta, r=r, theta=theta)[0]
        new_V_table = tf.reshape(new_V_table, (n_lnz, n_k))

        diff = tf.abs(new_V_table - V_table)
        max_diff.append(tf.reduce_max(diff))
        rel_diff.append(tf.reduce_max(diff / (V_table + 1e-5)))
        if max_diff[-1] < 1e-4 and rel_diff[-1] < 1e-2:
            print(f'Converged after {_} iterations.')
            V_table = new_V_table
            break
        V_table = new_V_table
    return V_table, lnz_vals, k_vals, tf.stack(max_diff), tf.stack(rel_diff)
