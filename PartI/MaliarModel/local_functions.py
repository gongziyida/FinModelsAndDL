import tensorflow as tf

def build_forward_nn(hidden_dim):
    ''' Builds a feedforward neural network model as in Maliar et al. 2021.
    The model maps input state (y, w) to a single output value, where
    y is exogenous income shock and w is cash-on-hand.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the model

    Returns
    -------
    model : tf.keras.Model
    '''
    input_y = tf.keras.Input(shape=(1,), name='y')
    input_w = tf.keras.Input(shape=(1,), name='w')
    x = tf.keras.layers.Concatenate(axis=-1)([input_y, input_w])
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    # x = tf.keras.layers.Dense(1, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=[input_y, input_w], outputs=x)
    return model

@tf.function
def step(w, c, y, r=0.04):
    ''' Computes the next cash-on-hand w' given current cash-on-hand w,
    consumption c, exogenous income shock y, and interest rate r.

    Parameters
    ----------
    w : tf.Tensor
        Current cash-on-hand, of shape (batch_size,)
    c : tf.Tensor
        Consumption, of shape (batch_size,)
    y : tf.Tensor
        Exogenous income shock, of shape (batch_size,)
    r : float, optional
        Interest rate, by default 0.04

    Returns
    -------
    w_next : tf.Tensor
        Next cash-on-hand  of shape (batch_size,)
    '''
    return (1 + r) * (w - c) + tf.exp(y)

@tf.function
def utility(c, gamma=2.0):
    ''' Computes the utility for given consumptions (states).
    Utility function is u(c) = (c^(1-gamma) - 1)/(1-gamma) as per Maliar et al. 2021.

    Parameters
    ----------
    c : tf.Tensor
        Current consumption 
    gamma : float
        Risk aversion parameter, by default 2

    Returns
    -------
    lf_reward : tf.Tensor
        Computed reward  of the shape as consumption
    '''
    return (tf.pow(c + 1e-2, 1 - gamma) - 1) / (1 - gamma) 

@tf.function
def utility_derivative(c, gamma=2.0):
    ''' Computes the derivative of the utility function for given consumptions (states).'''
    return tf.pow(c + 1e-2, -gamma)

@tf.function
def FB_euler_residual(c, next_c, c_w_ratios, gamma=2.0, beta=0.9, r=0.04):
    ''' Computes the Euler residual objective using the Fischer-Burmeister function 
    as per Maliar et al. 2021.

    Parameters
    ----------
    c : tf.Tensor
        Current consumption
    next_c : tf.Tensor
        Consumption at the next time step, of shape (*c.shape, rand_sample_size)
    c_w_ratios : tf.Tensor
        Current consumption-to-wealth ratio, of the same shape as `c`
    gamma : float, optional
        Risk aversion parameter, by default 2
    beta : float, optional
        Discount factor for future rewards, by default 0.9
    r : float, optional
        Interest rate, by default 0.04

    Returns
    -------
    fb : tf.Tensor
        Fischer-Burmeister function outputs, of shape (*c.shape, 1)
    '''
    du, next_du = utility_derivative(c, gamma), utility_derivative(next_c, gamma)
    h = beta * (1 + r) * tf.reduce_mean(next_du, axis=-1) / du
    a, b = 1 - c_w_ratios, 1 - h
    fb_err = a + b - tf.sqrt(tf.square(a) + tf.square(b))
    return tf.expand_dims(fb_err, axis=-1)

@tf.function
def FB_bellman_residual(c, c_w_ratios, next_dv, gamma=2.0, beta=0.9):
    ''' Computes the Bellman residual objective using the Fischer-Burmeister function 
    as per Maliar et al. 2021.

    Parameters
    ----------
    c : tf.Tensor
        Current consumption
    c_w_ratios : tf.Tensor
        Current consumption-to-wealth ratio, of the same shape as `V`
    next_dv : tf.Tensor
        Value function derivative at the next time step, of shape (*V.shape, rand_sample_size)
    gamma : float, optional
        Risk aversion parameter, by default 2
    beta : float, optional
        Discount factor for future rewards, by default 0.9

    Returns
    -------
    fb : tf.Tensor
        Fischer-Burmeister function outputs, of shape (*c.shape, 1)
    '''
    du = utility_derivative(c, gamma)
    a = 1 - c_w_ratios
    b = 1 - beta * tf.reduce_mean(next_dv, axis=-1) / du
    fb_err = a + b - tf.sqrt(tf.square(a) + tf.square(b))
    return tf.expand_dims(fb_err, axis=-1)