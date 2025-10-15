import tensorflow as tf
from PartI.MaliarModel.local_functions import *
from PartI.objectives import *
from PartI.functions import AR1

#### Lifetime reward ####
def train_step_lf_reward(model, optimizer, batch_size, T, w1=0.1, w2=4.0, 
                         beta=0.9, gamma=2.0, rho=0.0, sigma=0.1, r=0.04):
    ''' Train the model for one step on the lifetime reward objective.

    Parameters
    ----------
    model : tf.keras.Model
    optimizer : tf.keras.optimizers.Optimizer
    batch_size : int
    T : int
        Length of each sample path
    w1 : float, optional
        Lower bound of initial cash-on-hand, by default 0.1
    w2 : float, optional
        Upper bound of initial cash-on-hand, by default 0.1
    beta : float, optional
        discount factor, by default 0.9
    gamma : float, optional
        Risk aversion parameter, by default 2.0
    rho : float, optional
        Autoregressive coefficient, by default 0.9
    sigma : float, optional
        Standard deviation of the noise term, by default 0.1
    r : float, optional
        Interest rate, by default 0.04

    Returns
    -------
    lifetime_reward : tf.Tensor
        Averaged lifetime reward (scalar)
    '''
    # Initialize exogenous shocks y as an AR(1) process, cash-on-hand w, and consumption c
    y = AR1(batch_size, T, rho, sigma)
    w = tf.random.uniform(shape=(batch_size,), minval=w1, maxval=w2)
    c = tf.TensorArray(dtype=tf.float32, size=T)
    
    with tf.GradientTape() as tape:
        for t in range(T):
            # Consumption as a fraction of cash-on-hand
            c_w_ratio = tf.sigmoid(tf.squeeze(model([y[:,t], w]), axis=-1))
            c_t = c_w_ratio * w
            # w = step(w, tf.stop_gradient(c_t), y[:,t], r) # step; no gradient
            w = step(w, c_t, y[:,t])
            c = c.write(t, c_t)

        c = tf.transpose(c.stack(), perm=[1,0])
        neg_reward = -lifetime_reward(utility(c, gamma), beta) # minimize negative reward

    gradient = tape.gradient(neg_reward, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return -neg_reward # return positive reward

@tf.function
def eval_lf_reward(model, batch_size, T, w1=0.1, w2=4.0, 
                   beta=0.9, gamma=2.0, rho=0.0, sigma=0.1, r=0.04):
    ''' Evaluate the model on the lifetime reward objective.'''
    # Initialize exogenous shocks y as an AR(1) process, cash-on-hand w, and consumption c
    y = AR1(batch_size, T, rho, sigma)
    w = tf.random.uniform(shape=(batch_size,), minval=w1, maxval=w2)
    c = tf.TensorArray(dtype=tf.float32, size=T)

    for t in range(T):
        # Consumption as a fraction of cash-on-hand
        c_w_ratio = tf.sigmoid(tf.squeeze(model([y[:,t], w]), axis=-1))
        c_t = c_w_ratio * w
        w = step(w, c_t, y[:,t], r)
        c = c.write(t, c_t)

    c = tf.transpose(c.stack(), perm=[1,0])
    reward = lifetime_reward(utility(c, gamma), beta)

    return reward
    

#### Euler residual ####
def train_step_euler_residual(model, optimizer, batch_size, w1=0.1, w2=4.0, 
                              beta=0.9, gamma=2.0, rho=0.0, sigma=0.1, r=0.04):
    ''' Train the model for one step on the Euler residual objective.

    Parameters
    ----------
    model : tf.keras.Model
    optimizer : tf.keras.optimizers.Optimizer
    batch_size : int
    w1 : float, optional
        Lower bound of initial cash-on-hand, by default 0.1
    w2 : float, optional
        Upper bound of initial cash-on-hand, by default 0.1
    beta : float, optional
        discount factor, by default 0.9
    gamma : float, optional
        Risk aversion parameter, by default 2.0
    rho : float, optional
        Autoregressive coefficient, by default 0.9
    sigma : float, optional
        Standard deviation of the noise term, by default 0.1
    r : float, optional
        Interest rate, by default 0.04

    Returns
    -------
    loss : tf.Tensor
        Averaged Euler residual (scalar)
    '''
    # Initialize exogenous shocks y's and cash-on-hand w
    w = tf.random.uniform(shape=(batch_size,), minval=w1, maxval=w2)
    y0 = tf.random.normal(shape=(batch_size,), mean=0, stddev=sigma/tf.sqrt(1.0-rho))
    # shape (batch_size, batch_size), second dimension is next step randomness
    y1 = y0[:,None] * rho + tf.random.normal(shape=(1,batch_size), mean=0, stddev=sigma)

    with tf.GradientTape() as tape:
        model_out = tf.squeeze(model([y0, w]), axis=-1)
        # Consumption as a fraction of cash-on-hand
        c_w_ratio = tf.sigmoid(model_out)
        c = c_w_ratio * w
        
        # next w, dimensions matching y1
        # next_w = step(w, tf.stop_gradient(c), y0, r)
        next_w = step(w, c, y0, r)
        next_w = tf.tile(tf.expand_dims(next_w, axis=1), [1,batch_size])
        y1, next_w = tf.reshape(y1, [-1]), tf.reshape(next_w, [-1]) # flatten
        
        model_out = tf.squeeze(model([y1, next_w]), axis=-1)
        next_c = tf.sigmoid(model_out) * next_w
        next_c = tf.reshape(next_c, [batch_size, batch_size])

        cond = FB_euler_residual(c, next_c, c_w_ratio, gamma, beta, r) # (batch_size,)
        loss = euler_residual(cond[None,:], tf.ones([1]))

    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss

@tf.function
def eval_euler_residual(model, batch_size, w1=0.1, w2=4.0, 
                        beta=0.9, gamma=2.0, rho=0.0, sigma=0.1, r=0.04):
    ''' Evaluate the model on the Euler residual objective.'''
    # Initialize exogenous shocks y's and cash-on-hand w
    w = tf.random.uniform(shape=(batch_size,), minval=w1, maxval=w2)
    y0 = tf.random.normal(shape=(batch_size,), mean=0, stddev=sigma/tf.sqrt(1.0-rho))
    # shape (batch_size, batch_size), second dimension is next step randomness
    y1 = y0[:,None] * rho + tf.random.normal(shape=(1,batch_size), mean=0, stddev=sigma)

    model_out = tf.squeeze(model([y0, w]), axis=-1)
    # Consumption as a fraction of cash-on-hand
    c_w_ratio = tf.sigmoid(model_out)
    c = c_w_ratio * w
    
    # next w, dimensions matching y1
    next_w = tf.tile(tf.expand_dims(step(w, c, y0, r), axis=1), [1,batch_size])
    y1, next_w = tf.reshape(y1, [-1]), tf.reshape(next_w, [-1]) # flatten
    
    next_c = tf.sigmoid(tf.squeeze(model([y1, next_w]), axis=-1)) * next_w
    next_c = tf.reshape(next_c, [batch_size, batch_size])

    cond = FB_euler_residual(c, next_c, c_w_ratio, gamma, beta, r) # (batch_size,)
    loss = euler_residual(cond[None,:], tf.ones([1]))

    return loss 


#### Bellman residual ####
def train_step_bellman_residual(model, optimizer, batch_size, w1=0.1, w2=4.0, 
                                beta=0.9, gamma=2.0, rho=0.0, sigma=0.1, r=0.04):
    ''' Train the model for one step on the Bellman residual objective.

    Parameters
    ----------
    model : tf.keras.Model
    optimizer : tf.keras.optimizers.Optimizer
    batch_size : int
    w1 : float, optional
        Lower bound of initial cash-on-hand, by default 0.1
    w2 : float, optional
        Upper bound of initial cash-on-hand, by default 0.1
    beta : float, optional
        discount factor, by default 0.9
    gamma : float, optional
        Risk aversion parameter, by default 2.0
    rho : float, optional
        Autoregressive coefficient, by default 0.9
    sigma : float, optional
        Standard deviation of the noise term, by default 0.1
    r : float, optional
        Interest rate, by default 0.04

    Returns
    -------
    loss : tf.Tensor
        Averaged Bellman residual (scalar)
    '''
    # Initialize exogenous shocks y's and cash-on-hand w
    w = tf.random.uniform(shape=(batch_size,), minval=w1, maxval=w2)
    y0 = tf.random.normal(shape=(batch_size,), mean=0, stddev=sigma/tf.sqrt(1.0-rho))
    # shape (batch_size, batch_size), second dimension is next step randomness
    y1 = y0[:,None] * rho + tf.random.normal(shape=(1,batch_size), mean=0, stddev=sigma)

    with tf.GradientTape() as tape:
        model_out = tf.squeeze(model([y0, w]), axis=-1)
        v = model_out # value

        # Consumption as a fraction of cash-on-hand
        c_w_ratio = tf.sigmoid(model_out)
        c = c_w_ratio * w
        
        # next w, dimensions matching y1
        # next_w = step(w, tf.stop_gradient(c), y0, r)
        next_w = step(w, c, y0, r)
        next_w = tf.tile(tf.expand_dims(next_w, axis=1), [1,batch_size]) 
        y1, next_w = tf.reshape(y1, [-1]), tf.reshape(next_w, [-1]) # flatten

        # Calculate next value and its derivative w.r.t. next_w
        with tf.GradientTape() as tape_dV:
            tape_dV.watch(next_w)
            model_out = tf.squeeze(model([y1, next_w]), axis=-1)
            next_v = tf.reshape(model_out, [batch_size, batch_size])
        # next_dv = tf.stop_gradient(tape_dV.gradient(next_v, next_w))
        next_dv = tape_dV.gradient(next_v, next_w)
        next_dv = tf.reshape(next_dv, [batch_size, batch_size])
        # next_v = tf.stop_gradient(next_v)

        u = utility(c, gamma)

        cond = FB_bellman_residual(c, c_w_ratio, next_dv, gamma, beta) # (batch_size,)
        loss = bellman_residual(v, next_v, u, beta, cond[None,:], tf.ones([1]))

    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    return loss