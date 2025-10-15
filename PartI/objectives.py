''' Contains generic objective functions for training.'''
import tensorflow as tf

#@tf.function
def lifetime_reward(rewards, beta):
    ''' Computes the discounted sum of rewards averaged over initializations.

    Parameters
    ----------
    rewards : tf.Tensor
        Instantaneous reward over time, of shape (init_size, seq_len,)
    beta : float
        Discount factor for future rewards

    Returns
    -------
    lf_reward : tf.Tensor
        Computed lifetime reward (scalar)
    '''
    discount = tf.pow(beta, tf.range(tf.shape(rewards)[1], dtype=rewards.dtype))
    lf_reward = tf.reduce_mean(tf.reduce_sum(rewards * discount, axis=1))
    return lf_reward

#@tf.function
def euler_residual(conditions, weights):
    ''' Computes the Euler residual averaged over initializations and randomness.

    Parameters
    ----------
    conditions : tf.Tensor
        Optimality conditions, of shape (n_conditions, init_size, rand_sample_size,)
    weights : tf.Tensor
        Optimality conditions weights, of shape (n_conditions,)

    Returns
    -------
    residual : tf.Tensor
        Computed Euler residual (scalar)
    '''
    squared_avg = tf.square(tf.reduce_mean(conditions, axis=-1))
    residual = tf.linalg.matvec(squared_avg, weights, transpose_a=True)
    residual = tf.reduce_mean(residual)
    return residual 

#@tf.function
def bellman_residual(values, next_values, rewards, beta, conditions, weights):
    ''' Computes the Bellman residual averaged over initializations and randomness.

    Parameters
    ----------
    values : tf.Tensor
        Values of the current states, of shape (init_size,)
    next_values : tf.Tensor
        Values of the next states, of shape (init_size, rand_sample_size,)
    rewards : tf.Tensor
        Instantaneous rewards, of shape (init_size, rand_sample_size,)
    beta : float
        Discount factor for future rewards
    conditions : tf.Tensor
        Optimality conditions, of shape (n_conditions, init_size, rand_sample_size,)
    weights : tf.Tensor
        Optimality conditions weights, of shape (n_conditions,)

    Returns
    -------
    residual : tf.Tensor
        Computed Bellman residual (scalar)
    '''
    td_err = rewards + beta * tf.reduce_mean(next_values, axis=-1) - values
    squared_avg = tf.square(tf.reduce_mean(conditions, axis=-1))
    residual = tf.linalg.matvec(squared_avg, weights, transpose_a=True)
    residual = tf.reduce_mean(tf.square(td_err) + residual)
    return residual