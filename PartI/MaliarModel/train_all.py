from multiprocessing import Value
import os
import numpy as np
import tensorflow as tf
from PartI.MaliarModel.local_functions import *
from PartI.MaliarModel.training_utils import *

import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = lambda fname: os.path.join(script_dir, fname)

def main(objective, hidden_dim, num_epochs, batch_size, eval_interval, T):
    ''' Main function to train and test the model that optimizes the given objective.'''
    model = build_forward_nn(hidden_dim=hidden_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    objs_train = np.zeros(num_epochs)
    rewards_eval = np.zeros(num_epochs//eval_interval)

    # need to do that for each new model and optimizer
    if objective == 'lifetime reward':
        train_step = tf.function(train_step_lf_reward)
    elif objective == 'Euler residual':
        train_step = tf.function(train_step_euler_residual)
        train_step = (train_step_euler_residual)
    elif objective == 'Bellman residual':
        train_step = tf.function(train_step_bellman_residual)
    else:
        raise ValueError('Objective not known.')

    for epoch in range(num_epochs):
        if objective == 'lifetime reward':
            obj_train = train_step(model, optimizer, batch_size=batch_size, T=T)
        else:
            obj_train = train_step(model, optimizer, batch_size=batch_size)
        objs_train[epoch] = obj_train.numpy().copy()
        
        if epoch % eval_interval == 0: # evaluate every eval_interval epochs
            eval = eval_lf_reward(model, batch_size=batch_size*10, T=T)
            rewards_eval[epoch//eval_interval] = eval.numpy().copy()

        if epoch % 100 == 0:
            print(f'[{epoch}] {objective} = {objs_train[epoch]:.4f}')

    return objs_train, rewards_eval


def train_single_obj(objective):
    ''' Train on a single objective with different hidden dimensions
    and plots the training and evaluation results.
    '''
    num_epochs, eval_interval, batch_size = 10000, 50, 64
    T = 100 # for lifetime reward only
    # hidden_dims = [8]
    hidden_dims = [8, 16, 32, 64]
    all_train_objs, all_eval_rewards = [], []
    for h in hidden_dims:
        print(f'Testing hidden dimension: {h}')
        res = main(objective, hidden_dim=h, num_epochs=num_epochs, 
                   eval_interval=eval_interval, batch_size=batch_size, T=T)
        objs_train, rewards_eval = res
        all_train_objs.append(objs_train)
        all_eval_rewards.append(rewards_eval)

    # Plot training and testing results
    fig, ax = plt.subplots(1, 2, figsize=(8,4), sharex='all')
    ts = np.arange(0, num_epochs, eval_interval)
    for i, h in enumerate(hidden_dims):
        ax[0].plot(ts, all_train_objs[i][::eval_interval], label=f'{h} x {h} relu')
        ax[1].plot(ts, all_eval_rewards[i])
    ax[0].legend()
    ax[0].set(xlabel='Epoch', title='Objective: '+objective)
    if objective == 'lifetime reward':
        ylims = ax[0].get_ylim()
        ax[0].set(ylim=[max(-1, ylims[0]), min(1, ylims[1])])
    elif objective == 'Euler residual':
        ax[0].set(yscale='log')
    ylims = ax[1].get_ylim()
    ax[1].set(xlabel='Epoch', title='Evaluation: lifetime reward',
              ylim=[max(-1, ylims[0]), min(1, ylims[1])])
    fig.savefig(save_path(f'{objective.replace(' ', '_')}.jpg'), dpi=100)

if __name__ == '__main__':
    train_single_obj('lifetime reward')
    train_single_obj('Euler residual')
    train_single_obj('Bellman residual')