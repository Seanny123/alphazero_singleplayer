import argparse
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from mcts import MCTS
from viz import smooth, symmetric_remove
from helpers import check_space, store_safely, Database
from atari import is_atari_game
from rl.make_game import make_game


class Model:
    """Neural Networks"""

    def __init__(self, Env, lr: float, n_hidden_layers: int, n_hidden_units: int):
        # Check the Gym environment
        self.action_dim, self.action_discrete = check_space(Env.action_space)
        self.state_dim, self.state_discrete = check_space(Env.observation_space)
        if not self.action_discrete:
            raise ValueError('Continuous action space not implemented')

        # Placeholders
        if not self.state_discrete:
            # wtf? why have both `x` and `self.x`
            self.x = x = tf.placeholder("float32", shape=np.append(None, self.state_dim), name='x')  # state
        else:
            self.x = x = tf.placeholder("int32", shape=np.append(None, 1))  # state
            x = tf.squeeze(tf.one_hot(x, self.state_dim, axis=1), axis=2)

        # Feedforward: Can be modified to any representation function, e.g. convolutions, residual networks, etc.
        for i in range(n_hidden_layers):
            x = slim.fully_connected(x, n_hidden_units, activation_fn=tf.nn.elu)

        # Output
        log_pi_hat = slim.fully_connected(x, self.action_dim, activation_fn=None)
        self.pi_hat = tf.nn.softmax(log_pi_hat)  # policy head
        self.V_hat = slim.fully_connected(x, 1, activation_fn=None)  # value head

        # Loss
        self.V = tf.placeholder("float32", shape=[None, 1], name='V')
        self.pi = tf.placeholder("float32", shape=[None, self.action_dim], name='pi')
        self.V_loss = tf.losses.mean_squared_error(labels=self.V, predictions=self.V_hat)
        self.pi_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.pi, logits=log_pi_hat)
        self.loss = self.V_loss + tf.reduce_mean(self.pi_loss)

        self.lr = tf.Variable(lr, name="learning_rate", trainable=False)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sb, Vb, pib):
        self.sess.run(self.train_op, feed_dict={self.x: sb,
                                                self.V: Vb,
                                                self.pi: pib})

    def predict_V(self, s):
        return self.sess.run(self.V_hat, feed_dict={self.x: s})

    def predict_pi(self, s):
        return self.sess.run(self.pi_hat, feed_dict={self.x: s})


def agent(game: str, n_ep: int, n_mcts: int, max_ep_len: int, lr: float, c: float, gamma: float, data_size: int,
          batch_size: int, temp: float, n_hidden_layers: int, n_hidden_units: int, replay_epochs=1):
    """ Outer training loop """
    # tf.reset_default_graph()
    episode_returns = []  # storage
    timepoints = []

    # Environments
    Env = make_game(game)
    is_atari = is_atari_game(Env)
    mcts_env = make_game(game) if is_atari else None

    db = Database(max_size=data_size, batch_size=batch_size)
    model = Model(Env=Env, lr=lr, n_hidden_layers=n_hidden_layers, n_hidden_units=n_hidden_units)
    t_total = 0  # total steps
    R_best = -np.Inf

    with tf.Session() as sess:
        model.sess = sess
        sess.run(tf.global_variables_initializer())
        for ep in range(n_ep):
            start = time.time()
            s = Env.reset()
            R = 0.0  # Total return counter
            a_store = []
            seed = np.random.randint(1e7)  # draw some Env seed
            Env.seed(seed)
            if is_atari:
                mcts_env.reset()
                mcts_env.seed(seed)

            mcts = MCTS(root_index=s, model=model, na=model.action_dim, gamma=gamma)

            for t in range(max_ep_len):
                # MCTS step
                mcts.search(n_mcts=n_mcts, c=c, Env=Env, mcts_env=mcts_env)
                state, pi, V = mcts.return_results(temp)  # extract the root output
                db.store((state, V, pi))

                # Make the true step
                a = np.random.choice(len(pi), p=pi)
                a_store.append(a)
                s1, r, terminal, _ = Env.step(a)
                R += r
                t_total += n_mcts  # total number of environment steps (counts the mcts steps)

                if terminal:
                    break
                else:
                    mcts.forward(a, s1)

            # Finished episode
            episode_returns.append(R)  # store the total episode return
            timepoints.append(t_total)  # store the timestep count of the episode return
            store_safely(os.getcwd(), 'result', {'R': episode_returns, 't': timepoints})

            if R > R_best:
                a_best = a_store
                seed_best = seed
                R_best = R

            total_time = np.round((time.time() - start), 1)
            print(f'Finished episode {ep}, total return: {np.round(R, 2)}, total time: {total_time} sec')

            # Train using replay
            db.reshuffle()
            for epoch in range(replay_epochs):
                for sb, Vb, pib in db:
                    model.train(sb, Vb, pib)

    # Return results
    return episode_returns, timepoints, a_best, seed_best, R_best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='CartPole-v0', help='Training environment')
    parser.add_argument('--n_ep', type=int, default=500, help='Number of episodes')
    parser.add_argument('--n_mcts', type=int, default=25, help='Number of MCTS traces per step')
    parser.add_argument('--max_ep_len', type=int, default=300, help='Maximum number of steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--c', type=float, default=1.5, help='UCT constant')
    parser.add_argument('--temp', type=float, default=1.0,
                        help='Temperature in normalization of counts to policy target')
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount parameter')
    parser.add_argument('--data_size', type=int, default=1000, help='Dataset size (FIFO)')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--window', type=int, default=25, help='Smoothing window for visualization')

    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers in NN')
    parser.add_argument('--n_hidden_units', type=int, default=128, help='Number of units per hidden layers in NN')

    args = parser.parse_args()
    episode_returns, timepoints, a_best, seed_best, R_best = agent(game=args.game, n_ep=args.n_ep, n_mcts=args.n_mcts,
                                                                   max_ep_len=args.max_ep_len, lr=args.lr, c=args.c,
                                                                   gamma=args.gamma,
                                                                   data_size=args.data_size, batch_size=args.batch_size,
                                                                   temp=args.temp,
                                                                   n_hidden_layers=args.n_hidden_layers,
                                                                   n_hidden_units=args.n_hidden_units)

    # Finished training: Visualize
    fig, ax = plt.subplots(1, figsize=[7, 5])
    total_eps = len(episode_returns)
    episode_returns = smooth(episode_returns, args.window, mode='valid')
    ax.plot(symmetric_remove(np.arange(total_eps), args.window - 1), episode_returns, linewidth=4, color='darkred')
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode', color='darkred')
    plt.savefig(os.path.join(os.getcwd(), 'learning_curve.png'), bbox_inches="tight", dpi=300)

#    print('Showing best episode with return {}'.format(R_best))
#    Env = make_game(args.game)
#    Env = wrappers.Monitor(Env,os.getcwd() + '/best_episode',force=True)
#    Env.reset()
#    Env.seed(seed_best)
#    for a in a_best:
#        Env.step(a)
#        Env.render()
