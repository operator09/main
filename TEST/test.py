import numpy as np
import gym
import tensorflow as tf

ENV_NAME = 'CartPole-v0'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_action = env.action_space.n

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape(1,)+env.observation_space.shape),
                                    tf.keras.layers.Dense(16,activation='relu'),
                                    tf.keras.layers.Dense(nb_action, activation='linear'),
                                    ])
