# MODULES
from pathlib import Path
import numpy as np
from tensorforce import Configuration
from tensorforce.agents import PPOAgent
from tensorforce.core.networks import layered_network_builder

# SETUP
episode_size = 20 # number of calculations
training_length = 500 # number of episodes
savefile = 'xor_network'
save_frequency = 100 # number of episodes

# NETWORK
input_layer = dict(shape=(2,), type='float')
output_layer = dict(continuous=False, num_actions=2)
hidden_layer = layered_network_builder([
    dict(type='dense', size=15, activation='relu')
])

# LEARN AGENT
learn_agent = PPOAgent(config=Configuration(
    log_level='info',
    states=input_layer,
    actions=output_layer,
    network=hidden_layer,
    episodes=training_length,
    max_timesteps=episode_size,

    # HYPERPARAMS
    batch_size=1,
    epochs=3,
    optimizer_batch_size=1,
    learning_rate=0.001,
    loss_clipping=0.2,
    entropy_penalty=0.01,
    discount=0.99,
    gae_rewards=False,
    gae_lambda=0.97,
    normalize_advantage=False
))

