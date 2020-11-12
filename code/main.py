import os
import argparse
from datetime import datetime
import gym

from .agent import SacAgent


def train_sac(env, wandb_writer, config, runid):
    log_dir = os.path.join(
        'logs', runid,
        f'sac')

    agent = SacAgent(env=env, log_dir=log_dir, wandb_writer = wandb_writer, **config)
    if config.load_sac:
        agent.load_models(config.sac_path)
    agent.run()
    return agent