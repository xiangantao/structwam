"""Reuse ACT state/action processors; each frozen encoder owns its RGB normalization."""

from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_latent_act_pre_post_processors(config, dataset_stats=None):
    config.validate_features()
    return make_act_pre_post_processors(config, dataset_stats)
