"""Phrase identity and bundle occupancy without student training."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vace_latent_distill.bundles import phrase_positions, trajectory_density, word_heats


class Tokenizer:
    def __init__(self):
        self.words = {}

    def __call__(self, text, **_):
        for word in text.split():
            self.words.setdefault(word, len(self.words) + 1)
        return {"input_ids": [self.words[word] for word in text.split()]}


def test_phrase_masks_keep_repeated_entities_and_modifiers():
    masks = phrase_positions(Tokenizer(), "move red mug from plate to plate", ["red mug", "plate", "plate"])
    assert list(masks) == ["red mug", "plate"]
    assert np.flatnonzero(masks["red mug"]).tolist() == [1, 2]
    assert np.flatnonzero(masks["plate"]).tolist() == [4, 6]
    with pytest.raises(ValueError, match="absent"):
        phrase_positions(Tokenizer(), "move red mug", ["blue mug"])


def test_full_prompt_is_not_constant_word_sum():
    attention = np.asarray([[[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]], np.float32)
    heat = word_heats(attention, [True, True, False], {"first": [True, False, False],
                                                     "pair": [True, True, False]})
    assert np.allclose(heat[0], [[0.7, 0.3]])
    assert np.allclose(heat[1], [[0.7, 0.2]])
    assert np.allclose(heat[2], [[0.9, 0.5]])
    with pytest.raises(ValueError, match="content-token"):
        word_heats(attention, [False] * 3, {})


def test_density_keeps_support_anchor_identity_and_point_mass():
    paths = np.zeros((1, 2, 2, 3, 2), np.float32)
    support = np.zeros(paths.shape[:-1], bool)
    paths[0, 0, :, 1] = ((0.5, 0.5), (2, 2))
    support[0, 0, :, 1] = True
    paths[0, 1, 0, 2] = (3.7, 3.7)
    support[0, 1, 0, 2] = True
    heat = trajectory_density(paths, support, 4)
    assert heat.shape == (1, 2, 3, 4, 4)
    assert np.allclose(heat.sum((-1, -2)), [[[0, 1, 0], [0, 0, 1]]])
    assert heat[0, 0, 1, 0, 0] == 0.125
    assert heat[0, 0, 1, 2, 2] == 0.5
    assert heat[0, 1, 2, 3, 3] == 1
