import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting in tests

import pytest
import torch

from interp_toolkit import PromptPair, load_model, to_tokens

CLEAN = "The capital of France is"
CORRUPTED = "The capital of Italy is"


@pytest.fixture(scope="session")
def model():
    return load_model(device="cpu", seed=0)


@pytest.fixture(scope="session")
def pair():
    return PromptPair(CLEAN, CORRUPTED, " Paris", " Rome")


@pytest.fixture(scope="session")
def tokens(model):
    clean, corrupted = to_tokens(model, CLEAN), to_tokens(model, CORRUPTED)
    assert clean.shape == corrupted.shape
    return clean, corrupted


@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield
