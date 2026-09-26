from types import SimpleNamespace

import pytest

from agent_eval import format_time
from config import (LATEST_OBS_VERSION, MODEL_PATHS, TEST_TRACKS, TRACK_IDS, TRAIN_TRACKS,
                    VAL_TRACKS, model_path_for, obs_version_for, track_split)


def test_splits_are_disjoint_and_cover_every_track():
    train, val, test = set(TRAIN_TRACKS), set(VAL_TRACKS), set(TEST_TRACKS)
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(TRACK_IDS)
    assert track_split("jp-1962") == "test"  # Suzuka is the held-out circuit
    for t in VAL_TRACKS:
        assert track_split(t) == "val"


def test_model_path_for_resolves_keys_and_passes_paths_through():
    assert model_path_for("v15b") == MODEL_PATHS["v15b"]
    assert model_path_for("some/dir/model.zip") == "some/dir/model.zip"


def fake_model(size):
    return SimpleNamespace(observation_space=SimpleNamespace(shape=(size,)))


def test_obs_version_order_of_precedence():
    assert obs_version_for("v13", fake_model(65), override=3) == 3  # explicit override wins
    assert obs_version_for("v13", fake_model(65)) == 1              # registry beats input size
    assert obs_version_for("x.zip", fake_model(68)) == 3            # input size for unregistered paths
    assert obs_version_for("x.zip", fake_model(65)) == 2            # 65 is never guessed as version 1
    assert obs_version_for("x.zip", None) == LATEST_OBS_VERSION


@pytest.mark.parametrize("seconds, text", [(None, "DNF"), (87.9, "1:27.900"), (62.05, "1:02.050"), (59.5, "0:59.500")])
def test_format_time(seconds, text):
    assert format_time(seconds) == text


def test_default_results_path_names_file_after_model():
    from agent_eval import default_results_path
    assert default_results_path("v14") == "results/v14.csv"
    assert default_results_path("./models/sac_v16/best_model/best_model.zip") == "results/sac_v16.csv"
    assert default_results_path("ckpts/sac_racing_2000000_steps.zip") == "results/sac_racing_2000000_steps.zip"[:-4] + ".csv"
