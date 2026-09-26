"""
Shared configuration for the racing line optimization project.
Single source of truth for track IDs, widths, model paths, and defaults.
"""

# --- Track Configuration ---

TRACK_IDS = [
    "mc-1929", "az-2016", "nl-1948", "it-1953", "hu-1986", "sg-2008",
    "sa-2021", "pt-1972", "fr-1960", "au-1953", "ca-1978", "at-1969",
    "jp-1962", "mx-1962", "br-1940", "br-1977", "ar-1952", "za-1961",
    "us-1956", "us-2022", "us-2023", "es-2026", "de-1927", "de-1932",
    "it-1914", "be-1925", "it-1922", "es-1991", "ae-2009", "pt-2008",
    "tr-2005", "ru-2014", "qa-2004", "us-1909", "gb-1948", "bh-2002",
    "cn-2004", "us-2012", "my-1999", "fr-1969"
]

TRACK_WIDTHS = {
    "mc-1929": 8.5, "az-2016": 10.0, "nl-1948": 10.0,
    "it-1953": 11.0, "hu-1986": 11.0, "sg-2008": 11.0, "sa-2021": 11.0, "pt-1972": 11.0,
    "fr-1960": 11.5, "au-1953": 12.0, "ca-1978": 12.0, "at-1969": 12.0, "jp-1962": 12.0,
    "mx-1962": 12.0, "br-1940": 12.0, "br-1977": 12.0, "ar-1952": 12.0, "za-1961": 12.0,
    "us-1956": 12.0, "us-2022": 12.0, "us-2023": 12.0, "es-2026": 12.0,
    "de-1927": 13.0, "de-1932": 13.0, "it-1914": 13.0,
    "be-1925": 14.0, "it-1922": 14.0, "es-1991": 14.0, "ae-2009": 14.0, "pt-2008": 14.0,
    "tr-2005": 14.0, "ru-2014": 14.0, "qa-2004": 14.0, "us-1909": 14.0,
    "gb-1948": 15.0, "bh-2002": 15.0, "cn-2004": 15.0, "us-2012": 15.0, "my-1999": 15.0,
    "fr-1969": 15.0
}

DEFAULT_TRACK_WIDTH = 15.0

# Held out of training. Validation tracks drive best-model selection during
# training, so only the test track gives an unbiased generalization result.
VAL_TRACKS = ["mc-1929", "be-1925"]   # Monaco, Spa
TEST_TRACKS = ["jp-1962"]             # Suzuka
TRAIN_TRACKS = [t for t in TRACK_IDS if t not in VAL_TRACKS + TEST_TRACKS]


# Procedural tracks (track_generator.py). Fixed seeds so every run and every
# evaluation sees the same pools. Test tracks are never used for training or
# checkpoint selection.
PROC_SEEDS = {"train": 100, "val": 200, "test": 300}
PROC_SIZES = {"train": 400, "val": 16, "test": 50}


def track_split(track_id):
    """Return 'test', 'val' or 'train' for a track ID (real or procedural)."""
    if track_id.startswith("proc-"):
        seed = int(track_id.split("-")[1])
        return {v: k for k, v in PROC_SEEDS.items()}.get(seed, "train")
    if track_id in TEST_TRACKS:
        return "test"
    if track_id in VAL_TRACKS:
        return "val"
    return "train"

# --- Model Paths ---

MODEL_PATHS = {
    "v10": "./models/sac_v10/best_model/best_model.zip",
    "v11": "./models/sac_v11/best_model/best_model.zip",
    "v12": "./models/sac_v12/best_model/best_model.zip",
    "v13": "./models/sac_v13/best_model/best_model.zip",
    # Replace with the top pick from `python checkpoint_sweep.py --run sac_v14`
    "v14": "./models/sac_v14/best_model/best_model.zip",
    # Best validation checkpoint, at 2.0M steps (the run was stopped at 3.6M)
    "v15b": "./models/sac_v15b/best_model/best_model.zip",
}

DEFAULT_MODEL_VERSION = "v15b"

# Observation layout each model was trained with (see RacingEnv's obs_version):
# 1 = vertex-snapped lookahead/boundaries, 2 = interpolated/segment-accurate,
# 3 = adds track width, previous action and signed curvature.
OBS_VERSIONS = {"v10": 1, "v11": 1, "v12": 1, "v13": 1, "v14": 2, "v15b": 3}
LATEST_OBS_VERSION = 3

def get_model_path(version=None):
    """Get the model path for a given version string."""
    version = version or DEFAULT_MODEL_VERSION
    if version not in MODEL_PATHS:
        raise ValueError(f"Unknown model version '{version}'. Available: {list(MODEL_PATHS.keys())}")
    return MODEL_PATHS[version]


def model_path_for(version):
    """Resolve a version key from MODEL_PATHS, or treat the argument as a path."""
    return get_model_path(version) if version in MODEL_PATHS or version is None else version


def obs_version_for(version=None, model=None, override=None):
    """Observation version for a model key or loaded model.

    Order: explicit override, then the OBS_VERSIONS registry, then the loaded
    model's input size (68 -> 3, 65 -> 2; version 1 is never guessed, since
    it has the same size as 2), then the latest version.
    """
    if override is not None:
        return override
    key = version or DEFAULT_MODEL_VERSION
    if key in OBS_VERSIONS:
        return OBS_VERSIONS[key]
    if model is not None:
        size = model.observation_space.shape[0]
        return {68: 3, 65: 2}.get(size, LATEST_OBS_VERSION)
    return LATEST_OBS_VERSION
