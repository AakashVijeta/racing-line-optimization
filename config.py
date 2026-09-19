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

# --- Model Paths ---

MODEL_PATHS = {
    "v10": "./models/sac_v10/best_model/best_model.zip",
    "v11": "./models/sac_v11/best_model/best_model.zip",
    "v12": "./models/sac_v12/best_model/best_model.zip",
}

DEFAULT_MODEL_VERSION = "v12"

def get_model_path(version=None):
    """Get the model path for a given version string."""
    version = version or DEFAULT_MODEL_VERSION
    if version not in MODEL_PATHS:
        raise ValueError(f"Unknown model version '{version}'. Available: {list(MODEL_PATHS.keys())}")
    return MODEL_PATHS[version]
