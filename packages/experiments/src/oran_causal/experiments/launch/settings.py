"""Launch settings: SSH hosts, their base directories and the Kaggle account.

Each value is read from the environment when a launcher needs it. Hosts and the Kaggle owner have
no default; base directories and the Kaggle CLI do.
"""

import os

VPS_HOST = "ORAN_CAUSAL_VPS_HOST"
INSTABOX_HOST = "ORAN_CAUSAL_INSTABOX_HOST"
VPS_BASE = "ORAN_CAUSAL_VPS_BASE"
INSTABOX_BASE = "ORAN_CAUSAL_INSTABOX_BASE"
KAGGLE_OWNER = "ORAN_CAUSAL_KAGGLE_OWNER"
KAGGLE_CLI = "ORAN_CAUSAL_KAGGLE_CLI"

DEFAULT_VPS_BASE = "/opt/oran-causal"
DEFAULT_INSTABOX_BASE = "/data/home/oran-causal"
DEFAULT_KAGGLE_CLI = "kaggle"
BASE_DEFAULTS = {VPS_BASE: DEFAULT_VPS_BASE, INSTABOX_BASE: DEFAULT_INSTABOX_BASE}


class MissingSetting(RuntimeError):
    """A required launch setting is not set."""


def required(name: str) -> str:
    """The value of environment variable `name`.

    Raises:
        MissingSetting: When it is unset or empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSetting(f"set the environment variable {name}")
    return value


def optional(name: str, default: str) -> str:
    """The value of environment variable `name`, or `default` when it is unset or empty."""
    return os.environ.get(name, "").strip() or default


def base(name: str) -> str:
    """A base directory setting (`VPS_BASE` or `INSTABOX_BASE`), or its default."""
    return optional(name, BASE_DEFAULTS[name])


def vps_base() -> str:
    """Directory every remote file of a VPS job lives under."""
    return base(VPS_BASE)


def instabox_base() -> str:
    """Directory every remote file of an instabox job lives under."""
    return base(INSTABOX_BASE)


def kaggle_owner() -> str:
    """Kaggle account the code datasets and kernels belong to."""
    return required(KAGGLE_OWNER)


def kaggle_cli() -> str:
    """The `kaggle` executable: the setting, else `kaggle` on PATH."""
    return optional(KAGGLE_CLI, DEFAULT_KAGGLE_CLI)
