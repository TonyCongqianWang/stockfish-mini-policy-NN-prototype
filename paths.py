"""
Path Configuration & Resolution Module for Mini-NN Toolchain.
Enables portable execution across local and remote machines:
1. Resolves paths from `paths_config.json` if available.
2. Supports standalone repository deployments with internal `deps/` directory (e.g., `deps/Stockfish`, `deps/Monty`).
3. Falls back to parent monorepo root (expected to mirror `~/projects/repos/stockfish`).
4. Supports environment variable overrides (e.g., SF_REPO_ROOT, SF_PATHS_CONFIG).
"""

import json
import os
from typing import Dict, Optional

THIS_DIR = os.path.abspath(os.path.dirname(__file__))


def get_repo_root() -> str:
    """
    Identifies the base repository root directory:
    1. SF_REPO_ROOT environment variable (if set).
    2. Current working directory if it contains 'Stockfish' or 'tools'.
    3. Traversal from this file's location: tools/mini-NN-lmr/../../
    4. Fallback to THIS_DIR if running standalone.
    """
    if "SF_REPO_ROOT" in os.environ and os.path.isdir(os.environ["SF_REPO_ROOT"]):
        return os.path.abspath(os.environ["SF_REPO_ROOT"])

    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "Stockfish")) or os.path.isdir(os.path.join(cwd, "tools")):
        return os.path.abspath(cwd)

    parent_monorepo = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
    if os.path.isdir(os.path.join(parent_monorepo, "Stockfish")):
        return parent_monorepo

    return THIS_DIR


REPO_ROOT = get_repo_root()

# Primary monorepo relative paths
DEFAULT_REL_PATHS = {
    "stockfish_bin": os.path.join("Stockfish", "src", "stockfish"),
    "monty_bin": os.path.join("tools", "Monty", "monty"),
    "epd_file": os.path.join("scratch", "popular_ajedrezdata_maxPlies60.epd.scored"),
    "cache_dir": "scratch",
    "calibration_config": os.path.join("tools", "stockfish-mini-policy-NN-prototype", "calibration_config.json")
}

# Standalone internal deps relative paths (inside stockfish-mini-policy-NN-prototype repo)
INTERNAL_REL_PATHS = {
    "stockfish_bin": os.path.join("deps", "Stockfish", "src", "stockfish"),
    "monty_bin": os.path.join("deps", "Monty", "monty"),
    "epd_file": os.path.join("data", "popular_ajedrezdata_maxPlies60.epd.scored"),
    "cache_dir": "scratch",
    "calibration_config": "calibration_config.json"
}


def find_paths_config() -> Optional[str]:
    """
    Finds the paths_config.json file location.
    """
    if "SF_PATHS_CONFIG" in os.environ and os.path.isfile(os.environ["SF_PATHS_CONFIG"]):
        return os.path.abspath(os.environ["SF_PATHS_CONFIG"])

    candidates = [
        os.path.join(THIS_DIR, "paths_config.json"),
        os.path.join(REPO_ROOT, "paths_config.json"),
        os.path.join(THIS_DIR, "paths.json"),
        os.path.join(REPO_ROOT, "paths.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


PATHS_CONFIG_PATH = find_paths_config()


def load_paths_config() -> Dict[str, str]:
    if PATHS_CONFIG_PATH and os.path.exists(PATHS_CONFIG_PATH):
        try:
            with open(PATHS_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_LOADED_CONFIG = load_paths_config()


def resolve_path(key: str, default_rel: Optional[str] = None) -> str:
    """
    Resolves a file/directory path:
    1. Checks if `key` is present in paths_config.json.
    2. If present and absolute -> uses it if exists, else checks relative.
    3. Checks internal `deps/` path (inside this repo).
    4. Checks monorepo fallback relative path from REPO_ROOT.
    """
    monorepo_rel = default_rel or DEFAULT_REL_PATHS.get(key, "")
    monorepo_abs = os.path.abspath(os.path.join(REPO_ROOT, monorepo_rel)) if monorepo_rel else ""

    internal_rel = INTERNAL_REL_PATHS.get(key, "")
    internal_abs = os.path.abspath(os.path.join(THIS_DIR, internal_rel)) if internal_rel else ""

    cfg_val = _LOADED_CONFIG.get(key)
    if cfg_val:
        if os.path.isabs(cfg_val):
            if os.path.exists(cfg_val):
                return cfg_val
            if monorepo_abs and os.path.exists(monorepo_abs):
                return monorepo_abs
            if internal_abs and os.path.exists(internal_abs):
                return internal_abs
            return cfg_val
        else:
            # Check relative to REPO_ROOT or THIS_DIR
            c1 = os.path.abspath(os.path.join(REPO_ROOT, cfg_val))
            if os.path.exists(c1):
                return c1
            c2 = os.path.abspath(os.path.join(THIS_DIR, cfg_val))
            if os.path.exists(c2):
                return c2
            return c1

    # Check existence in monorepo vs internal deps
    if monorepo_abs and os.path.exists(monorepo_abs):
        return monorepo_abs
    if internal_abs and os.path.exists(internal_abs):
        return internal_abs

    return monorepo_abs or internal_abs


# Standardized Resolved Paths
STOCKFISH_BIN = resolve_path("stockfish_bin")
MONTY_BIN = resolve_path("monty_bin")
EPD_FILE = resolve_path("epd_file")
CACHE_DIR = resolve_path("cache_dir")
CALIB_CONFIG_PATH = resolve_path("calibration_config")
