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
    3. Traversal from this file's location: tools/stockfish-mini-policy-NN-prototype/../../
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

DEFAULT_REL_PATHS = {
    "stockfish_bin": os.path.join("Stockfish", "src", "stockfish"),
    "monty_bin": os.path.join("tools", "Monty", "monty"),
    "epd_file": os.path.join("scratch", "popular_ajedrezdata_maxPlies60.epd.scored"),
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
    Resolves a file/directory path with multi-tier automatic discovery:
    1. paths_config.json custom override (if file exists).
    2. Internal standalone deps (e.g. deps/Stockfish/src/stockfish, deps/Monty/monty).
    3. Monorepo sibling paths (e.g. Stockfish/src/stockfish, tools/Monty/monty).
    4. Sane default path.
    """
    cfg_val = _LOADED_CONFIG.get(key)
    if cfg_val:
        candidates = [
            cfg_val if os.path.isabs(cfg_val) else "",
            os.path.abspath(os.path.join(REPO_ROOT, cfg_val)),
            os.path.abspath(os.path.join(THIS_DIR, cfg_val)),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c

    if key == "stockfish_bin":
        candidates = [
            os.path.abspath(os.path.join(THIS_DIR, "deps", "Stockfish", "src", "stockfish")),
            os.path.abspath(os.path.join(THIS_DIR, "Stockfish", "src", "stockfish")),
            os.path.abspath(os.path.join(REPO_ROOT, "Stockfish", "src", "stockfish")),
            os.path.abspath(os.path.join(REPO_ROOT, "deps", "Stockfish", "src", "stockfish")),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    elif key == "monty_bin":
        candidates = [
            os.path.abspath(os.path.join(THIS_DIR, "deps", "Monty", "monty")),
            os.path.abspath(os.path.join(THIS_DIR, "Monty", "monty")),
            os.path.abspath(os.path.join(THIS_DIR, "tools", "Monty", "monty")),
            os.path.abspath(os.path.join(REPO_ROOT, "tools", "Monty", "monty")),
            os.path.abspath(os.path.join(REPO_ROOT, "Monty", "monty")),
            os.path.abspath(os.path.join(REPO_ROOT, "deps", "Monty", "monty")),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    elif key == "calibration_config":
        candidates = [
            os.path.abspath(os.path.join(THIS_DIR, "calibration_config.json")),
            os.path.abspath(os.path.join(REPO_ROOT, "tools", "stockfish-mini-policy-NN-prototype", "calibration_config.json")),
            os.path.abspath(os.path.join(REPO_ROOT, "calibration_config.json")),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    elif key == "epd_file":
        candidates = [
            os.path.abspath(os.path.join(THIS_DIR, "data", "popular_ajedrezdata_maxPlies60.epd.scored")),
            os.path.abspath(os.path.join(THIS_DIR, "scratch", "popular_ajedrezdata_maxPlies60.epd.scored")),
            os.path.abspath(os.path.join(REPO_ROOT, "scratch", "popular_ajedrezdata_maxPlies60.epd.scored")),
            os.path.abspath(os.path.join(REPO_ROOT, "data", "popular_ajedrezdata_maxPlies60.epd.scored")),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    elif key == "cache_dir":
        candidates = [
            os.path.abspath(os.path.join(THIS_DIR, "scratch")),
            os.path.abspath(os.path.join(REPO_ROOT, "scratch")),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return os.path.abspath(os.path.join(THIS_DIR, "scratch"))

    monorepo_rel = default_rel or DEFAULT_REL_PATHS.get(key, "")
    return os.path.abspath(os.path.join(REPO_ROOT, monorepo_rel))


# Standardized Resolved Paths
STOCKFISH_BIN = resolve_path("stockfish_bin")
MONTY_BIN = resolve_path("monty_bin")
EPD_FILE = resolve_path("epd_file")
CACHE_DIR = resolve_path("cache_dir")
CALIB_CONFIG_PATH = resolve_path("calibration_config")
