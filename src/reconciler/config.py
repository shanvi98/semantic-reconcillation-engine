"""Central, overridable settings.

Precedence, lowest to highest: dataclass defaults -> config/settings.yaml -> env vars.

Booleans get an explicit coercion pass because both YAML and env vars can hand
us the *string* "false", which is truthy in Python — silently flipping a run
onto the paid production stack is exactly the kind of surprise a config layer
exists to prevent.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = ROOT / "config" / "settings.yaml"

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off", "", "none"}


class ConfigError(ValueError):
    """Raised when settings.yaml is malformed or holds an uninterpretable value."""


def _as_bool(value, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{key}={value!r} is not a boolean — use true/false")


@dataclass
class Settings:
    data_raw_dir: Path = ROOT / "data" / "raw"
    data_processed_dir: Path = ROOT / "data" / "processed"
    data_outputs_dir: Path = ROOT / "data" / "outputs"
    llm_model: str = "claude-haiku-4-5-20251001"
    use_real_llm: bool = False  # False -> HeuristicLLMClient (offline, deterministic)
    embedding_model: str = "all-MiniLM-L6-v2"
    use_real_embeddings: bool = True  # False -> HashingEmbedder (offline, deterministic)

    @property
    def stack_description(self) -> str:
        """One-line summary of which backends this run will actually use."""
        embedder = self.embedding_model if self.use_real_embeddings else "HashingEmbedder (offline)"
        adjudicator = self.llm_model if self.use_real_llm else "HeuristicLLMClient (offline)"
        return f"embeddings={embedder} · adjudicator={adjudicator}"


_BOOL_FIELDS = {"use_real_llm", "use_real_embeddings"}
_PATH_FIELDS = {"data_raw_dir", "data_processed_dir", "data_outputs_dir"}


def load_settings(settings_path: Path | None = None) -> Settings:
    """Build Settings from defaults, then settings.yaml, then environment."""
    settings = Settings()
    path = settings_path or _SETTINGS_PATH
    known = {f.name for f in fields(Settings)}

    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a mapping, got {type(raw).__name__}")

        for key, value in raw.items():
            if key not in known:
                # Loud, because a typo'd key that is silently ignored looks
                # exactly like a setting that did not take effect.
                logger.warning("%s: ignoring unknown setting %r", path, key)
                continue
            if key in _BOOL_FIELDS:
                value = _as_bool(value, key=key)
            elif key in _PATH_FIELDS:
                value = Path(value)
            setattr(settings, key, value)

    env_overrides = {
        "use_real_llm": "RECON_USE_REAL_LLM",
        "use_real_embeddings": "RECON_USE_REAL_EMBEDDINGS",
        "llm_model": "RECON_LLM_MODEL",
        "embedding_model": "RECON_EMBEDDING_MODEL",
        "data_raw_dir": "RECON_DATA_RAW_DIR",
        "data_processed_dir": "RECON_DATA_PROCESSED_DIR",
        "data_outputs_dir": "RECON_DATA_OUTPUTS_DIR",
    }
    for key, env_var in env_overrides.items():
        if env_var not in os.environ:
            continue
        value = os.environ[env_var]
        if key in _BOOL_FIELDS:
            value = _as_bool(value, key=env_var)
        elif key in _PATH_FIELDS:
            value = Path(value)
        setattr(settings, key, value)

    return settings
