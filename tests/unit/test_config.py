"""Config precedence is defaults -> settings.yaml -> env, and booleans get an
explicit coercion pass. Both YAML and env vars can hand us the *string*
"false", which is truthy in Python — a config layer that flips a run onto the
paid production stack because of that is worse than no config layer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from reconciler.config import ConfigError, Settings, load_settings


def _yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _clean_recon_env(monkeypatch):
    """CI sets RECON_USE_REAL_LLM/RECON_USE_REAL_EMBEDDINGS globally so the
    rest of the suite runs the offline stack. Left in place, those ambient
    vars would silently outrank the YAML values these tests set, since env
    beats YAML by design. Tests that care about env precedence set it back
    explicitly via monkeypatch.setenv."""
    for key in (
        "RECON_USE_REAL_LLM",
        "RECON_USE_REAL_EMBEDDINGS",
        "RECON_LLM_MODEL",
        "RECON_EMBEDDING_MODEL",
        "RECON_DATA_RAW_DIR",
        "RECON_DATA_PROCESSED_DIR",
        "RECON_DATA_OUTPUTS_DIR",
    ):
        monkeypatch.delenv(key, raising=False)


def test_defaults_are_offline_safe():
    """The default stack must never require an API key: a fresh clone should
    run green with no credentials and no network."""
    settings = Settings()
    assert settings.use_real_llm is False


def test_yaml_overrides_defaults(tmp_path):
    settings = load_settings(_yaml(tmp_path, "llm_model: some-other-model\nuse_real_llm: true\n"))
    assert settings.llm_model == "some-other-model"
    assert settings.use_real_llm is True


def test_env_beats_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("RECON_USE_REAL_LLM", "false")
    settings = load_settings(_yaml(tmp_path, "use_real_llm: true\n"))
    assert settings.use_real_llm is False


@pytest.mark.parametrize("raw", ["false", "False", "FALSE", "0", "no", "off"])
def test_falsey_strings_are_not_truthy(tmp_path, monkeypatch, raw):
    """Regression guard: `bool("false")` is True."""
    monkeypatch.setenv("RECON_USE_REAL_LLM", raw)
    assert load_settings(_yaml(tmp_path, "")).use_real_llm is False


@pytest.mark.parametrize("raw", ["true", "True", "1", "yes", "on"])
def test_truthy_strings_are_accepted(tmp_path, monkeypatch, raw):
    monkeypatch.setenv("RECON_USE_REAL_LLM", raw)
    assert load_settings(_yaml(tmp_path, "")).use_real_llm is True


def test_uninterpretable_boolean_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(_yaml(tmp_path, "use_real_llm: maybe\n"))


def test_unknown_key_is_ignored_but_warned(tmp_path, caplog):
    """A typo'd key that is silently dropped looks exactly like a setting that
    did not take effect, so it has to be audible."""
    with caplog.at_level("WARNING"):
        load_settings(_yaml(tmp_path, "use_reel_llm: true\n"))
    assert "use_reel_llm" in caplog.text


def test_malformed_yaml_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(_yaml(tmp_path, "this: [is: not: valid\n"))


def test_non_mapping_yaml_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(_yaml(tmp_path, "- just\n- a\n- list\n"))


def test_missing_settings_file_falls_back_to_defaults(tmp_path):
    settings = load_settings(tmp_path / "absent.yaml")
    assert settings.llm_model == Settings().llm_model


def test_paths_can_be_overridden_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RECON_DATA_RAW_DIR", str(tmp_path / "elsewhere"))
    settings = load_settings(tmp_path / "absent.yaml")
    assert settings.data_raw_dir == tmp_path / "elsewhere"


def test_stack_description_names_the_backends_actually_in_use(tmp_path):
    offline = load_settings(_yaml(tmp_path, "use_real_llm: false\nuse_real_embeddings: false\n"))
    assert "offline" in offline.stack_description.lower()

    online = load_settings(_yaml(tmp_path, "use_real_llm: true\nuse_real_embeddings: true\n"))
    assert online.llm_model in online.stack_description
