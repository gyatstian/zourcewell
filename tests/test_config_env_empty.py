"""A blank ``ZOURCEWELL_*`` var counts as unset: config.json decides.

Reproduced before the fix, with a config.json holding the file value but an
empty env var exported (exactly the shape `set -a; . .env` produces, and
``.env.example`` shows empties as the keep-config idiom):

  ``ZOURCEWELL_STREAM_MODE=`` turned ``stream_mode: buffered`` into "live",
  ``ZOURCEWELL_PORT=`` turned 5123 into 5001, ``ZOURCEWELL_BROWSER_BIN=``
  turned a pinned Chrome path into "" (which disables the real-browser login),
  and ``ZOURCEWELL_DATA=`` made ``config.DATA_DIR`` "." -- the working
  directory. Same for chat_model / enable_tools / log_level / provider /
  active_account / append_top / max_retries and the ``providers`` list.

The rule (documented on ``config._env_str``) is that an empty override is
indistinguishable from an accidental one, so it is treated as absent. Every
test here also asserts the same var with a real value still overrides, so the
fix cannot be "ignore the environment".
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zourcewell import config as config_mod
from zourcewell.config import load_settings

REPO_ROOT = Path(__file__).resolve().parent.parent

# (field, env key, config value). Each config value differs from the field's
# built-in default where the type allows it, so a test can tell "config.json
# won" from "the default won". Booleans have only two values: enable_tools is
# True (its default), which is exactly the case the old empty-env mapping was
# wrong about (``_as_bool("")`` is False), and the override test below pins the
# other direction.
_BLANK_CASES = [
    ("port", "ZOURCEWELL_PORT", 5123),
    ("stream_mode", "ZOURCEWELL_STREAM_MODE", "buffered"),
    ("chat_model", "ZOURCEWELL_CHAT_MODEL", "cfg-model"),
    ("enable_tools", "ZOURCEWELL_ENABLE_TOOLS", True),
    ("log_level", "ZOURCEWELL_LOG_LEVEL", "DEBUG"),
    ("provider", "ZOURCEWELL_PROVIDER", "qwen"),
    ("active_account", "ZOURCEWELL_ACTIVE_ACCOUNT", "cfg@x.com"),
    ("browser_bin", "ZOURCEWELL_BROWSER_BIN", "C:/cfg/chrome.exe"),
    ("append_top", "ZOURCEWELL_APPEND_TOP", "CFG-TOP"),
    ("max_retries", "ZOURCEWELL_MAX_RETRIES", 4),
    ("obscura_bin", "OBSCURA_BIN", "C:/cfg/obscura.exe"),
    ("obscura_profile", "OBSCURA_PROFILE", "C:/cfg/profile"),
]

# (field, env key, config value, env value, expected). The env value differs
# from both the config value and the default, so "the env won" is real.
_OVERRIDE_CASES = [
    ("port", "ZOURCEWELL_PORT", 5123, "6123", 6123),
    ("stream_mode", "ZOURCEWELL_STREAM_MODE", "live", "buffered", "buffered"),
    ("chat_model", "ZOURCEWELL_CHAT_MODEL", "cfg-model", "env-model", "env-model"),
    ("enable_tools", "ZOURCEWELL_ENABLE_TOOLS", True, "0", False),
    ("log_level", "ZOURCEWELL_LOG_LEVEL", "DEBUG", "ERROR", "ERROR"),
    ("provider", "ZOURCEWELL_PROVIDER", "qwen", "opencode", "opencode"),
    ("active_account", "ZOURCEWELL_ACTIVE_ACCOUNT", "cfg@x.com", "env@x.com", "env@x.com"),
    ("browser_bin", "ZOURCEWELL_BROWSER_BIN", "C:/cfg/chrome.exe",
     "D:/env/chrome.exe", "D:/env/chrome.exe"),
    ("append_top", "ZOURCEWELL_APPEND_TOP", "CFG-TOP", "ENV-TOP", "ENV-TOP"),
    ("max_retries", "ZOURCEWELL_MAX_RETRIES", 2, "3", 3),
]

# "" and "   " are the same thing to a shell: a present variable with no value.
_BLANKS = ("", "   ")


def _cfg(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("field,env_key,cfg_value", _BLANK_CASES)
def test_blank_env_keeps_the_config_value(field, env_key, cfg_value,
                                         monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, {field: cfg_value})
    for blank in _BLANKS:
        monkeypatch.setenv(env_key, blank)
        assert getattr(load_settings(cfg), field) == cfg_value


@pytest.mark.parametrize("field,env_key,cfg_value,env_value,expected", _OVERRIDE_CASES)
def test_real_env_value_still_overrides(field, env_key, cfg_value, env_value,
                                        expected, monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, {field: cfg_value})
    monkeypatch.setenv(env_key, env_value)
    assert getattr(load_settings(cfg), field) == expected


def test_blank_browser_bin_keeps_the_auto_default(monkeypatch, tmp_path):
    """A fresh config has no ``browser_bin`` key and loads as "auto"; a blank
    env var must not turn that into "" (login disabled)."""
    cfg = _cfg(tmp_path, {"keys": []})
    for blank in _BLANKS:
        monkeypatch.setenv("ZOURCEWELL_BROWSER_BIN", blank)
        assert load_settings(cfg).browser_bin == "auto"


# --- fields the rule already covered: keep them that way --------------------


def test_blank_keys_env_keeps_the_config_keys(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, {"keys": ["filekey"]})
    for blank in _BLANKS:
        monkeypatch.setenv("ZOURCEWELL_KEYS", blank)
        assert load_settings(cfg).keys == ["filekey"]


@pytest.mark.parametrize("blank", _BLANKS)
def test_blank_keys_env_is_unset_not_junk(blank, monkeypatch, tmp_path, caplog):
    """A blank ZOURCEWELL_KEYS must not fire the "no usable key" warning:
    there is no override to complain about (the ZOURCEWELL_KEYS="," trap
    still warns -- see the dedicated test in the config-integrity suite)."""
    cfg = _cfg(tmp_path, {"keys": ["filekey"]})
    monkeypatch.setenv("ZOURCEWELL_KEYS", blank)
    with caplog.at_level(logging.WARNING, logger="zourcewell.config"):
        assert load_settings(cfg).keys == ["filekey"]
    assert "contains no usable key" not in caplog.text


def test_blank_providers_env_keeps_the_config_list(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, {"providers": ["deepseek", "qwen"]})
    for blank in _BLANKS:
        monkeypatch.setenv("ZOURCEWELL_PROVIDERS", blank)
        assert load_settings(cfg).providers == ["deepseek", "qwen"]
    monkeypatch.setenv("ZOURCEWELL_PROVIDERS", "qwen")
    assert load_settings(cfg).providers == ["qwen"]


@pytest.mark.parametrize("blank", _BLANKS)
def test_blank_config_env_still_resolves_the_cwd_file(blank, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ZOURCEWELL_CONFIG", blank)
    assert config_mod._default_config_path() == tmp_path / "config.json"


def test_real_config_env_still_overrides(monkeypatch, tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ZOURCEWELL_CONFIG", str(target))
    assert config_mod._default_config_path() == target


# --- the account env vars are alias pairs ----------------------------------
#
# ZOURCEWELL_TOKEN / EMAIL / PASSWORD fall back to the DEEPSEEK_* aliases. A
# whitespace-only value used to shadow the alias (or, with no alias, mint an
# account out of nothing: ZOURCEWELL_TOKEN="   " appended an empty account to
# a config that already had one, and ZOURCEWELL_EMAIL="  " produced an email
# of two spaces).


def test_blank_account_env_vars_do_not_mint_an_account(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, {"accounts": [{"email": "cfg@x.com", "token": "t" * 32}]})
    for var in ("ZOURCEWELL_TOKEN", "DEEPSEEK_TOKEN", "ZOURCEWELL_EMAIL",
                "DEEPSEEK_EMAIL", "ZOURCEWELL_PASSWORD", "DEEPSEEK_PASSWORD"):
        monkeypatch.setenv(var, "   ")
    s = load_settings(cfg)
    assert [a.email for a in s.accounts] == ["cfg@x.com"]


def test_whitespace_account_env_falls_through_to_the_alias(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, {})
    monkeypatch.setenv("ZOURCEWELL_EMAIL", "  ")
    monkeypatch.setenv("ZOURCEWELL_PASSWORD", "  ")
    monkeypatch.setenv("DEEPSEEK_EMAIL", "alias@x.com")
    monkeypatch.setenv("DEEPSEEK_PASSWORD", "pw")
    monkeypatch.delenv("ZOURCEWELL_TOKEN", raising=False)
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    s = load_settings(cfg)
    assert [(a.email, a.password) for a in s.accounts] == [("alias@x.com", "pw")]


def test_blank_provider_env_falls_back_to_the_config_provider(monkeypatch, tmp_path):
    """A real env token plus a blank ZOURCEWELL_PROVIDER must bind the account
    to the config's provider, not to AccountConfig's deepseek default."""
    cfg = _cfg(tmp_path, {"provider": "qwen"})
    monkeypatch.setenv("ZOURCEWELL_PROVIDER", "   ")
    monkeypatch.setenv("ZOURCEWELL_TOKEN", "t" * 32)
    monkeypatch.delenv("DEEPSEEK_TOKEN", raising=False)
    s = load_settings(cfg)
    assert [(a.token, a.provider) for a in s.accounts] == [("t" * 32, "qwen")]


def test_blank_port_env_keeps_the_port_zero_guard(monkeypatch, tmp_path):
    """``"port": 0`` warns and runs as DEFAULT_PORT, and an unrelated save must
    write the operator's own 0 back. A blank ZOURCEWELL_PORT used to
    neutralize that clamp bookkeeping, so the save rewrote 0 as 5001."""
    cfg = _cfg(tmp_path, {"port": 0})
    monkeypatch.setenv("ZOURCEWELL_PORT", "   ")
    s = load_settings(cfg)
    assert s.port == config_mod.DEFAULT_PORT
    s.save()
    assert json.loads(Path(cfg).read_text(encoding="utf-8"))["port"] == 0


# --- ZOURCEWELL_DATA is an import-time constant ----------------------------
#
# ``DATA_DIR`` is resolved once at import, so only a fresh process can show
# what an empty value does to it. Before the fix ``Path("")`` was ``Path(".")``
# and DATA_DIR became the process cwd -- the PoW wasm cache, the opencode
# managed dir and anything else keyed off DATA_DIR then landed wherever the
# server happened to be started from.


def _data_dir_from_a_fresh_interpreter(tmp_path: Path, value: str) -> tuple[Path, Path]:
    """``(DATA_DIR, the config.py that produced it)``.

    The module path is checked too: conftest's source-tree guard covers this
    process, not a subprocess, and a stale editable install would otherwise
    make the verdict be about code nobody edited.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("ZOURCEWELL_", "DEEPSEEK_", "OBSCURA_"))}
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["ZOURCEWELL_DATA"] = value
    proc = subprocess.run(
        [sys.executable, "-c",
         "import zourcewell.config as c; print(c.DATA_DIR); print(c.__file__)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    data_dir, module_file = proc.stdout.strip().splitlines()
    return Path(data_dir), Path(module_file).resolve()


@pytest.mark.parametrize("blank", _BLANKS)
def test_blank_data_env_keeps_the_builtin_default(blank, tmp_path):
    data_dir, module_file = _data_dir_from_a_fresh_interpreter(tmp_path, blank)
    assert module_file == REPO_ROOT / "src" / "zourcewell" / "config.py"
    assert data_dir.is_absolute()
    assert data_dir == REPO_ROOT / "data"


def test_real_data_env_still_overrides(tmp_path):
    target = tmp_path / "elsewhere"
    data_dir, _ = _data_dir_from_a_fresh_interpreter(tmp_path, str(target))
    assert data_dir == target
