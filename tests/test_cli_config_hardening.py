"""Repair P3: CLI secret intake, config clamp persistence, TUI keys, provider fallback.

Four independent regressions, one file, each with the pre-change failure named
in the test docstring. The tree under test is this checkout's ``src/`` (see
``tests/conftest.py``: ``python -m pytest`` from the repo root).

* **P3-1** ``cli._secret_from_arg`` returned ``""`` for a non-tty stdin, so
  ``deepseaport login --email a@x.com < password.txt`` (service wrappers, CI,
  any piped secret) authenticated with an empty password. The pre-change
  behaviour was ``getpass.getpass`` -> ``getpass.fallback_getpass``, which
  reads ONE LINE from a non-tty stdin. The helper's docstring claimed scripted
  behaviour was unchanged; it was the opposite.
* **P3-2** ``load_settings`` clamped ``max_retries`` to ``MAX_RETRIES_LIMIT`` in
  memory only. ``Settings.save()`` serialised the clamped value, and save() runs
  on every account add/select/unblock/token-refresh and every TUI toggle, so
  ``"max_retries": 9`` became a permanent ``5`` on disk after one unrelated
  action. This checkout's own config.json holds 7.
* **P3-3** the TUI key editor's two ``settings.keys = keys`` assignments were the
  only writer of the key list that did NOT go through ``config._normalize_keys``,
  while ``load_settings`` does. Measured on the received tree: both editor
  input paths (``getpass`` and the ``_ask`` fallback) do strip the typed value,
  so the reachable defect is the inherited list - an un-normalized key already
  in the config survived an add/delete, was saved back verbatim, and was
  stripped on the next start, silently changing a key that clients are already
  configured with. Plus a padded paste of an EXISTING key was stored as a
  second entry that can never authenticate. Both assignment points now write
  ``_normalize_keys`` output.
* **P3-4** ``accounts.add_account`` assigned the unvalidated
  ``settings.provider`` to the new account as its FALLBACK, persisting an id no
  pool, login or ban check can match - the same defect the HTTP route was fixed
  for (P1-6). The same unvalidated fallback existed in the two frontends that
  pre-assign the selection before calling it (``cli.cmd_accounts`` and
  ``cli._upsert_account``) and in ``tui._ask_account_provider``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import types

import pytest

import deepseaport.accounts as accounts_mod
import deepseaport.cli as cli_mod
from deepseaport.config import (
    MAX_RETRIES_LIMIT,
    AccountConfig,
    Settings,
    load_settings,
)
from support import write_config, on_disk

from deepseaport.core import registry


class _PipedStdin(io.StringIO):
    """A non-tty stdin (``< password.txt``).

    ``io.StringIO.isatty()`` is False, which is the whole point: the helper must
    take the piped path even though pytest's capture replaces the real stream.
    """

    def isatty(self) -> bool:
        return False


class _BoomStdin:
    """pytest's captured stdin behaves like this: a readable attribute, OSError
    on the actual read."""

    isatty = staticmethod(lambda: False)

    def readline(self):
        raise OSError("pytest: reading from stdin while output is captured")


# --------------------------------------------------------------------------- #
# P3-1: a piped secret must reach the login driver                              #
# --------------------------------------------------------------------------- #


def test_login_takes_the_piped_password_from_a_non_tty_stdin(monkeypatch, tmp_path):
    """P3-1. Pre-change: the driver was called with password='' and the
    scripted login silently authenticated as an empty password.

    Reproduces ``deepseaport login --email a@x.com < password.txt``.
    """
    seen: dict = {}
    monkeypatch.setattr(sys, "stdin", _PipedStdin("s3cret\n"))
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: seen.update(email=e, password=p)
                        or ("tok-from-login", False, ""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    rc = cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="", provider=None))

    assert rc == 0
    assert seen["password"] == "s3cret", (
        "the piped secret never reached the login driver")
    assert seen["email"] == "a@x.com"
    # ...and the account it created carries it.
    assert on_disk(cfg)["accounts"][0]["password"] == "s3cret"


def test_piped_secret_eof_is_the_no_password_case(monkeypatch, tmp_path):
    """P3-1, EOF half. No flag, no tty, no data on stdin -> "" and the
    caller's existing "no password" path decides; never a crash."""
    seen: dict = {}
    monkeypatch.setattr(sys, "stdin", _PipedStdin(""))  # immediate EOF
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: seen.update(password=p)
                        or ("", False, ""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    rc = cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="a@x.com", password="", provider=None))

    assert seen["password"] == ""
    assert rc == 3  # no token -> login failed, the documented no-secret path
    assert on_disk(cfg)["accounts"] == []  # nothing persisted


def test_accounts_add_reads_piped_token_and_password_in_prompt_order(monkeypatch,
                                                                    tmp_path):
    """P3-1, the other two call sites (``accounts add --token/--password``).

    One line per prompt, exactly what two ``fallback_getpass`` calls consumed:
    the token prompt reads line 1, the password prompt line 2.
    """
    monkeypatch.setattr(sys, "stdin", _PipedStdin("tok-from-pipe\npw-from-pipe\n"))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_mod.cmd_accounts(argparse.Namespace(
        accounts_action="add", identifier="", config=cfg, email="a@x.com", mobile="",
        token="", password="", provider=None)) == 0

    acc = on_disk(cfg)["accounts"][0]
    assert (acc["token"], acc["password"]) == ("tok-from-pipe", "pw-from-pipe")


def test_secret_flag_still_wins_and_never_consumes_stdin(monkeypatch, tmp_path,
                                                         capsys):
    """P3-1, precedence: an explicit flag wins, warns, and leaves the piped
    stream unread for the next prompt (getpass would not have read it either)."""
    monkeypatch.setattr(sys, "stdin", _PipedStdin("line-1\nline-2\n"))
    cfg = write_config(tmp_path, {"keys": [], "accounts": []})

    assert cli_mod.cmd_accounts(argparse.Namespace(
        accounts_action="add", identifier="", config=cfg, email="a@x.com", mobile="",
        token="tok-flag", password="", provider=None)) == 0

    assert "process list" in capsys.readouterr().out
    acc = on_disk(cfg)["accounts"][0]
    assert acc["token"] == "tok-flag"
    # The second prompt still gets its own line: nothing was consumed early.
    assert acc["password"] == "line-1"


@pytest.mark.parametrize("stdin_obj,expected", [
    (None, ""),  # pythonw.exe / frozen launcher: no sys.stdin at all
    (types.SimpleNamespace(), ""),  # a stream object without readline
    (_BoomStdin(), ""),  # a stream that refuses to be read
])
def test_read_secret_line_is_safe_without_a_readable_stdin(monkeypatch, stdin_obj,
                                                           expected):
    """P3-1, the "no tty, no data" guard. Every unreadable-stdin shape yields
    "" instead of raising, so a headless launch cannot crash on a secret."""
    monkeypatch.setattr(sys, "stdin", stdin_obj)
    assert cli_mod._read_secret_line() == expected


def test_tty_still_uses_the_hidden_prompt(monkeypatch):
    """P3-1, tty half unchanged: no flag + a console -> getpass, so the secret
    never lands in the process list or the shell history."""
    prompted: list[str] = []
    monkeypatch.setattr(cli_mod, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr("getpass.getpass",
                        lambda p="": prompted.append(p) or "typed-secret")

    assert cli_mod._secret_from_arg("", "--password", "DeepSeek password") \
        == "typed-secret"
    assert any("DeepSeek password" in p for p in prompted)


# --------------------------------------------------------------------------- #
# P3-2: the max_retries clamp must not rewrite the operator's config.json        #
# --------------------------------------------------------------------------- #


def test_max_retries_clamp_is_not_written_back_by_a_save(tmp_path, caplog):
    """P3-2. Pre-change: runtime 5 and the FILE became 5 after one save, i.e.
    any unrelated action (account add, TUI toggle) destroyed the operator's 9.

    Asserted on the file, not on the in-memory object, because the whole point
    is what survives to disk.
    """
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 9})
    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        settings = load_settings(cfg)

    # The clamp is still a real runtime bound...
    assert settings.max_retries == MAX_RETRIES_LIMIT
    assert any("clamped" in r.getMessage() for r in caplog.records)
    assert any("max_retries" in r.getMessage() for r in caplog.records)

    # ...but an unrelated save (here: nothing but save() itself) leaves the
    # operator's own number on disk.
    settings.active_account = "someone@example.invalid"
    settings.save()

    assert on_disk(cfg)["max_retries"] == 9
    # Still clamped on the next start: the file is the operator's, the runtime
    # bound is ours.
    assert load_settings(cfg).max_retries == MAX_RETRIES_LIMIT
    assert on_disk(cfg)["max_retries"] == 9


def test_max_retries_in_range_round_trips_unchanged(tmp_path):
    """P3-2, control: a value inside the documented range is persisted as-is
    (the bookkeeping must not shadow a legitimate value)."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 3})
    settings = load_settings(cfg)
    assert settings.max_retries == 3

    settings.save()

    assert on_disk(cfg)["max_retries"] == 3
    assert load_settings(cfg).max_retries == 3


def test_deliberate_max_retries_change_is_persisted_over_the_operator_value(tmp_path):
    """P3-2, the other side of the coin: once the value stops being the clamp
    (TUI "Retry count" -> 2), that is the intentional number and must win.
    Otherwise the file would be pinned to 9 forever and the TUI would lie."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 9})
    settings = load_settings(cfg)
    assert settings.max_retries == MAX_RETRIES_LIMIT

    settings.max_retries = 2  # what tui._edit_retries does
    settings.save()

    assert on_disk(cfg)["max_retries"] == 2
    assert load_settings(cfg).max_retries == 2


def test_save_does_not_warn_about_the_clamp_bookkeeping(tmp_path, caplog):
    """P3-2, mirror guard: the clamp bookkeeping is a plain instance attribute,
    not a dataclass field, so the guard in save() stays meaningful and silent.
    Pre-change there was no bookkeeping, and adding one as a field would have
    made save() log 'Settings field(s) not persisted' on every save."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 9})
    settings = load_settings(cfg)
    assert not set(settings.__dataclass_fields__) & {
        "_raw_persisted_values", "max_retries_raw"}

    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        settings.save()

    assert [r.getMessage() for r in caplog.records
            if "not persisted" in r.getMessage()] == []


def test_unpersisted_field_still_trips_the_mirror_guard(tmp_path, caplog):
    """P3-2, the guard still guards: a NEW dataclass field is reported, so
    widening the exempt set for the bookkeeping was not the way to go."""
    settings = Settings(config_path=write_config(tmp_path, {"keys": []}))
    # Bypass the frozen-ness of __dataclass_fields__ the way a real added field
    # would appear: the point is that the guard compares field names.
    settings.__dataclass_fields__ = dict(settings.__dataclass_fields__,
                                        brand_new_field=object())

    with caplog.at_level(logging.WARNING, logger="deepseaport.config"):
        settings.save()

    assert any("brand_new_field" in r.getMessage() for r in caplog.records)


def test_max_retries_env_override_is_not_recorded_as_a_file_value(monkeypatch,
                                                                  tmp_path):
    """P3-2, judgement call made explicit: only a CONFIG-FILE clamp is
    recorded. An env override is transient and is not owned by config.json, so
    there is no on-disk number to protect (pre-existing behaviour unchanged)."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 3})
    monkeypatch.setenv("DEEPSEAPORT_MAX_RETRIES", "99")

    settings = load_settings(cfg)
    assert settings.max_retries == MAX_RETRIES_LIMIT
    settings.save()

    assert on_disk(cfg)["max_retries"] == MAX_RETRIES_LIMIT


# --------------------------------------------------------------------------- #
# P3-3: the TUI key editor must store the key the server will match             #
# --------------------------------------------------------------------------- #


def _edit_keys(monkeypatch, settings, answers, secrets):
    """Drive tui._edit_keys with scripted answers / typed secrets.

    ``answers`` feeds the menu prompts, ``secrets`` the getpass calls.
    """
    from deepseaport.tui import _edit_keys as run

    answers = iter(answers)
    secrets = iter(secrets)
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    monkeypatch.setattr("getpass.getpass", lambda *a, **k: next(secrets))
    run(settings)


def _edit_keys_add(monkeypatch, settings, typed: str):
    """One 'add key', then back."""
    _edit_keys(monkeypatch, settings, ["a", "b"], [typed])


def test_tui_add_normalizes_the_whole_stored_list(monkeypatch, tmp_path):
    """P3-3, the regression. Pre-change ``settings.keys = keys`` wrote whatever
    the list held, so an un-normalized key already in the config survived the
    edit and was saved back verbatim - and load_settings stripped it on the next
    start, silently changing a key that clients are already configured with.

    The two editor input paths do strip the typed value, so the reachable drift
    is the inherited list, not the new entry: this asserts the assignment
    itself is the normalizer's output.
    """
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})
    settings = Settings(keys=["  sk-padded  "], config_path=cfg)

    _edit_keys_add(monkeypatch, settings, "sk-new")

    assert settings.keys == ["sk-padded", "sk-new"]
    settings.save()
    assert on_disk(cfg)["keys"] == ["sk-padded", "sk-new"]
    # The stored key is the loaded key: no silent change on the next start.
    assert load_settings(cfg).keys == settings.keys


def test_tui_delete_normalizes_the_whole_stored_list(monkeypatch, tmp_path):
    """P3-3, second assignment point: the delete branch had the same unguarded
    ``settings.keys = keys``, and a delete is an ordinary save. The surviving
    padded key is the one that must come back normalized."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})
    settings = Settings(keys=["  sk-padded  ", "sk-drop-me"], config_path=cfg)

    _edit_keys(monkeypatch, settings, ["d 2", "y", "b"], [])

    assert settings.keys == ["sk-padded"]
    settings.save()
    assert load_settings(cfg).keys == settings.keys == ["sk-padded"]


def test_tui_add_key_stores_the_stripped_value(monkeypatch, tmp_path):
    """P3-3, intent: a padded paste is stored trimmed, so the value the server
    matches is the value on disk (the input paths already strip; the
    normalizer at the assignment is what keeps that true for every key)."""
    settings = Settings(keys=[], config_path=write_config(
        tmp_path, {"keys": [], "max_retries": 1}))

    _edit_keys_add(monkeypatch, settings, "  sk-typed-with-padding  ")

    assert settings.keys == ["sk-typed-with-padding"]
    # And what was persisted matches what is in memory: no silent change on
    # the next start.
    settings.save()
    assert on_disk(settings.config_path)["keys"] == ["sk-typed-with-padding"]
    assert load_settings(settings.config_path).keys == settings.keys


def test_tui_add_key_round_trips_unchanged_through_a_reload(monkeypatch, tmp_path):
    """P3-3, reload half: two adds in a row round-trip unchanged."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})
    settings = load_settings(cfg)

    _edit_keys_add(monkeypatch, settings, "sk-one ")
    _edit_keys_add(monkeypatch, settings, "sk-two\t")

    settings.save()
    reloaded = load_settings(cfg)
    assert reloaded.keys == settings.keys == ["sk-one", "sk-two"]


def test_tui_blank_key_is_cancelled_not_stored(monkeypatch, tmp_path):
    """P3-3, ordering: the normalizer runs BEFORE the empty check, so a
    whitespace-only paste is cancelled. Stored, it normalizes to a blank entry
    on the next load, and a blank key is an auth bypass, not a weaker key."""
    settings = Settings(keys=["sk-existing-key"], config_path=write_config(
        tmp_path, {"keys": ["sk-existing-key"], "max_retries": 1}))

    _edit_keys_add(monkeypatch, settings, "   ")

    assert settings.keys == ["sk-existing-key"]


def test_tui_duplicate_check_sees_the_stripped_key(monkeypatch, tmp_path):
    """P3-3: an existing key re-typed with padding is a duplicate, not a
    second entry that can never authenticate."""
    settings = Settings(keys=["sk-existing-key"], config_path=write_config(
        tmp_path, {"keys": ["sk-existing-key"], "max_retries": 1}))

    _edit_keys_add(monkeypatch, settings, " sk-existing-key  ")

    assert settings.keys == ["sk-existing-key"]


# --------------------------------------------------------------------------- #
# P3-4: the add path must not persist an unusable provider id                    #
# --------------------------------------------------------------------------- #


def test_add_account_does_not_override_an_explicitly_chosen_provider(tmp_path):
    """P3-4, scope: only the FALLBACK is coerced. An id the operator chose is
    stored verbatim - the CLI/TUI accept a provider the registry does not know
    (registered later), and that call is theirs. Pinned here because
    tests/test_provider_menu.py::test_accounts_add_provider_flag_unregistered_ok
    is the same contract from the other direction."""
    settings = load_settings(write_config(
        tmp_path, {"keys": [], "accounts": [], "max_retries": 1}))
    cfg = AccountConfig(email="e@x.com", token="t", provider="future-provider")

    accounts_mod.add_account(settings, accounts_mod.AccountPool(settings.accounts),
                             cfg)

    assert settings.accounts[0].provider == "future-provider"


def test_usable_provider_survives_an_unimportable_registry(monkeypatch):
    """P3-4, defensive: a registry that cannot be imported must not break
    account storage, so the configured id is kept as-is (and an empty one still
    falls back)."""
    # A None entry in sys.modules makes ``from .core import registry`` raise
    # ImportError, which is the failure shape the guard is there for.
    monkeypatch.setitem(sys.modules, "deepseaport.core.registry", None)
    assert accounts_mod.usable_provider("qwen") == "qwen"
    assert accounts_mod.usable_provider("") == accounts_mod.DEFAULT_PROVIDER_ID


def test_cli_accounts_add_never_persists_an_unregistered_provider(monkeypatch,
                                                                  tmp_path):
    """P3-4 end-to-end, the reachable door: `accounts add` used to copy
    settings.provider onto the account before add_account saw it, so the
    coercion in accounts never ran."""
    monkeypatch.setenv("DEEPSEAPORT_PROVIDER", "junk-provider")
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})

    assert cli_mod.cmd_accounts(argparse.Namespace(
        accounts_action="add", identifier="", config=cfg, email="e@x.com", mobile="",
        token="tok", password="", provider=None)) == 0

    stored = on_disk(cfg)["accounts"][0]["provider"]
    assert stored in registry.registered(), (
        f"accounts add persisted the unusable provider {stored!r}")


def test_cli_accounts_add_keeps_the_configured_registered_provider(monkeypatch,
                                                                  tmp_path):
    """P3-4, control on the same door: DEEPSEAPORT_PROVIDER=qwen still binds
    the account to qwen (no flattening to the registry default)."""
    monkeypatch.setenv("DEEPSEAPORT_PROVIDER", "qwen")
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})

    assert cli_mod.cmd_accounts(argparse.Namespace(
        accounts_action="add", identifier="", config=cfg, email="e@x.com", mobile="",
        token="tok", password="", provider=None)) == 0

    assert on_disk(cfg)["accounts"][0]["provider"] == "qwen"


def test_cli_accounts_add_keeps_an_explicit_unregistered_provider(tmp_path):
    """P3-4, scope: `--provider <id the registry does not know>` is the
    operator's choice and is stored verbatim (the CLI may be pointed at a
    provider registered later). Only the inherited selection is coerced."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})

    assert cli_mod.cmd_accounts(argparse.Namespace(
        accounts_action="add", identifier="", config=cfg, email="e@x.com", mobile="",
        token="tok", password="", provider="future-provider")) == 0

    assert on_disk(cfg)["accounts"][0]["provider"] == "future-provider"


def test_login_account_never_inherits_an_unregistered_provider(monkeypatch, tmp_path):
    """P3-4, the login door: `deepseaport login` created its account through
    cli._upsert_account, which bound the same unvalidated settings.provider."""
    monkeypatch.setenv("DEEPSEAPORT_PROVIDER", "junk-provider")
    monkeypatch.setattr(cli_mod, "_obscura_login_token",
                        lambda e, p, s: ("tok-from-login", False, ""))
    cfg = write_config(tmp_path, {"keys": [], "accounts": [], "max_retries": 1})

    assert cli_mod.cmd_login(argparse.Namespace(
        config=cfg, email="e@x.com", password="pw", provider=None)) == 0

    assert on_disk(cfg)["accounts"][0]["provider"] in registry.registered()


def test_tui_ask_account_provider_coerces_an_unregistered_selection(monkeypatch):
    """P3-4, the TUI door: single mode returns the inherited selection, which
    load_settings does not registry-check."""
    from deepseaport.tui import _ask_account_provider

    settings = Settings(provider="junk-provider")

    assert _ask_account_provider(settings) in registry.registered()


def test_tui_ask_account_provider_keeps_a_registered_selection():
    """P3-4, control on the TUI door."""
    from deepseaport.tui import _ask_account_provider

    assert _ask_account_provider(Settings(provider="qwen")) == "qwen"


def test_tui_ask_account_provider_empty_answer_stays_in_the_active_set(monkeypatch):
    """The multi-mode default was the literal "deepseek", so with
    ``providers=["qwen","opencode"]`` a bare Enter filed the account under a
    provider that is not active -- the ``login --provider glm`` class: an
    account no pool builds from and no request can serve."""
    from deepseaport.core import pools
    from deepseaport.tui import _ask_account_provider

    settings = Settings(providers=["qwen", "opencode"])
    shown: dict = {}

    def _ask(prompt, default=""):
        shown["default"] = default
        return ""

    monkeypatch.setattr("deepseaport.tui._ask", _ask)

    picked = _ask_account_provider(settings)

    assert picked in pools.active_provider_ids(settings)
    assert picked != "deepseek"
    assert shown["default"] == picked, "the on-screen default differs from the result"


def test_tui_ask_account_provider_eof_stays_in_the_active_set(monkeypatch):
    """Same guarantee on the EOF/KeyboardInterrupt path, which returns its own
    literal."""
    from deepseaport.core import pools
    from deepseaport.tui import _ask_account_provider

    settings = Settings(providers=["qwen", "opencode"])

    def _eof(*a, **k):
        raise EOFError

    monkeypatch.setattr("deepseaport.tui._ask", _eof)

    assert _ask_account_provider(settings) in pools.active_provider_ids(settings)


def test_tui_ask_account_provider_keeps_deepseek_as_the_default(monkeypatch):
    """Control: while deepseek IS active the historical default is unchanged."""
    from deepseaport.tui import _ask_account_provider

    monkeypatch.setattr("deepseaport.tui._ask", lambda *a, **k: "")

    assert _ask_account_provider(Settings(providers=["deepseek", "qwen"])) == "deepseek"
