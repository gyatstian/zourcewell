"""F11: ``accounts set-token`` / ``accounts remove`` were provider-UNSCOPED.

One identifier can exist once per provider (``config.account_identity``), and
neither subcommand had a ``--provider`` flag: with one email on two providers the
command silently rewrote (or deleted) the FIRST record, and there was no way to
name the other one. The account ROUTES already scope this way
(``server_routes``: ``pid or pool.provider_id``), so the CLI was the odd one.

The flag-less call keeps its old target: the fallback scope is the management
pool's provider, which is the first bound record's own provider -- the record the
unscoped first match would have hit in every single-provider config.
"""

from __future__ import annotations

import argparse

from support import on_disk, write_config

from deepseaport.cli import main as cli_main


def _two_provider_config(tmp_path) -> str:
    """One email on both providers: two legitimate, distinct records."""
    return write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "same@x.com", "token": "ds-old", "provider": "deepseek"},
        {"email": "same@x.com", "token": "qw-old", "provider": "qwen"},
    ]})


def _tokens(path) -> dict:
    return {a["provider"]: a["token"] for a in on_disk(path)["accounts"]}


def test_set_token_provider_flag_touches_only_that_provider(tmp_path):
    cfg = _two_provider_config(tmp_path)

    assert cli_main(["--config", cfg, "accounts", "set-token", "same@x.com",
                     "--token", "qw-new", "--provider", "qwen"]) == 0

    assert _tokens(cfg) == {"deepseek": "ds-old", "qwen": "qw-new"}


def test_remove_provider_flag_removes_only_that_provider(tmp_path):
    cfg = _two_provider_config(tmp_path)

    assert cli_main(["--config", cfg, "accounts", "remove", "same@x.com",
                     "--provider", "qwen"]) == 0

    assert _tokens(cfg) == {"deepseek": "ds-old"}


def test_remove_provider_flag_refuses_when_only_another_provider_matches(
        tmp_path, capsys):
    """The dangerous direction: the scoped id must never fall back to the
    unscoped first match and delete a record the operator did not name."""
    cfg = write_config(tmp_path, {"keys": [], "accounts": [
        {"email": "ds@x.com", "token": "ds-old", "provider": "deepseek"},
    ]})

    assert cli_main(["--config", cfg, "accounts", "remove", "ds@x.com",
                     "--provider", "qwen"]) == 1

    assert "not found" in capsys.readouterr().out
    assert _tokens(cfg) == {"deepseek": "ds-old"}


def test_set_token_flagless_keeps_the_first_match_default(tmp_path):
    """Prior behaviour, pinned: no flag still writes the first matching record."""
    cfg = _two_provider_config(tmp_path)

    assert cli_main(["--config", cfg, "accounts", "set-token", "same@x.com",
                     "--token", "ds-new"]) == 0

    assert _tokens(cfg) == {"deepseek": "ds-new", "qwen": "qw-old"}


def test_remove_flagless_keeps_the_first_match_default(tmp_path):
    cfg = _two_provider_config(tmp_path)

    assert cli_main(["--config", cfg, "accounts", "remove", "same@x.com"]) == 0

    assert _tokens(cfg) == {"qwen": "qw-old"}


def test_set_token_scope_does_not_fall_back_on_a_bad_provider_flag(
        tmp_path, capsys):
    """A typo'd scope is a refusal, not the pool's provider: silently widening
    the scope back is exactly the bug this flag exists to close."""
    cfg = _two_provider_config(tmp_path)

    assert cli_main(["--config", cfg, "accounts", "set-token", "same@x.com",
                     "--token", "new", "--provider", "q w"]) == 2

    assert "invalid provider id" in capsys.readouterr().out
    assert _tokens(cfg) == {"deepseek": "ds-old", "qwen": "qw-old"}


def test_remove_scope_does_not_fall_back_on_a_bad_provider_flag(
        tmp_path, capsys):
    cfg = _two_provider_config(tmp_path)

    assert cli_main(["--config", cfg, "accounts", "remove", "same@x.com",
                     "--provider", "q w"]) == 2

    assert "invalid provider id" in capsys.readouterr().out
    assert _tokens(cfg) == {"deepseek": "ds-old", "qwen": "qw-old"}


def test_remove_scope_survives_a_namespace_without_the_provider_attr(
        tmp_path, capsys):
    """Direct callers (tests, embedders) build Namespaces by hand; a missing
    ``provider`` attribute must read as "no flag", not raise."""
    from deepseaport.cli import cmd_accounts

    cfg = _two_provider_config(tmp_path)
    args = argparse.Namespace(config=cfg, accounts_action="remove",
                              identifier="same@x.com")

    assert cmd_accounts(args) == 0
    assert _tokens(cfg) == {"qwen": "qw-old"}
    capsys.readouterr()
