"""F29-lite: ``deepseaport --version``.

The main parser had no version action, so the one obvious way to ask a one-shot
CLI what it is exited 2 ("unrecognized arguments") and printed usage. The string
must come from ``deepseaport.__version__`` -- the single version name derived
from ``pyproject.toml`` -- never a second literal.
"""

from __future__ import annotations

import pytest

import deepseaport
from deepseaport.cli import main as cli_main


def test_version_flag_prints_the_package_version_and_exits_0(capsys):
    with pytest.raises(SystemExit) as exc:
        cli_main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"deepseaport {deepseaport.__version__}"


def test_version_flag_works_with_a_config_arg_first(capsys, tmp_path):
    """It is a parser-level option, so argument order must not matter."""
    with pytest.raises(SystemExit) as exc:
        cli_main(["--config", str(tmp_path / "config.json"), "--version"])

    assert exc.value.code == 0
    assert deepseaport.__version__ in capsys.readouterr().out


def test_unknown_flag_still_exits_2(capsys):
    """Control: the new action must not turn argparse's refusal into a 0."""
    with pytest.raises(SystemExit) as exc:
        cli_main(["--nope"])

    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
