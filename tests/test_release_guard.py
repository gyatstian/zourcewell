"""The release guard must fail on a credential store and pass on a clean tree.

Every fixture here is built inside ``tmp_path``. The point of the guard is that
``dist/config.json`` holds real passwords and 64-char bearer tokens, so pointing
a test at the real ``dist/`` would make the suite read (and, on a bug, copy) the
operator's live credentials -- and a passing run would prove nothing if the
"clean" case were the real tree anyway.

The fixtures are synthetic and deliberately shaped like the real file:
``config.json`` with populated accounts, a ``*.bak`` left by a degraded save, a
``mkstemp`` leftover from an interrupted one, and ``config.example.json``
carrying the PUBLIC example key that the README already prints.
"""

from __future__ import annotations

import importlib.util
import json
import tomllib
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "prepare_release.py"


def _load():
    spec = importlib.util.spec_from_file_location("prepare_release", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare_release = _load()

# One synthetic account, same shape as config.py reads: plaintext password and
# a 64-char token. Never a real value.
_LIVE = {
    "keys": ["sk-not-a-real-key"],
    "accounts": [{
        "email": "fixture@example.invalid",
        "mobile": "",
        "password": "fixture-password-not-real",
        "token": "a" * 64,
        "token_expires_at": 0.0,
        "provider": "deepseek",
        "banned": False,
        "banned_until": 0.0,
    }],
    "active_account": "",
    "obscura_bin": "",
    "obscura_profile": "",
    "browser_bin": "auto",
    "browser_headless": False,
    "port": 5001,
    "listen": False,
    "provider": "deepseek",
    "providers": [],
}


def _tree(tmp_path: Path, **files) -> Path:
    """A fake PyInstaller output dir. Returns its path."""
    src = tmp_path / "dist"
    src.mkdir(exist_ok=True)
    for rel, content in files.items():
        target = src / rel.replace("__", "/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        elif isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_text(json.dumps(content, indent=2), encoding="utf-8")
    return src


def _clean(src: Path) -> Path:
    """Minimum realistic artifact: the exe, the downloaded wasm, the template."""
    (src / "deepseaport.exe").write_bytes(b"MZ" + b"\0" * 512)
    (src / "data").mkdir(exist_ok=True)
    (src / "data" / "sha3_wasm_bg.7b9ca65ddd.wasm").write_bytes(b"\0asm" + b"\0" * 64)
    (src / "config.example.json").write_text(json.dumps({
        "keys": ["sk-local-dev"],
        "accounts": [{"email": "", "mobile": "", "password": "", "token": "",
                      "token_expires_at": 0.0, "provider": "deepseek",
                      "banned": False, "banned_until": 0.0}],
        "active_account": "", "obscura_bin": "", "obscura_profile": "",
        "browser_bin": "auto", "browser_headless": False, "port": 5001,
        "listen": False, "enable_tools": True, "warmup_on_startup": True,
        "auto_delete_session": True, "max_retries": 1,
        "parallel_challenge_fetch": True, "tool_args_max_chars": 200000,
        "forgiving_toolcalls": False, "use_multiple_accounts": True,
        "log_level": "INFO", "stream_mode": "live",
        "chat_model": "deepseek-flash", "provider": "deepseek", "providers": [],
        "enable_append": True, "append_top": "", "append_bottom": "",
        "send_dry_run_to_frontend": True}, indent=2), encoding="utf-8")
    return src


# --- the guard detects credentials -----------------------------------------

def test_populated_config_json_blocks_the_release(tmp_path):
    """The real failure mode: dist/config.json IS the frozen app's credential
    store, so packaging it publishes every account the operator owns."""
    src = _clean(_tree(tmp_path, **{"config.json": _LIVE}))
    out = tmp_path / "release"
    rc = prepare_release.main(["--dist", str(src), "--out", str(out)])
    assert rc == 1
    assert not out.exists(), "nothing may be written when a secret is present"


def test_check_mode_blocks_without_writing(tmp_path):
    src = _clean(_tree(tmp_path, **{"config.json": _LIVE}))
    out = tmp_path / "release"
    rc = prepare_release.main(["--check", "--dist", str(src), "--out", str(out)])
    assert rc == 1
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [src], "audit mode wrote something"


def test_blank_config_json_is_still_blocked(tmp_path):
    """Positional rule, not a content rule: an "empty" config.json is empty
    only until the first run of the frozen exe writes real credentials into it.
    """
    src = _clean(_tree(tmp_path, **{"config.json": {"keys": [], "accounts": []}}))
    assert prepare_release.main(["--dist", str(src),
                                 "--out", str(tmp_path / "release")]) == 1


def test_other_json_with_credentials_blocks_the_release(tmp_path):
    """A renamed credential store must not slip through on a filename."""
    src = _clean(_tree(tmp_path, **{"state.json": _LIVE}))
    out = tmp_path / "release"
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 1
    assert not out.exists()


def test_nested_config_json_blocks_the_release(tmp_path):
    src = _clean(_tree(tmp_path, **{"sub__config.json": _LIVE}))
    assert prepare_release.main(["--dist", str(src),
                                 "--out", str(tmp_path / "release")]) == 1


def test_block_names_the_offending_paths(tmp_path):
    """A refusal that does not say which file is not actionable."""
    src = _clean(_tree(tmp_path, **{"config.json": _LIVE}))
    assert prepare_release.main(["--dist", str(src),
                                 "--out", str(tmp_path / "release")]) == 1
    listed = [rel.as_posix() for rel, _r, _d in prepare_release.survey(src)[2]]
    assert listed == ["config.json"]


def test_blank_secret_values_do_not_block(tmp_path):
    """``"token": ""`` is how a fresh account entry spells "not set"; blocking
    on it would fail a legitimate clean-tree release over an empty field.
    Every SECRET_KEYS field is blank here on purpose -- a populated ``email``
    is a credential in its own right (see the sibling test below), so a
    "blank" fixture with a filled one no longer tests what it claims to.
    """
    src = _clean(_tree(tmp_path, **{"state.json": {
        "keys": ["", "   "],
        "accounts": [{"email": "", "mobile": "", "password": "",
                      "token": "", "banned_until": 0.0}],
        "port": 5001}}))
    assert prepare_release.main(["--dist", str(src),
                                 "--out", str(tmp_path / "release")]) == 0


# --- a clean tree packages, and the template is allowed through -----------

def test_clean_tree_packages(tmp_path):
    src = _clean(_tree(tmp_path))
    out = tmp_path / "release"
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 0
    zip_path = out / "dist.zip"
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert f"dist/deepseaport.exe" in names
    assert f"dist/data/sha3_wasm_bg.7b9ca65ddd.wasm" in names
    # The public template passes: it holds the example key the README prints.
    assert f"dist/config.example.json" in names


def test_check_mode_on_a_clean_tree_passes_and_writes_nothing(tmp_path):
    src = _clean(_tree(tmp_path))
    out = tmp_path / "release"
    assert prepare_release.main(["--check", "--dist", str(src),
                                 "--out", str(out)]) == 0
    assert not out.exists()


# --- exclusions -----------------------------------------------------------

def test_backups_and_temp_files_are_excluded(tmp_path):
    """Degraded saves leave both, and both are copies of the credential file."""
    src = _clean(_tree(tmp_path, **{
        "config.json.bak": _LIVE,
        "config.json.abc123.tmp": _LIVE,
        "build.log": "noise",
    }))
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"\0")
    out = tmp_path / "release"
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 0
    with zipfile.ZipFile(out / "dist.zip") as zf:
        names = zf.namelist()
    assert not [n for n in names if n.endswith((".bak", ".tmp", ".log", ".pyc"))]


def test_env_and_pem_leftovers_are_excluded(tmp_path):
    src = _clean(_tree(tmp_path, **{".env": "DEEPSEAPORT_TOKEN=x", "k.pem": "x"}))
    out = tmp_path / "release"
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 0
    with zipfile.ZipFile(out / "dist.zip") as zf:
        names = zf.namelist()
    assert f"dist/.env" not in names
    assert f"dist/k.pem" not in names


def test_survey_classifies_without_writing(tmp_path):
    src = _clean(_tree(tmp_path, **{"config.json": _LIVE}))
    included, excluded, secrets = prepare_release.survey(src)
    assert [rel.as_posix() for rel, _r, _d in secrets] == ["config.json"]
    assert any(rel.as_posix() == "deepseaport.exe" for rel, _s in included)
    assert any(rel.as_posix() == "config.example.json" for rel, _s in included)


# --- the guard never touches the source tree ------------------------------

def test_source_tree_is_never_modified(tmp_path):
    """dist/config.json is the operator's LIVE config for the frozen exe. The
    guard may block the release; it may not move or delete their credentials."""
    src = _clean(_tree(tmp_path, **{"config.json": _LIVE}))
    before = {p.relative_to(src).as_posix(): p.read_bytes()
              for p in sorted(src.rglob("*")) if p.is_file()}
    assert prepare_release.main(["--dist", str(src),
                                 "--out", str(tmp_path / "release")]) == 1
    after = {p.relative_to(src).as_posix(): p.read_bytes()
             for p in sorted(src.rglob("*")) if p.is_file()}
    assert after == before


def test_output_inside_the_source_tree_is_refused(tmp_path):
    src = _clean(_tree(tmp_path))
    with pytest.raises(SystemExit):
        prepare_release.build(str(src), str(src / "release"), False)


def test_rerunning_is_safe_and_produces_the_same_artifact(tmp_path):
    """Two runs in a row: the second replaces the staging tree instead of
    merging into it, so the zip is byte-for-byte the same content set."""
    src = _clean(_tree(tmp_path))
    out = tmp_path / "release"
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 0
    first = sorted(zipfile.ZipFile(out / "dist.zip").namelist())
    stale = out / "dist" / "leftover-from-a-previous-build.txt"
    stale.write_text("stale", encoding="utf-8")
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 0
    second = sorted(zipfile.ZipFile(out / "dist.zip").namelist())
    assert first == second
    assert not stale.exists(), "stale file survived a rebuild"
    assert src.is_dir() and (src / "deepseaport.exe").is_file()


# --- account identifiers are credentials too --------------------------------

def test_json_with_only_an_email_and_mobile_blocks_the_release(tmp_path):
    """``email`` / ``mobile`` were absent from SECRET_KEYS while the guard's own
    docstring and the README both named account emails first, so a store
    carrying nothing but identifiers shipped clean. They are the half of an
    account that survives a password rotation.
    """
    src = _clean(_tree(tmp_path, **{"state.json": {
        "accounts": [{"email": "fixture@example.invalid",
                      "mobile": "+48123456789"}],
        "port": 5001}}))
    out = tmp_path / "release"
    assert prepare_release.main(["--dist", str(src), "--out", str(out)]) == 1
    assert not out.exists()


def test_check_flags_email_and_mobile_and_writes_nothing(tmp_path):
    """The audit path must reach the same verdict, because the release workflow
    gates the upload on `--check` alone."""
    src = _clean(_tree(tmp_path, **{"state.json": {
        "accounts": [{"email": "fixture@example.invalid",
                      "mobile": "+48123456789"}],
        "port": 5001}}))
    out = tmp_path / "release"
    assert prepare_release.main(["--check", "--dist", str(src),
                                 "--out", str(out)]) == 1
    assert not out.exists(), "--check must write nothing"
    assert list(tmp_path.iterdir()) == [src], "audit mode wrote something"
    _inc, _exc, secrets = prepare_release.survey(src)
    assert [rel.as_posix() for rel, _r, _d in secrets] == ["state.json"]


# --- one version, three consumers ------------------------------------------

def _pyproject_version() -> str:
    return tomllib.loads(
        (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]


def test_spec_derives_the_frozen_version_from_pyproject():
    """deepseaport.spec stamps [project] version into the exe's version
    resource. It used to wrap that read in `except Exception: VERSION =
    "0.0.0"`, so a build on an interpreter without tomllib -- or against a
    moved pyproject.toml -- shipped an exe whose version read 0.0.0 with no
    build failure and no filename difference to notice it by.

    The block is exec'd out of the spec source rather than reimplemented here,
    so putting the fallback back turns this into a SyntaxError instead of a
    silent pass.
    """
    src = (_ROOT / "deepseaport.spec").read_text(encoding="utf-8")
    # "\nVERSION = " (line-anchored) so the prose in the spec's own comment
    # about the fallback it used to have cannot be mistaken for the code.
    block = src[src.index("\nVERSION = ") + 1:src.index("datas = []")]
    ns: dict = {"tomllib": tomllib, "Path": Path, "SPECPATH": str(_ROOT)}
    exec(block, ns)
    assert ns["VERSION"] == _pyproject_version()


def test_package_version_is_the_pyproject_version():
    """deepseaport.__version__ is what FastAPI's `version=` and any /health
    field read. It used to be a hardcoded literal that had drifted to 0.1.0
    while nothing imported it at all.
    """
    import deepseaport

    assert deepseaport.__version__ == _pyproject_version()