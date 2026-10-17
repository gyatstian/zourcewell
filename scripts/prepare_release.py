#!/usr/bin/env python
"""Assemble a release directory + zip from ``dist/`` without shipping credentials.

``dist/`` doubles as the frozen app's live credential store; ``ARCHITECTURE.md``
("The release guard is ``scripts/prepare_release.py``") owns the rationale.

So this script is the packaging step, not a lint on it:

* it never writes to, deletes from, or moves anything inside the source tree --
  the operator's live ``config.json`` stays exactly where the exe expects it;
* it drops build noise and secret-shaped leftovers that are not the credential
  store (the globs and suffixes below are the list);
* it REFUSES to produce an artifact when the tree holds a ``config.json`` or any
  other JSON carrying a non-empty ``email`` / ``mobile`` / ``token`` /
  ``password`` / ``keys`` / ``api_key`` / ``secret`` value, and names the file
  and the path inside it. A JSON file it cannot read or decode is blocked the
  same way: "clean" is only ever the verdict for a file it actually read and
  walked.

Content scanning is JSON-only: a secret in ``settings.ini``, ``credentials.txt``,
a ``.har`` capture or a screenshot of the TUI is dropped only if its NAME matches
one of the exclusion globs below, and otherwise ships.

Dependency-free (stdlib only) so it runs on the same interpreter that built the
exe with nothing else installed.

    python scripts/prepare_release.py --check          # audit, write nothing
    python scripts/prepare_release.py                  # release/ + zip
    python scripts/prepare_release.py --dist build\\out --out stage

Exit codes: 0 packaged (or clean under ``--check``), 1 credentials found,
2 usage/IO error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from fnmatch import fnmatch
from pathlib import Path

# JSON keys whose non-empty value is a credential. ``email`` / ``mobile`` /
# ``token`` / ``password`` / ``keys`` are the shapes config.json itself uses --
# ``email`` and ``mobile`` were missing here once, so a store carrying nothing
# but account identifiers passed clean while the docstring above named emails
# first. ``api_key`` / ``secret`` are here so a renamed credential store is still
# caught by the content scan instead of sailing through on a filename this
# script has never heard of.
SECRET_KEYS = frozenset({"email", "mobile", "token", "password", "keys",
                         "api_key", "secret"})

# The live credential store, by exact name, at any depth. Rejected outright and
# not merely excluded: an "empty" dist/config.json is empty only until the first
# run of the frozen exe, which writes real credentials into it. The rule is
# positional rather than content-based because the position IS the risk.
LIVE_CONFIG_NAME = "config.json"

# Tracked templates, exempt by name so they pass through and are not scanned:
# config.example.json carries the PUBLIC example key from the README, so
# scanning it would fail every release. A real credential in a tracked file is a
# git-level mistake, not a packaging one.
EXAMPLE_SUFFIX = ".example.json"

# Build noise and secret-shaped leftovers. Dropped and reported, not fatal: none
# of these is a file the exe reads, so keeping them only wastes bytes or
# republishes a copy of a secret some other tool left lying around.
EXCLUDE_SUFFIXES = (".bak", ".tmp", ".orig", ".log", ".pyc", ".pyd", ".pyo")
EXCLUDE_DIR_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache",
                               ".ruff_cache", "logs"})
EXCLUDE_NAMES = frozenset({".env", "credentials", "credentials.json", "id_rsa"})
# Matched against the file name AND the whole relative path, so a leftover
# profile directory (WAF cookies + logged-in localStorage, gitignored for the
# same reason config.json is) is dropped whole rather than one file at a time.
EXCLUDE_GLOBS = ("*.pem", "*.key", "secrets*.json", "*obscura-profile*",
                 "*profile*/Cookies")


def exclusion_reason(rel: Path) -> str | None:
    """Why ``rel`` is dropped from the artifact, or ``None`` to include it."""
    for part in rel.parts[:-1]:
        if part in EXCLUDE_DIR_NAMES:
            return f"inside build cache directory {part!r}"
    low = rel.name.lower()
    if low in EXCLUDE_NAMES:
        return "secret-shaped filename"
    for suffix in EXCLUDE_SUFFIXES:
        if low.endswith(suffix):
            return f"{suffix} file"
    flat = rel.as_posix().lower()
    for pattern in EXCLUDE_GLOBS:
        if fnmatch(low, pattern) or fnmatch(flat, pattern):
            return f"matches {pattern}"
    return None


def _has_value(value) -> bool:
    """Does ``value`` carry a credential, as opposed to spelling "not set"?

    Blank counts as empty, and so does a container whose every leaf is blank,
    because that is how the credential store itself is written when unused:
    ``"keys": ["", "   "]`` and ``"token": ""`` are what ``config.json`` looks
    like before anything is set (``_normalize_keys`` reads that list as *no*
    usable key), and a guard that blocked on them would fail every legitimate
    clean-tree release over an empty field.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_value(item) for item in value.values())
    return True


def _walk_json(node, trail: str = "") -> list[str]:
    """Dotted paths of every non-empty SECRET_KEYS value inside ``node``."""
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{trail}.{key}" if trail else str(key)
            if str(key).lower() in SECRET_KEYS and _has_value(value):
                hits.append(path)
            hits.extend(_walk_json(value, path))
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            hits.extend(_walk_json(value, f"{trail}[{idx}]"))
    return hits


def scan_json(path: Path) -> list[str]:
    """Credential paths inside one JSON file; never clean by default.

    Every outcome that is not "parsed and walked" is a hit: unparseable JSON,
    an unreadable file, and an undecodable byte stream all report rather than
    wave a file through. Returning ``[]`` for undecodable input was a
    fail-open hole: PowerShell 5.1's ``>`` writes UTF-16LE, so an operator's
    redirected copy of the credential store decoded as nothing at all and
    shipped. A file in a release artifact that claims to be JSON and cannot be
    read is exactly the condition under which a human should look.

    UTF-16 is the second attempt because it is what that redirect produces
    (BOM included); anything that fails both decoders is reported, not retried
    further.
    """
    text: str | None = None
    last_exc: Exception | None = None
    for encoding in ("utf-8", "utf-16"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except OSError as exc:
            # PermissionError, a vanished file, a locked handle: no decoder
            # can help, so do not retry.
            last_exc = exc
            break
        except UnicodeError as exc:  # includes UnicodeDecodeError
            last_exc = exc
    if text is None:
        return [f"<unreadable: {path.name} ({type(last_exc).__name__})>"]
    try:
        data = json.loads(text)
    except ValueError as exc:
        return [f"<unparseable JSON: {exc}>"]
    return _walk_json(data)


def survey(src: Path) -> tuple[list[tuple[Path, int]],
                              list[tuple[Path, str]],
                              list[tuple[Path, str, str]]]:
    """Classify every file under ``src`` (read-only).

    Returns (included, excluded, secrets): included is (relpath, bytes),
    excluded is (relpath, reason), secrets is (relpath, reason, detail).
    """
    included: list[tuple[Path, int]] = []
    excluded: list[tuple[Path, str]] = []
    secrets: list[tuple[Path, str, str]] = []
    for path in sorted(src.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(src)
        size = path.stat().st_size
        if rel.name.endswith(EXAMPLE_SUFFIX):
            included.append((rel, size))  # public template: through unscanned
            continue
        if rel.name == LIVE_CONFIG_NAME:
            detail = ", ".join(scan_json(path)[:6]) or "no values read (empty)"
            secrets.append((rel, "live credential store", detail))
            continue
        reason = exclusion_reason(rel)
        if reason is not None:
            excluded.append((rel, reason))
            continue
        if rel.suffix.lower() == ".json":
            hits = scan_json(path)
            if hits:
                secrets.append((rel, "credential values in JSON",
                                ", ".join(hits[:6])))
                continue
        included.append((rel, size))
    return included, excluded, secrets


def _report(included, excluded, secrets, stream=sys.stdout) -> None:
    for rel, reason in excluded:
        print(f"  exclude  {rel}  ({reason})", file=stream)
    for rel, reason, detail in secrets:
        print(f"  BLOCK    {rel}  ({reason})", file=stream)
        print(f"           {detail}", file=stream)
    print(f"  package  {len(included)} file(s), "
          f"{sum(size for _, size in included):,} bytes", file=stream)


def _guard_paths(src: Path, out: Path) -> None:
    """Refuse a layout where the output could clobber the source tree."""
    src_r, out_r = src.resolve(), out.resolve()
    if src_r == out_r or out_r == Path(out_r.anchor) or src_r == Path(src_r.anchor):
        raise SystemExit(f"refusing to operate on {src_r} -> {out_r}")
    if out_r.is_relative_to(src_r):
        raise SystemExit(f"refusing to write inside the source tree: "
                         f"{out_r} is under {src_r}")


def _copy_into(stage_dir: Path, src: Path, included) -> None:
    """Fresh staged tree: the previous one is removed, never merged into.

    Removing is safe because ``stage_dir`` is always under ``--out`` (checked by
    ``_guard_paths``) and always a directory this script created. Replacing
    rather than merging is what makes a second run produce the same artifact as
    the first instead of inheriting leftovers from a previous one.
    """
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    for rel, _size in included:
        target = stage_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, target)


def write_artifact(stage_dir: Path, zip_path: Path, included) -> None:
    """Zip the staged tree through a .part file, then atomically rename."""
    tmp_zip = zip_path.with_name(zip_path.name + ".part")
    if tmp_zip.exists():
        tmp_zip.unlink()
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, _size in included:
            zf.write(stage_dir / rel,
                     arcname=f"{zip_path.stem}/{rel.as_posix()}")
    os.replace(tmp_zip, zip_path)


def build(dist: str, out: str, check: bool) -> int:
    src = Path(dist).resolve()
    out_dir = Path(out).resolve()
    if not src.is_dir():
        print(f"error: not a directory: {src}", file=sys.stderr)
        return 2
    _guard_paths(src, out_dir)
    included, excluded, secrets = survey(src)

    if check:
        print(f"check {src}  (nothing will be written)")
        _report(included, excluded, secrets)
        if secrets:
            print(f"\nFAIL: {len(secrets)} credential-bearing file(s) in the "
                  f"artifact.", file=sys.stderr)
            print("Move or delete them, then re-run. The exe reads its config "
                  "at startup, so moving the live copy out of the source tree "
                  "is safe; this script deliberately will not do it for you.",
                  file=sys.stderr)
            return 1
        print("\nOK: nothing to strip, nothing to block. "
              "Re-run without --check to package.")
        return 0

    if secrets:
        print(f"refusing to package {src}: {len(secrets)} credential-bearing "
              f"file(s) present", file=sys.stderr)
        for rel, reason, detail in secrets:
            print(f"  BLOCK    {rel}  ({reason})\n           {detail}",
                  file=sys.stderr)
        print("\nNothing was written, and the source tree is untouched "
              "including the live config.json the exe reads.", file=sys.stderr)
        return 1

    stage_dir = out_dir / src.name
    zip_path = out_dir / f"{src.name}.zip"
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_into(stage_dir, src, included)
    write_artifact(stage_dir, zip_path, included)
    print(f"packaged {src}")
    _report(included, excluded, secrets)
    print(f"\n  dir  {stage_dir}")
    print(f"  zip  {zip_path}\n")
    print("Ship the zip: it carries no config.json, no *.bak and no "
          "credential-bearing JSON. The source tree, including the live "
          "config.json the exe reads, is unchanged.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prepare_release.py",
        description="Assemble a release dir + zip from the PyInstaller output "
                    "without shipping credentials. Never modifies the source "
                    "tree.")
    ap.add_argument("--dist", default="dist", metavar="DIR",
                    help="PyInstaller output directory (default: dist)")
    ap.add_argument("--out", default="release", metavar="DIR",
                    help="destination for the staged dir and zip "
                         "(default: release)")
    ap.add_argument("--check", action="store_true",
                    help="audit only: report what would be packaged, write "
                         "nothing, exit 1 if a credential is present")
    args = ap.parse_args(argv)
    try:
        return build(args.dist, args.out, args.check)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())