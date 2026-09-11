"""Guard against drift between the Makefile and its Windows PowerShell shim.

GNU make does not ship with Windows, so the repository carries both a ``Makefile``
(used by CI on ubuntu) and a ``make.ps1`` shim (used for local development on Windows).
Two files describing one task interface will diverge unless something checks them, and
a target that exists in only one of them is a build that passes for one contributor and
fails for another. This module is that check.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "Makefile"
SHIM = REPO_ROOT / "make.ps1"

#: A Makefile target: a name in the first column, followed by a colon.
_MAKE_TARGET = re.compile(r"^(?P<name>[a-z][a-z0-9-]*):", re.MULTILINE)

#: A case label in the ``switch ($Target)`` block of make.ps1, e.g. ``    'lint' {``.
_SHIM_TARGET = re.compile(r"^ {4}'(?P<name>[a-z][a-z0-9-]*)'\s*\{", re.MULTILINE)


def _makefile_targets() -> set[str]:
    """Return the target names declared in the Makefile."""
    return set(_MAKE_TARGET.findall(MAKEFILE.read_text(encoding="utf-8")))


def _shim_targets() -> set[str]:
    """Return the target names handled by the make.ps1 switch statement."""
    return set(_SHIM_TARGET.findall(SHIM.read_text(encoding="utf-8")))


def test_both_task_runners_exist() -> None:
    """Both task runners are present at the repository root."""
    assert MAKEFILE.is_file()
    assert SHIM.is_file()


def test_makefile_and_shim_expose_the_same_targets() -> None:
    """Every Makefile target has a make.ps1 counterpart, and vice versa."""
    make_only = _makefile_targets() - _shim_targets()
    shim_only = _shim_targets() - _makefile_targets()
    assert not make_only, f"targets missing from make.ps1: {sorted(make_only)}"
    assert not shim_only, f"targets missing from Makefile: {sorted(shim_only)}"


def test_expected_targets_are_present() -> None:
    """The task interface the README documents is actually implemented."""
    expected = {
        "help",
        "sync",
        "lint",
        "format",
        "typecheck",
        "test",
        "check",
        "clean",
        "bench",
        "bench-fast",
        "figures",
    }
    assert expected <= _makefile_targets()


def test_makefile_recipes_are_tab_indented() -> None:
    """Recipe lines use tabs, not spaces.

    Git on Windows can rewrite line endings and editors can expand tabs; either turns
    a valid Makefile into one GNU make rejects with "missing separator". Catching it
    here is cheaper than catching it in a CI run.
    """
    offenders = [
        (number, line)
        for number, line in enumerate(
            MAKEFILE.read_text(encoding="utf-8").splitlines(), start=1
        )
        if line.startswith("    ")
    ]
    assert not offenders, f"space-indented recipe lines: {offenders}"


def test_unimplemented_targets_are_declared_as_failing() -> None:
    """The M6 placeholder targets exit non-zero rather than silently succeeding.

    Guardrail 1 forbids anything that looks like a benchmark result but is not one.
    A ``make bench`` that prints nothing and returns success is exactly that.
    """
    makefile_text = MAKEFILE.read_text(encoding="utf-8")
    for target in ("bench", "bench-fast", "figures"):
        recipe = makefile_text.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0]
        assert "not implemented" in recipe
        assert "exit 1" in recipe
