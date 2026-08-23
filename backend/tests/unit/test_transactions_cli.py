from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from offerdelta.config import get_settings
from transactions import build_parser, main

HEADER = "Date,Description,Amount\n"


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "aug.csv"
    path.write_text(HEADER + "2026-08-17,BLUE BOTTLE,-4.50\n", encoding="utf-8")
    return path


def test_an_unknown_flag_is_an_error(tmp_path: Path) -> None:
    """The bug: --dayfirst was silently discarded and dates committed wrong."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["preview", str(_file(tmp_path)), "--dayfirst"])


def test_a_misspelled_map_flag_is_an_error(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["preview", str(_file(tmp_path)), "--mapping=Date:Date"])


def test_commit_requires_a_mode(tmp_path: Path) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["commit", str(_file(tmp_path)), "--account=checking", "--yes"])


def test_snapshot_commit_requires_a_window(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["commit", str(_file(tmp_path)), "--account=checking", "--mode=snapshot", "--yes"]
    )
    assert args.window_start is None  # the service refuses; parser does not guess


def test_preview_never_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Preview is read-only. It has no --yes and no commit path."""
    code = main(["preview", str(_file(tmp_path)), "--dates=ISO"])
    out = capsys.readouterr().out
    assert code == 0
    assert "committed" not in out.lower()


def test_commit_without_yes_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old CLI rendered a preview and committed in the same breath."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(
        [
            "commit",
            str(_file(tmp_path)),
            "--account=checking",
            "--mode=snapshot",
            "--from=2026-08-01",
            "--to=2026-08-31",
        ]
    )
    out = capsys.readouterr().out
    assert code != 0
    assert "--yes" in out


def test_commit_does_not_render_the_preview_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    main(
        [
            "commit",
            str(_file(tmp_path)),
            "--account=checking",
            "--mode=snapshot",
            "--from=2026-08-01",
            "--to=2026-08-31",
        ]
    )
    out = capsys.readouterr().out
    assert "merchant" not in out  # the preview table header


def test_a_valid_map_flag_is_used_for_real(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A well-formed --map builds a ColumnMapping that preview_csv actually uses.

    No database needed: preview never touches one.
    """
    code = main(
        [
            "preview",
            str(_file(tmp_path)),
            "--map=date:Date,description:Description,amount:Amount",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "parsed 1, failed 0" in out
    assert "BLUE BOTTLE" in out


def test_commit_prints_the_summary_even_with_yes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unattended run must still say what it is about to do.

    The account does not exist, so `import_csv` raises `ValidationError` and
    `main` returns 1 before any write -- this only needs a read against the
    account table, never a write, to prove the summary already printed.
    """
    if not get_settings().database_available:
        pytest.skip("CONNECTION_STRING is not set; needs a live PostgreSQL")

    account = f"nonexistent-{uuid.uuid4().hex[:12]}"
    code = main(
        [
            "commit",
            str(_file(tmp_path)),
            f"--account={account}",
            "--mode=snapshot",
            "--from=2026-08-01",
            "--to=2026-08-31",
            "--yes",
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "aug.csv" in out
    assert account in out
    assert "snapshot" in out
    assert "2026-08-01" in out
    assert "2026-08-31" in out


def test_commit_refuses_when_stdin_gives_no_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """isatty() can wrongly report a terminal; EOF must still refuse, not crash."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def _raise_eof(*_: object) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)

    code = main(
        [
            "commit",
            str(_file(tmp_path)),
            "--account=checking",
            "--mode=snapshot",
            "--from=2026-08-01",
            "--to=2026-08-31",
        ]
    )

    assert code != 0
    assert "refusing to write" in capsys.readouterr().out
