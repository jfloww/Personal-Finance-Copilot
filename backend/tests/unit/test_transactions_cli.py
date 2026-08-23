from __future__ import annotations

from pathlib import Path

import pytest

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
