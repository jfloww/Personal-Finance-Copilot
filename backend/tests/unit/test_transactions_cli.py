from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres import engine as pg_engine
from offerdelta.ingest.mapping import AmountSign
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


def test_commit_reports_a_missing_connection_string_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_engine() raises RuntimeError when CONNECTION_STRING is unset; main()
    must turn that into a clean message and a non-zero exit, not an uncaught
    traceback. Needs no database: the RuntimeError fires before any connection
    is attempted, while building the engine's DSN.
    """
    monkeypatch.setenv("CONNECTION_STRING", "")
    get_settings.cache_clear()
    pg_engine.get_engine.cache_clear()
    try:
        code = main(
            [
                "commit",
                str(_file(tmp_path)),
                "--account=checking",
                "--mode=snapshot",
                "--from=2026-08-01",
                "--to=2026-08-31",
                "--yes",
            ]
        )
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()
        pg_engine.get_engine.cache_clear()

    assert code == 1
    assert "CONNECTION_STRING" in out
    assert "Traceback" not in out


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


# ---------------------------------------------------------------- add


def test_add_requires_every_field() -> None:
    parser = build_parser()
    for missing in (
        ["add", "--date=2026-08-17", "--description=x", "--amount=-1"],
        ["add", "--account=checking", "--description=x", "--amount=-1"],
        ["add", "--account=checking", "--date=2026-08-17", "--amount=-1"],
        ["add", "--account=checking", "--date=2026-08-17", "--description=x"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(missing)


def test_add_rejects_a_bad_date() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["add", "--account=checking", "--date=17/08/2026", "--description=x", "--amount=-1"]
        )


def test_add_rejects_an_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "add",
                "--account=checking",
                "--date=2026-08-17",
                "--description=x",
                "--amount=-1",
                "--force",
            ]
        )


def test_add_writes_nothing_without_yes(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same gate as commit: a write cannot happen by accident."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(
        [
            "add",
            "--account=checking",
            "--date=2026-08-17",
            "--description=Blue Bottle",
            "--amount=-4.50",
        ]
    )
    out = capsys.readouterr().out
    assert code != 0
    assert "refusing to write" in out


def test_add_prints_the_summary_before_asking(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    main(
        [
            "add",
            "--account=checking",
            "--date=2026-08-17",
            "--description=Blue Bottle",
            "--amount=-4.50",
            "--repeat",
        ]
    )
    out = capsys.readouterr().out
    assert "2026-08-17" in out
    assert "Blue Bottle" in out
    assert "checking" in out
    assert "(repeat)" in out


# ---------------------------------------------------------------- --sign wiring


def test_commit_passes_the_sign_convention_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag that parses but never reaches the request is worse than no flag.

    This exact wiring was missed once: --sign worked on preview, was accepted
    by commit, and was silently dropped before the import - so 245 Amex rows
    imported with every charge recorded as income.
    """
    captured: dict[str, object] = {}

    def _fake_import_csv(_session: object, request: object) -> object:
        captured["sign"] = request.amount_sign  # type: ignore[attr-defined]
        raise ValidationError("stop here; the request is what we are testing")

    monkeypatch.setattr("transactions.import_csv", _fake_import_csv)
    monkeypatch.setattr("transactions.get_engine", lambda: None)
    monkeypatch.setattr("transactions.Session", lambda _engine: _NullSession())

    main(
        [
            "commit",
            str(_file(tmp_path)),
            "--account=checking",
            "--mode=snapshot",
            "--from=2026-08-01",
            "--to=2026-08-31",
            "--sign=outflow-positive",
            "--yes",
        ]
    )

    assert captured["sign"] is AmountSign.OUTFLOW_POSITIVE


def test_commit_defaults_to_no_sign_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_import_csv(_session: object, request: object) -> object:
        captured["sign"] = request.amount_sign  # type: ignore[attr-defined]
        raise ValidationError("stop here")

    monkeypatch.setattr("transactions.import_csv", _fake_import_csv)
    monkeypatch.setattr("transactions.get_engine", lambda: None)
    monkeypatch.setattr("transactions.Session", lambda _engine: _NullSession())

    main(
        [
            "commit",
            str(_file(tmp_path)),
            "--account=checking",
            "--mode=snapshot",
            "--from=2026-08-01",
            "--to=2026-08-31",
            "--yes",
        ]
    )

    assert captured["sign"] is None


class _NullSession:
    """Enough Session for the CLI's `with` block; the import is faked out."""

    def __enter__(self) -> _NullSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def commit(self) -> None:
        return None
