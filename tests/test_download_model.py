"""Testes do script de download do modelo, sem rede (spec 003 T4, RF4)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from tools.download_model import main

PAYLOAD = b"conteudo-falso-do-modelo" * 100


class FakeResponse(io.BytesIO):
    """Imita a resposta de ``urlopen`` (gerenciador de contexto com ``read``)."""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeUrlopen:
    """``urlopen`` falso: registra as URLs pedidas e devolve ``PAYLOAD``."""

    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: Any, *args: Any, **kwargs: Any) -> FakeResponse:
        self.urls.append(url if isinstance(url, str) else url.full_url)
        return FakeResponse(self.payload)


def test_writes_to_models_dir(tmp_path: Path) -> None:
    fake = FakeUrlopen()
    code = main(["--models-dir", str(tmp_path / "models"), "--variant", "lite"], urlopen=fake)
    assert code == 0
    target = tmp_path / "models" / "pose_landmarker_lite.task"
    assert target.read_bytes() == PAYLOAD
    assert fake.urls == [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    ]


def test_default_variant_is_full(tmp_path: Path) -> None:
    code = main(["--models-dir", str(tmp_path)], urlopen=FakeUrlopen())
    assert code == 0
    assert (tmp_path / "pose_landmarker_full.task").is_file()


def test_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "pose_landmarker_full.task"
    target.write_bytes(b"antigo")
    fake = FakeUrlopen()
    code = main(["--models-dir", str(tmp_path)], urlopen=fake)
    assert code != 0
    assert target.read_bytes() == b"antigo"
    assert fake.urls == []  # nem tenta baixar


def test_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "pose_landmarker_full.task"
    target.write_bytes(b"antigo")
    code = main(["--models-dir", str(tmp_path), "--force"], urlopen=FakeUrlopen())
    assert code == 0
    assert target.read_bytes() == PAYLOAD


def test_custom_url(tmp_path: Path) -> None:
    fake = FakeUrlopen()
    main(["--models-dir", str(tmp_path), "--url", "https://exemplo.test/m.task"], urlopen=fake)
    assert fake.urls == ["https://exemplo.test/m.task"]


def test_invalid_variant_exits_with_usage_error(tmp_path: Path) -> None:
    fake = FakeUrlopen()
    with pytest.raises(SystemExit) as info:
        main(["--models-dir", str(tmp_path), "--variant", "ultra"], urlopen=fake)
    assert info.value.code == 2
    assert fake.urls == []


def test_network_error_returns_failure_and_leaves_no_partial_file(tmp_path: Path) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> FakeResponse:
        raise OSError("sem rede")

    code = main(["--models-dir", str(tmp_path)], urlopen=broken)
    assert code != 0
    assert list(tmp_path.iterdir()) == []


def test_empty_download_preserves_previous_model(tmp_path: Path) -> None:
    target = tmp_path / "pose_landmarker_full.task"
    target.write_bytes(b"antigo")
    code = main(["--models-dir", str(tmp_path), "--force"], urlopen=FakeUrlopen(b""))
    assert code == 1
    assert target.read_bytes() == b"antigo"
    assert not target.with_suffix(".task.part").exists()
