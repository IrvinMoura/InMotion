"""Resumo das frases de voz de um log (calibração de ``voice.min_confidence``, 27/11/2026)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.voice_report import Heard, main, parse, summary

LOG = """\
INFO inmotion.app: Rodando
INFO inmotion.capture.microphone: voz: frase="abrir menu" confianca=0.97 palavras=1.00,0.97 aceita=sim pico=0.41
INFO inmotion.capture.microphone: voz: frase="abrir menu" confianca=0.72 palavras=0.72,1.00 aceita=nao pico=0.30
INFO inmotion.capture.microphone: voz: frase="abrir menu" confianca=0.85 palavras=1.00,0.85 aceita=nao pico=0.35
INFO inmotion.capture.microphone: voz: frase="[unk]" confianca=1.00 palavras=1.00 aceita=nao pico=0.20
INFO inmotion.capture.microphone: voz: frase="pausar" confianca=0.95 palavras=0.95 aceita=nao pico=0.64 vocabulario=nao
"""  # noqa: E501


def test_parse_reads_only_voice_lines() -> None:
    heard = parse(LOG.splitlines())
    assert heard[0] == Heard("abrir menu", 0.97, True, 0.41)
    assert [h.text for h in heard] == ["abrir menu", "abrir menu", "abrir menu", "[unk]", "pausar"]
    assert heard[0].in_vocabulary is None  # log antigo, sem o campo: decide pelo vocabulário
    assert heard[-1].in_vocabulary is False


def test_summary_counts_acceptance_per_threshold() -> None:
    table = summary(parse(LOG.splitlines()), thresholds=(0.7, 0.8, 0.9))
    rows = table.splitlines()
    assert rows[0].startswith("| frase | comando? | vezes | conf. mín | mediana | máx | pico")
    assert rows[2] == '| "abrir menu" | sim | 3 | 0,72 | 0,85 | 0,97 | 0,35 | 3/3 | 2/3 | 1/3 |'
    # Palavra solta: nunca vira comando, com qualquer limiar (dados de 27/11).
    assert (
        rows[3]
        == '| "pausar" | não (frase incompleta) | 1 | 0,95 | 0,95 | 0,95 | 0,64 | — | — | — |'
    )
    assert rows[4] == "| [unk] | não | 1 | — | — | — | 0,20 | — | — | — |"
    assert "frases incompletas: 1 de 5" in table


def test_main_prints_table_and_fails_without_voice_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "voz.log"
    log.write_text(LOG, encoding="utf-8")
    assert main([str(log)]) == 0
    assert '"abrir menu"' in capsys.readouterr().out
    empty = tmp_path / "vazio.log"
    empty.write_text("INFO nada\n", encoding="utf-8")
    assert main([str(empty)]) == 1
    assert main([str(tmp_path / "nao_existe.log")]) == 1
