"""Resumo das frases de voz de um log do app, para escolher ``voice.min_confidence``.

O microfone grava uma linha por frase fechada (``voz: frase=... confianca=... pico=...``).
Para cada frase reconhecida, a tabela mostra quantas vezes ela apareceu, a confiança da
palavra mais fraca (mínima, mediana e máxima), o pico de nível e quantas teriam virado
comando com cada limiar candidato. Palavras soltas do vocabulário ("pausar" sem
"controle") aparecem como frase incompleta: nenhum limiar as transforma em comando.

Uso::

    python -m inmotion --game minecraft --profile profiles/voz.yaml --log-file voz_2m.log
    python -m tools.voice_report logs/voz_2m.log
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from inmotion.voice import VoiceConfig

THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
_LINE = re.compile(
    r'voz: frase="(?P<text>[^"]*)" confianca=(?P<conf>[\d.]+|-) '
    r"palavras=\S+ aceita=(?P<ok>sim|nao) pico=(?P<peak>[\d.]+)"
    r"(?: vocabulario=(?P<known>sim|nao))?"
)


@dataclass(frozen=True)
class Heard:
    """Uma frase fechada pelo reconhecedor."""

    text: str
    confidence: float | None
    accepted: bool
    peak: float
    in_vocabulary: bool | None = None  # None: log antigo, sem o campo


def parse(lines: Sequence[str]) -> list[Heard]:
    """As frases do log, na ordem; linhas que não são de voz são ignoradas."""
    out = []
    for line in lines:
        m = _LINE.search(line)
        if m is None:
            continue
        conf = None if m["conf"] == "-" else float(m["conf"])
        known = None if m["known"] is None else m["known"] == "sim"
        out.append(Heard(m["text"], conf, m["ok"] == "sim", float(m["peak"]), known))
    return out


def _num(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def _is_command(text: str, items: Sequence[Heard], vocabulary: frozenset[str]) -> bool:
    """Se a frase é um comando: pelo campo do log ou, em logs antigos, pelo vocabulário."""
    marked = [h.in_vocabulary for h in items if h.in_vocabulary is not None]
    return marked[0] if marked else text in vocabulary


def summary(
    heard: Sequence[Heard],
    thresholds: Sequence[float] = THRESHOLDS,
    vocabulary: frozenset[str] | None = None,
) -> str:
    """Tabela Markdown por frase reconhecida.

    Só frases de comando entram nas colunas de limiar: palavras soltas do vocabulário
    ("pausar" sem "controle") e ``[unk]`` nunca viram comando, com qualquer limiar.
    """
    vocab = vocabulary if vocabulary is not None else frozenset(VoiceConfig().vocabulary)
    groups: dict[str, list[Heard]] = defaultdict(list)
    for h in heard:
        groups[h.text].append(h)
    head = ["frase", "comando?", "vezes", "conf. mín", "mediana", "máx", "pico (mediana)"]
    head += [f"≥ {_num(t)}" for t in thresholds]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = sorted(groups, key=lambda k: (k == "[unk]", not _is_command(k, groups[k], vocab), k))
    partial = 0
    for text in order:
        items = groups[text]
        confs = [h.confidence for h in items if h.confidence is not None]
        peak = _num(statistics.median(h.peak for h in items))
        if text == "[unk]":
            row = [text, "não", str(len(items)), "—", "—", "—", peak]
            lines.append("| " + " | ".join(row + ["—"] * len(thresholds)) + " |")
            continue
        command = _is_command(text, items, vocab)
        stats = (
            [_num(min(confs)), _num(statistics.median(confs)), _num(max(confs))]
            if confs
            else ["—", "—", "—"]
        )
        label = "sim" if command else "não (frase incompleta)"
        row = [f'"{text}"', label, str(len(items)), *stats, peak]
        if command and confs:
            row += [f"{sum(c >= t for c in confs)}/{len(confs)}" for t in thresholds]
        else:
            partial += 0 if command else len(items)
            row += ["—"] * len(thresholds)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"frases incompletas: {partial} de {len(heard)}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumo das frases de voz de um log.")
    parser.add_argument("log", type=Path, help="arquivo de log do app (logs/NOME)")
    args = parser.parse_args(argv)
    try:
        lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        sys.stderr.write(f"não consegui ler {args.log}: {exc}\n")
        return 1
    heard = parse(lines)
    if not heard:
        sys.stderr.write("nenhuma frase de voz no log (rodou com --profile da voz?)\n")
        return 1
    sys.stdout.write(summary(heard))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
