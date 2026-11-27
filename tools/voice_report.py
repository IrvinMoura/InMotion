"""Resumo das frases de voz de um log do app, para escolher ``voice.min_confidence``.

O microfone grava uma linha por frase fechada (``voz: frase=... confianca=... pico=...``).
Para cada frase reconhecida, a tabela mostra quantas vezes ela apareceu, a confiança da
palavra mais fraca (mínima, mediana e máxima), o pico de nível e quantas teriam virado
comando com cada limiar candidato.

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

THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
_LINE = re.compile(
    r'voz: frase="(?P<text>[^"]*)" confianca=(?P<conf>[\d.]+|-) '
    r"palavras=\S+ aceita=(?P<ok>sim|nao) pico=(?P<peak>[\d.]+)"
)


@dataclass(frozen=True)
class Heard:
    """Uma frase fechada pelo reconhecedor."""

    text: str
    confidence: float | None
    accepted: bool
    peak: float


def parse(lines: Sequence[str]) -> list[Heard]:
    """As frases do log, na ordem; linhas que não são de voz são ignoradas."""
    out = []
    for line in lines:
        m = _LINE.search(line)
        if m is None:
            continue
        conf = None if m["conf"] == "-" else float(m["conf"])
        out.append(Heard(m["text"], conf, m["ok"] == "sim", float(m["peak"])))
    return out


def _num(v: float) -> str:
    return f"{v:.2f}".replace(".", ",")


def summary(heard: Sequence[Heard], thresholds: Sequence[float] = THRESHOLDS) -> str:
    """Tabela Markdown por frase reconhecida; ``[unk]`` (fora do vocabulário) só é contada."""
    groups: dict[str, list[Heard]] = defaultdict(list)
    for h in heard:
        groups[h.text].append(h)
    head = ["frase", "vezes", "conf. mín", "mediana", "máx", "pico (mediana)"]
    head += [f"≥ {_num(t)}" for t in thresholds]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for text in sorted(groups, key=lambda k: (k == "[unk]", k)):
        items = groups[text]
        confs = [h.confidence for h in items if h.confidence is not None]
        peak = _num(statistics.median(h.peak for h in items))
        if text == "[unk]" or not confs:
            row = [text, str(len(items)), "—", "—", "—", peak] + ["—"] * len(thresholds)
        else:
            row = [
                f'"{text}"',
                str(len(items)),
                _num(min(confs)),
                _num(statistics.median(confs)),
                _num(max(confs)),
                peak,
            ] + [f"{sum(c >= t for c in confs)}/{len(confs)}" for t in thresholds]
        lines.append("| " + " | ".join(row) + " |")
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
