"""Rastreabilidade fórmula → código (spec M3): onde cada A1-A9 e R1-R7 é citado.

Percorre as docstrings de ``inmotion/`` (módulos, classes, funções e métodos) com ``ast``
e procura os códigos do modelo matemático do ROADMAP §5. Faixas como ``R1-R3`` ou
``A1-A9`` (hífen ou meia-risca) contam para todos os códigos do intervalo.

Uso::

    python -m tools.traceability                    # tabela Markdown no terminal
    python -m tools.traceability --out reports/rastreabilidade.md
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("traceability")

DEFAULT_ROOT = Path("inmotion")
# A1-A9 (álgebra linear e imagem) e R1-R7 (recorrências), do ROADMAP §5.
REQUIRED: tuple[str, ...] = (*(f"A{i}" for i in range(1, 10)), *(f"R{i}" for i in range(1, 8)))
_EN_DASH = chr(0x2013)  # meia-risca, comum nas faixas das docstrings
_RANGE = re.compile(r"\b([AR])([1-9])\s*[-" + _EN_DASH + r"]\s*\1?([1-9])\b")
CODE_PATTERN = re.compile(r"\b([AR][1-9])\b")
_DocNode = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Citation:
    """Um código citado numa docstring: ``qualname`` é ``<módulo>`` para a do arquivo."""

    code: str
    path: str
    qualname: str
    summary: str


def codes_in(text: str) -> list[str]:
    """Códigos citados em ``text``, na ordem, sem repetir; faixas viram todos os códigos."""
    found: list[str] = []
    for m in _RANGE.finditer(text):
        letter, lo, hi = m.group(1), int(m.group(2)), int(m.group(3))
        found += [f"{letter}{i}" for i in range(lo, hi + 1)]
    found += CODE_PATTERN.findall(text)
    return list(dict.fromkeys(found))


def _documented(tree: ast.Module) -> Iterator[tuple[str, _DocNode]]:
    """``(qualname, nó)`` do módulo e de cada classe/função, com o nome qualificado."""
    yield "<módulo>", tree

    def walk(node: ast.AST, prefix: str) -> Iterator[tuple[str, _DocNode]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                yield name, child
                yield from walk(child, f"{name}.")

    yield from walk(tree, "")


def collect_citations(root: Path, base: Path | None = None) -> list[Citation]:
    """Todas as citações nas docstrings dos ``.py`` sob ``root`` (caminhos relativos a
    ``base``, padrão o diretório pai de ``root``), em ordem de arquivo e de linha."""
    base = base if base is not None else root.parent
    out: list[Citation] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(base).as_posix()
        for qualname, node in _documented(tree):
            doc = ast.get_docstring(node)
            if not doc:
                continue
            summary = doc.strip().splitlines()[0]
            out += [Citation(code, rel, qualname, summary) for code in codes_in(doc)]
    return out


def missing_codes(citations: Sequence[Citation], required: Iterable[str]) -> list[str]:
    """Códigos de ``required`` que não aparecem em nenhuma citação (na ordem dada)."""
    cited = {c.code for c in citations}
    return [code for code in required if code not in cited]


def _key(code: str) -> tuple[str, int]:
    return code[0], int(code[1:])


def to_markdown(citations: Sequence[Citation]) -> str:
    """Tabela ``código | arquivo | função | resumo``, ordenada por código e arquivo.

    Citações só no docstring do módulo aparecem apenas se o código não tiver nenhuma
    função que o cite (evita repetir o cabeçalho do arquivo em cada linha).
    """
    by_func = {c.code for c in citations if c.qualname != "<módulo>"}
    rows = [c for c in citations if c.qualname != "<módulo>" or c.code not in by_func]
    rows.sort(key=lambda c: (_key(c.code), c.path, c.qualname))
    lines = ["| Código | Arquivo | Função | Resumo |", "|---|---|---|---|"]
    for c in rows:
        summary = c.summary.replace("|", "\\|")
        lines.append(f"| {c.code} | `{c.path}` | `{c.qualname}` | {summary} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tabela fórmula → arquivo:função.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="pacote a percorrer")
    parser.add_argument("--out", type=Path, help="grava a tabela neste arquivo .md")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    citations = collect_citations(args.root)
    table = to_markdown(citations)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        logger.info("tabela: %s", args.out)
    else:
        sys.stdout.write(table)
    missing = missing_codes(citations, REQUIRED)
    if missing:
        logger.error("códigos sem citação: %s", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
