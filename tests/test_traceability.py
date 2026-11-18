"""Spec M3 — rastreabilidade fórmula → código (CA1)."""

from __future__ import annotations

from pathlib import Path

from tools.traceability import (
    REQUIRED,
    Citation,
    codes_in,
    collect_citations,
    main,
    missing_codes,
    to_markdown,
)

ROOT = Path(__file__).resolve().parents[1] / "inmotion"


def test_codes_in_expands_ranges_and_ignores_lookalikes() -> None:
    assert codes_in("R1-R3 e A7/A8") == ["R1", "R2", "R3", "A7", "A8"]
    assert codes_in("A2" + chr(0x2013) + "A9") == [f"A{i}" for i in range(2, 10)]
    assert codes_in("RF6, CA1, R10, xA1") == []  # requisito, critério e números maiores
    assert codes_in("A6 (A6)") == ["A6"]


def test_collector_finds_codes_in_docstrings(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "m.py").write_text(
        '"""Módulo de A1."""\n\n'
        "def f():\n"
        '    """A6: normaliza com M = S·R·T."""\n\n'
        "class C:\n"
        '    """Classe sem fórmula."""\n\n'
        "    def g(self):\n"
        '        """R5 e R6 no método."""\n',
        encoding="utf-8",
    )
    (pkg / "sub" / "n.py").write_text("x = 1  # A9 em comentário não conta\n", encoding="utf-8")
    cites = collect_citations(pkg)
    assert cites == [
        Citation("A1", "pkg/m.py", "<módulo>", "Módulo de A1."),
        Citation("A6", "pkg/m.py", "f", "A6: normaliza com M = S·R·T."),
        Citation("R5", "pkg/m.py", "C.g", "R5 e R6 no método."),
        Citation("R6", "pkg/m.py", "C.g", "R5 e R6 no método."),
    ]
    assert missing_codes(cites, ["A1", "A9", "R6"]) == ["A9"]


def test_all_model_codes_cited() -> None:
    """CA1: cada A1-A9 e R1-R7 do ROADMAP §5 aparece numa docstring de ``inmotion/``."""
    assert missing_codes(collect_citations(ROOT), REQUIRED) == []


def test_markdown_table_has_all_codes() -> None:
    table = to_markdown(collect_citations(ROOT))
    assert table.startswith("| Código | Arquivo | Função | Resumo |")
    for code in REQUIRED:
        assert f"| {code} | `inmotion/" in table


def test_markdown_prefers_functions_over_module_docstrings() -> None:
    cites = [
        Citation("A1", "p/m.py", "<módulo>", "Cabeçalho."),
        Citation("A1", "p/m.py", "f", "A1 | com barra."),
        Citation("R1", "p/e.py", "<módulo>", "Só no módulo."),
    ]
    lines = to_markdown(cites).splitlines()[2:]
    assert lines == [
        "| A1 | `p/m.py` | `f` | A1 \\| com barra. |",
        "| R1 | `p/e.py` | `<módulo>` | Só no módulo. |",
    ]


def test_main_writes_table_and_fails_on_missing(tmp_path: Path) -> None:
    out = tmp_path / "t.md"
    assert main(["--root", str(ROOT), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("| Código")
    pkg = tmp_path / "vazio"
    pkg.mkdir()
    (pkg / "m.py").write_text('"""Sem fórmulas."""\n', encoding="utf-8")
    assert main(["--root", str(pkg), "--out", str(tmp_path / "v.md")]) == 1
