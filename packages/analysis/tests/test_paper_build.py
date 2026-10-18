import importlib.util
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "paper"


def load_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location("paper_build", PAPER / "build.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_build()

AUX = "\n".join(
    [
        r"\relax",
        r"\newlabel{esm:method}{{S1}{1}{Method}{section.1}{}}",
        r"\newlabel{esm:setting}{{S2.1}{4}{Setting}{subsection.2.1}{}}",
        r"\newlabel{tab:s_a}{{S3}{2}{Rules of $(f, r, k)$ and {\em more}}{table.caption.5}{}}",
        r"\newlabel{fig:s_b}{{S1}{9}{Subsamples}{figure.caption.7}{}}",
        r"\newlabel{lem:s_c}{{4}{5}{}{lemma.4}{}}",
        r"\newlabel{eq:s_d}{{S2}{5}{}{equation.2.2}{}}",
        r"\newlabel{x:other}{{7}{5}{}{Item.7}{}}",
        r"\newlabel{x:short}{{7}{5}}",
    ]
)


def test_esm_labels_reads_kind_and_number_from_the_anchor() -> None:
    assert build.esm_labels(AUX) == {
        "esm:method": ("Sect.", "S1"),
        "esm:setting": ("Sect.", "S2.1"),
        "tab:s_a": ("Table", "S3"),
        "fig:s_b": ("Fig.", "S1"),
        "lem:s_c": ("Lemma", "4"),
        "eq:s_d": ("Eq.", "(S2)"),
    }


def test_esm_tex_defines_every_label_in_order() -> None:
    text = build.esm_tex({"tab:b": ("Table", "S2"), "esm:a": ("Sect.", "S1")})
    rows = [line for line in text.splitlines() if line.startswith(r"\defesm")]
    assert rows == [r"\defesm{esm:a}{Sect.}{S1}", r"\defesm{tab:b}{Table}{S2}"]
    assert r"\newcommand{\esm}" in text
