import re

import pytest

from oran_causal.figures import cli, paths
from oran_causal.figures.plotted import pdf_layout


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.OUT_ENV, "")
    return tmp_path


def test_list_names_every_figure(capsys):
    assert cli.main(["list"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert any(line.startswith("fig_method_pmrt") for line in listed)
    assert any(line.startswith("specimen_bars") for line in listed)


def test_unknown_figure_is_refused(capsys):
    assert cli.main(["build", "fig_missing"]) == 2
    assert "unknown figure(s): fig_missing" in capsys.readouterr().err


def test_build_writes_pdf_and_preview_under_the_override(out_dir, capsys):
    assert cli.main(["build", "fig_method_pmrt", "--out", str(out_dir)]) == 0
    pdf = out_dir / "figures" / "fig_method_pmrt.pdf"
    assert pdf.is_file()
    assert (out_dir / "preview" / "fig_method_pmrt.png").is_file()
    assert str(pdf) in capsys.readouterr().out


def test_preview_only_skips_the_pdf(out_dir):
    assert cli.main(["build", "fig_method_pmrt", "--preview", "--out", str(out_dir)]) == 0
    assert not (out_dir / "figures").exists()


def test_specimens_never_reach_the_journal(out_dir):
    assert cli.main(["build", "specimen_bars", "--out", str(out_dir)]) == 0
    assert not (out_dir / "figures").exists()


def test_pdf_is_one_arial_page_of_the_declared_width(out_dir):
    cli.main(["build", "fig_system_e6p", "--out", str(out_dir)])
    data = (out_dir / "figures" / "fig_system_e6p.pdf").read_bytes()
    assert len(re.findall(rb"/Type\s*/Page\b", data)) == 1
    names = {
        re.sub(r"^[A-Z]{6}\+", "", m.decode())
        for m in re.findall(rb"/BaseFont\s*/([\w+\-,]+)", data)
    }
    assert names
    assert all(n.startswith("Arial") for n in names)
    box = [float(v) for v in re.search(rb"/MediaBox\s*\[([^\]]*)\]", data).group(1).split()]
    assert box[2] == pytest.approx(5.15 * 72, abs=2)


def test_pdf_layout_reads_the_same_page(build_figure):
    layout = pdf_layout(build_figure("fig_system_e6p"))
    assert layout["pages"] == 1
    assert all(f.startswith("Arial") for f in layout["fonts"])
