"""Package the manuscript for submission.

Writes submission/sn-article.tex with every \\input expanded (the template forbids \\input) and internal LaTeX
comments removed, and copies the class, style, bibliography, the figures it uses and Online Resource 1 next to it.
Run after `uv run python paper/build.py sn-article supplementary`:  uv run python paper/tools/flatten.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'submission'
INPUT = re.compile(r'^\s*\\input\{([^}]+)\}\s*$')
COMMENT = re.compile(r'(?<!\\)%')


def strip_comment(line):
    """Drop a comment but keep its '%' when text precedes it, so the end-of-line space is still suppressed."""
    m = COMMENT.search(line)
    if not m:
        return line
    head = line[:m.start()]
    if not head.strip():
        return None  # full-line comment: drop the line
    return head + '%'


def expand(path, depth=0):
    assert depth < 5, path
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    out = []
    for line in text.split('\n'):
        m = INPUT.match(line)
        if m:
            name = m.group(1)
            sub = ROOT / (name if name.endswith('.tex') else name + '.tex')
            out.extend(expand(sub, depth + 1))
            continue
        s = strip_comment(line)
        if s is not None:
            out.append(s)
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'figures').mkdir(parents=True)
    lines = expand(ROOT / 'sn-article.tex')
    body = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)) + '\n'
    assert '\\input{' not in body
    (OUT / 'sn-article.tex').write_text(body, encoding='utf-8', newline='\n')
    for f in ['sn-jnl.cls', 'sn-mathphys-num.bst', 'sn-bibliography.bib']:
        shutil.copy(ROOT / f, OUT / f)
    for fig in sorted(set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', body))):
        src = ROOT / fig
        if not src.suffix:
            src = src.with_suffix('.pdf')
        shutil.copy(src, OUT / 'figures' / src.name)
    esm = ROOT / 'build' / 'supplementary.pdf'
    shutil.copy(esm, OUT / 'ESM_1.pdf')
    print('wrote', OUT / 'sn-article.tex', len(body.split('\n')), 'lines')


if __name__ == '__main__':
    main()
