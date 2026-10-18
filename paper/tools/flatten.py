"""Package the manuscript for submission.

Writes submission/sn-article.tex with every \\input expanded (the template forbids \\input), internal LaTeX
comments removed, \\val{key} replaced by its literal from generated/values.tex, \\esm{label} and \\esmref{label}
by their Online Resource numbers from generated/esm.tex, and \\claim{ID} dropped (the submission uses no custom
macros), and copies the class, style, bibliography, the figures it uses and Online
Resource 1 next to it.
Run after `uv run python paper/build.py sn-article supplementary`:  uv run python paper/tools/flatten.py
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'submission'
INPUT = re.compile(r'^\s*\\input\{([^}]+)\}\s*$')
COMMENT = re.compile(r'(?<!\\)%')
DEFVAL = re.compile(r'^\\defval\{([^}]+)\}\{(.*)\}$')
VAL = re.compile(r'\\val\{([^}]+)\}')
DEFESM = re.compile(r'^\\defesm\{([^}]+)\}\{([^}]*)\}\{([^}]*)\}$')
ESM = re.compile(r'\\esm(ref)?\{([^}]+)\}')
GENERATED = ('generated/values', 'generated/esm')
CLAIM = re.compile(r'\\claim\{[^}]+\}')
CLAIMS_SETUP = re.compile(r'^\s*(\\usepackage\{claims\}%?|\\showclaims(false|true))\s*$')


def load_values(path):
    """Reads the \\defval lines of generated/values.tex into a key -> literal mapping."""
    values = {}
    for line in path.read_text(encoding='utf-8').split('\n'):
        m = DEFVAL.match(line)
        if m:
            values[m.group(1)] = m.group(2)
    return values


def load_esm(path):
    """Reads the \\defesm lines of generated/esm.tex into a label -> (kind, number) mapping."""
    labels = {}
    for line in path.read_text(encoding='utf-8').split('\n'):
        m = DEFESM.match(line)
        if m:
            labels[m.group(1)] = (m.group(2), m.group(3))
    return labels


def esm_literal(m, esm):
    """The literal of one \\esm{label} or \\esmref{label} match."""
    kind, number = esm[m.group(2)]
    return number if m.group(1) else f'Online Resource~1, {kind}~{number}'


def strip_macros(line, values, esm=None):
    """Removes \\claim{ID} and replaces \\val{key}, \\esm{label} and \\esmref{label} by their literals.

    An unknown key or label raises KeyError.
    """
    line = CLAIM.sub('', line)
    line = ESM.sub(lambda m: esm_literal(m, esm or {}), line)
    return VAL.sub(lambda m: values[m.group(1)], line)


def strip_comment(line):
    """Drop a comment but keep its '%' when text precedes it, so the end-of-line space is still suppressed."""
    m = COMMENT.search(line)
    if not m:
        return line
    head = line[:m.start()]
    if not head.strip():
        return None  # full-line comment: drop the line
    return head + '%'


def expand(path, values, depth=0, esm=None):
    assert depth < 5, path
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n')
    out = []
    for line in text.split('\n'):
        if CLAIMS_SETUP.match(line):
            continue
        m = INPUT.match(line)
        if m and m.group(1) in GENERATED:
            continue
        if m:
            name = m.group(1)
            sub = ROOT / (name if name.endswith('.tex') else name + '.tex')
            out.extend(expand(sub, values, depth + 1, esm))
            continue
        s = strip_comment(line)
        if s is not None:
            out.append(strip_macros(s, values, esm))
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'figures').mkdir(parents=True)
    values = load_values(ROOT / 'generated' / 'values.tex')
    esm = load_esm(ROOT / 'generated' / 'esm.tex')
    lines = expand(ROOT / 'sn-article.tex', values, esm=esm)
    body = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)) + '\n'
    assert '\\input{' not in body
    assert not re.search(r'\\(val|claim|defval|esm|esmref|defesm)\{', body)
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
