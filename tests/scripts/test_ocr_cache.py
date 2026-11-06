from pathlib import Path

import pytest

from graphragexpr.cache import atomic_write_text
from scripts import preprocess_pdf as ocr


class Document:
    page_count = 1
    def __init__(self, path): self.content = path.read_text()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __getitem__(self, index): return self.content


@pytest.fixture
def backend(monkeypatch):
    calls = []
    monkeypatch.setattr(ocr, '_open_pdf', Document)
    monkeypatch.setattr(ocr, '_create_engine', lambda: object())
    monkeypatch.setattr(ocr, '_backend_versions', lambda: {'engine': 'v1'})
    def recognize(page, engine, dpi, min_score):
        calls.append((page, dpi, min_score))
        return [page]
    monkeypatch.setattr(ocr, 'ocr_page', recognize)
    return calls


def run(monkeypatch, source, output, *extra):
    monkeypatch.setattr('sys.argv', ['preprocess_pdf.py', '--pdf', str(source), '--output', str(output), *extra])
    assert ocr.main() == 0


def test_different_pdfs_do_not_share_page_cache(tmp_path, monkeypatch, backend):
    first, second = tmp_path / 'first.pdf', tmp_path / 'second.pdf'
    first.write_text('FIRST TEXT')
    second.write_text('SECOND TEXT')
    run(monkeypatch, first, tmp_path / 'first.md')
    run(monkeypatch, second, tmp_path / 'second.md')
    text = (tmp_path / 'second.md').read_text()
    assert 'SECOND TEXT' in text and 'FIRST TEXT' not in text
    assert len(backend) == 2
    # Resume need not construct the expensive OCR engine.
    monkeypatch.setattr(ocr, '_create_engine', lambda: pytest.fail('cache should be reused'))
    run(monkeypatch, second, tmp_path / 'resumed.md')
    assert len(backend) == 2


@pytest.mark.parametrize('change', ['content', 'dpi', 'score', 'version'])
def test_cache_follows_input_and_ocr_configuration(tmp_path, monkeypatch, backend, change):
    source = tmp_path / 'text.pdf'; source.write_text('FIRST')
    output = tmp_path / 'text.md'
    run(monkeypatch, source, output)
    args = ()
    if change == 'content': source.write_text('SECOND')
    if change == 'dpi': args = ('--dpi', '200')
    if change == 'score': args = ('--min-score', '0.7')
    if change == 'version': monkeypatch.setattr(ocr, '_backend_versions', lambda: {'engine': 'v2'})
    run(monkeypatch, source, output, *args)
    assert len(backend) == 2


def test_interrupted_refresh_keeps_completed_cache_and_output(tmp_path, monkeypatch, backend):
    source = tmp_path / 'text.pdf'; source.write_text('FIRST')
    output = tmp_path / 'text.md'
    run(monkeypatch, source, output)
    cache = next((tmp_path / '.ocr-pages').glob('*/page-*.md'))
    before = (output.read_bytes(), cache.read_bytes())
    def interrupt(*args): raise KeyboardInterrupt
    monkeypatch.setattr(ocr, 'ocr_page', interrupt)
    with pytest.raises(KeyboardInterrupt): run(monkeypatch, source, output, '--force')
    assert (output.read_bytes(), cache.read_bytes()) == before


def test_atomic_write_preserves_previous_file_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / 'page.md'; path.write_text('complete')
    def fail(*args): raise OSError('disk error')
    monkeypatch.setattr(Path, 'replace', fail)
    with pytest.raises(OSError): atomic_write_text(path, 'replacement')
    assert path.read_text() == 'complete'
    assert list(tmp_path.iterdir()) == [path]
