from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel, create_model

from graphragexpr.config.settings import KGSettings, RetrievalSettings
from graphragexpr.retrieval.router import route_query
from scripts import build_kg_v2 as build


class Result(BaseModel):
    value: str


class FakeClient:
    def __init__(self, model='first', endpoint='https://first.invalid'):
        self.cache_identity = {'model': model, 'endpoint': endpoint}
        self.calls = 0

    def complete_json(self, *, schema, **kwargs):
        self.calls += 1
        return schema(value=self.cache_identity['model'])


def test_settings_read_environment_at_instantiation(tmp_path, monkeypatch):
    env = tmp_path / '.env'
    env.write_text('KG_FINAL_ACCEPT=0.99\nKG_USE_I3=false\nRETRIEVAL_PATH_MAX_HOPS=2\n')
    for key in ('KG_FINAL_ACCEPT', 'KG_USE_I3', 'RETRIEVAL_PATH_MAX_HOPS'):
        monkeypatch.delenv(key, raising=False)
    assert KGSettings().use_cross_chunk is True
    load_dotenv(env, override=True)
    try:
        assert KGSettings().final_accept_threshold == 0.99
        assert KGSettings().use_cross_chunk is False
        assert KGSettings(use_cross_chunk=True).use_cross_chunk is True
        assert RetrievalSettings().path_max_hops == 2
        plan = route_query('人参和黄芪有什么关系？', [
            {'canonical_id': 'HERB:a', 'entity_type': 'HERB'},
            {'canonical_id': 'HERB:b', 'entity_type': 'HERB'},
        ])
        assert plan.max_hops == 2
    finally:
        # load_dotenv writes os.environ directly, so track its writes for fixture cleanup.
        for key in ('KG_FINAL_ACCEPT', 'KG_USE_I3', 'RETRIEVAL_PATH_MAX_HOPS'):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(build, 'CACHE_DIR', tmp_path)
    return tmp_path


def call(client, *, settings=None, system='system', schema=Result):
    wrapper = build.CachingLLMClient(client, settings or KGSettings())
    result = wrapper.complete_json(system=system, user='same input', schema=schema, purpose='extract')
    return result, wrapper


def test_identical_request_uses_cache(cache):
    first = FakeClient()
    call(first)
    second = FakeClient()
    result, wrapper = call(second)
    assert result.value == 'first'
    assert second.calls == 0
    assert wrapper.hits == 1


@pytest.mark.parametrize('changed', ['model', 'endpoint', 'system', 'temperature', 'schema'])
def test_generation_changes_invalidate_cache(cache, changed):
    call(FakeClient())
    client = FakeClient(model='second' if changed == 'model' else 'first',
                        endpoint='https://second.invalid' if changed == 'endpoint' else 'https://first.invalid')
    kwargs = {}
    if changed == 'system': kwargs['system'] = 'new instruction'
    if changed == 'temperature': kwargs['settings'] = KGSettings(llm_temperature=0.3)
    if changed == 'schema': kwargs['schema'] = create_model('Result', value=(str, ...), extra_value=(int, 1))
    call(client, **kwargs)
    assert client.calls == 1
    assert len(list(cache.glob('*.json'))) == 2


def test_corrupt_cache_is_recomputed(cache):
    call(FakeClient())
    path = next(cache.glob('*.json'))
    path.write_text('{incomplete')
    client = FakeClient()
    call(client)
    assert client.calls == 1
    assert Result.model_validate_json(path.read_text()).value == 'first'


def test_dimension_mismatch_fails_before_writes():
    store = SimpleNamespace(run=lambda *a, **kw: [{'dimensions': 768}, {'dimensions': 512}])
    with pytest.raises(ValueError, match='EMBED_DIM=768'):
        build.validate_embedding_dimensions(store, 768)


def test_build_uses_configured_dimension(tmp_path, monkeypatch):
    # Stop immediately after schema preparation to test the real CLI configuration path.
    class StopAfterSchema(Exception): pass
    store = SimpleNamespace(verify=lambda: None, close=lambda: None,
                            run=lambda *a, **kw: [{'dimensions': 768}])
    monkeypatch.setenv('EMBED_DIM', '768')
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'Neo4jStore', lambda: store)
    seen = []
    def schema(_store, **kwargs):
        seen.append(kwargs['embedding_dim'])
        raise StopAfterSchema
    monkeypatch.setattr(build, 'ensure_schema', schema)
    monkeypatch.setattr('sys.argv', ['build_kg_v2.py'])
    with pytest.raises(StopAfterSchema): build.main()
    assert seen == [768]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    from graphragexpr.domain.models import ChunkRecord
    events = []
    store = SimpleNamespace(verify=lambda: None, close=lambda: events.append('close'),
                            run=lambda *a, **kw: [])
    monkeypatch.setattr(build, 'ROOT', tmp_path)
    monkeypatch.setattr(build, 'CACHE_DIR', tmp_path / 'cache')
    monkeypatch.setattr(build, 'Neo4jStore', lambda: store)
    monkeypatch.setattr(build, 'ensure_schema', lambda *a, **kw: {})
    monkeypatch.setattr(build, 'check_vector_dimension', lambda *a, **kw: None)
    monkeypatch.setattr(build.run_store, 'start_run',
                        lambda *a, **kw: SimpleNamespace(run_id='new-run'))
    monkeypatch.setattr(build.run_store, 'mark_failed',
                        lambda *a: events.append(('failed', a[1])))
    monkeypatch.setattr(build.run_store, 'activate',
                        lambda *a, **kw: events.append(('active', a[1])))
    chunk = ChunkRecord(chunk_id='c1', document_id='d1', text='原文',
                        normalized_text='原文', page_start=1, page_end=1, content_hash='h1')
    monkeypatch.setattr(build, 'load_chunks', lambda *a: ([chunk], {'c1': 1}))
    monkeypatch.setattr(build, 'migrate_chunks', lambda *a: 1)
    monkeypatch.setattr(build, 'OpenAICompatibleClient', lambda *a: SimpleNamespace(
        usage=SimpleNamespace(calls=0, total_tokens=0)))
    outcome = SimpleNamespace(assertions=[], rejected=[], unknown_types=[], unknown_predicates=[])
    monkeypatch.setattr(build, 'ChunkExtractor', lambda **kw: SimpleNamespace(extract=lambda c: outcome))
    monkeypatch.setattr(build.assertion_store, 'upsert_entities', lambda *a, **kw: 0)
    monkeypatch.setattr(build.assertion_store, 'upsert_assertions', lambda *a, **kw: 0)
    monkeypatch.setattr(build.serving_projection, 'rebuild_projection', lambda *a, **kw: {})
    monkeypatch.setattr(build.assertion_store, 'quality_report',
                        lambda *a: {'accepted': 0, 'evidence_coverage': 1.0})
    monkeypatch.setattr('sys.argv', ['build_kg_v2.py', '--activate', '--workers', '1'])
    return events


def test_empty_extraction_cannot_publish_an_empty_graph(pipeline):
    assert build.main() == 1
    assert pipeline == [('failed', 'new-run'), 'close']


def test_empty_corpus_is_rejected_and_marked_failed(pipeline, monkeypatch):
    monkeypatch.setattr(build, 'load_chunks', lambda *a: ([], {}))
    with pytest.raises(RuntimeError, match='没有可用的 Chunk'): build.main()
    assert pipeline == [('failed', 'new-run'), 'close']


@pytest.mark.parametrize('error', [RuntimeError('read failed'), KeyboardInterrupt()])
def test_failed_build_is_marked_without_publishing(pipeline, monkeypatch, error):
    def fail(*args): raise error
    monkeypatch.setattr(build, 'load_chunks', fail)
    with pytest.raises(type(error)): build.main()
    assert pipeline == [('failed', 'new-run'), 'close']
