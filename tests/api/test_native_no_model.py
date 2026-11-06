"""Native demo modes must remain usable without inventing a model answer."""

import pytest

from .conftest import NormalLLM, ask


@pytest.mark.parametrize('method', ['vector', 'keyword_graph'])
@pytest.mark.parametrize('token', [None, '  '])
def test_missing_model_configuration_returns_evidence_without_calling_model(
    client, wire, monkeypatch, method, token
):
    quote = '测试原文：这里只提供教材证据，不包含组合使用结论。'
    llm = NormalLLM(error=AssertionError('No provider should be initialized'))
    wire(llm=llm, native_chunks=[{
        'text': quote, 'page_start': 12, 'page_end': 13, 'matched_terms': ['测试对象'],
    }])
    if token is None:
        monkeypatch.delenv('LLM_TOKEN', raising=False)
    else:
        monkeypatch.setenv('LLM_TOKEN', token)
    status, data = ask(client, '请介绍测试对象的特点', method)
    assert status == 200
    assert quote in data['answer'] and 'PDF 第 12-13 页' in data['answer']
    assert '未生成或核验回答' in data['answer']
    assert llm.constructed == llm.calls == 0
    assert data['method'] == method
    assert not data.get('claims')


@pytest.mark.parametrize('method', ['vector', 'keyword_graph'])
def test_configured_model_keeps_the_existing_generation_path(client, wire, method):
    model = NormalLLM(text='configured provider response')
    wire(llm=model)
    status, data = ask(client, '请介绍测试对象的特点', method)
    assert status == 200 and data['answer'] == 'configured provider response'
    assert model.calls == 1
