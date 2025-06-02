import pytest
from unittest.mock import patch, MagicMock
from models.gpt_model import generate_response_stream

def mock_streaming_response(chunks):
    mock_resp = MagicMock()
    mock_resp.iter_lines = lambda: (chunk.encode('utf-8') for chunk in chunks)
    return mock_resp

def test_generate_response_stream_success():
    context = ["Una foto di un cane che corre."]
    question = "Cosa sta facendo il cane?"

    chunks = [
        '{"response": "Il cane sta correndo in un campo."}',
        '{"response": " Sembra felice e in movimento."}',
        '{"done": true}',
        '[DONE]'
    ]

    with patch("models.gpt_model.requests.post") as mock_post:
        mock_post.return_value = mock_streaming_response(chunks)

        results = list(generate_response_stream(context, question, model="test-model", lang="it", max_tokens=50))

        assert any("cane" in r.lower() for r in results)
        assert not any("error" in r.lower() for r in results)
