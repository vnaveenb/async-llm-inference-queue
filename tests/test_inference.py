"""Tests for the inference executor."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.inference import _mock_inference, _real_inference, run_inference
from src.models import InferenceJob


@pytest.fixture
def job():
    return InferenceJob(
        prompt="What is the capital of France?",
        model="gemini/gemini-2.5-flash",
        max_tokens=256,
        temperature=0.0,
    )


@pytest.mark.asyncio
async def test_mock_inference_returns_result(job):
    result = await _mock_inference(job)
    assert "result" in result
    assert isinstance(result["result"], str)
    assert result["input_tokens"] > 0
    assert result["output_tokens"] > 0
    assert result["latency_ms"] > 0


@pytest.mark.asyncio
async def test_mock_inference_latency_range(job):
    result = await _mock_inference(job)
    # Mock latency should be between 200ms and 3000ms
    assert 200 <= result["latency_ms"] <= 3000


@pytest.mark.asyncio
async def test_run_inference_uses_mock_mode(job):
    with patch("src.inference.get_config") as mock_cfg:
        mock_cfg.return_value.inference.mock = True
        mock_cfg.return_value.tracker.enabled = False
        result = await run_inference(job)
        assert "result" in result


@pytest.mark.asyncio
async def test_real_inference_calls_litellm(job):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Paris"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 15
    mock_response.usage.completion_tokens = 5

    with patch("src.inference.get_config") as mock_cfg:
        mock_cfg.return_value.inference.mock = False
        mock_cfg.return_value.inference.timeout_seconds = 30
        mock_cfg.return_value.tracker.enabled = False
        mock_cfg.return_value.tracker.project_id = "test"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await _real_inference(job)

    assert result["result"] == "Paris"
    assert result["input_tokens"] == 15
    assert result["output_tokens"] == 5
    assert result["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_real_inference_with_tracker_metadata(job):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 3

    with patch("src.inference.get_config") as mock_cfg:
        mock_cfg.return_value.inference.mock = False
        mock_cfg.return_value.inference.timeout_seconds = 30
        mock_cfg.return_value.tracker.enabled = True
        mock_cfg.return_value.tracker.project_id = "async-inference-queue"

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock_call:
            result = await _real_inference(job)

    # Verify metadata was passed for cost tracking
    call_kwargs = mock_call.call_args[1]
    assert call_kwargs["metadata"]["project_id"] == "async-inference-queue"
    assert call_kwargs["metadata"]["job_id"] == job.id
