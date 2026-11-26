"""Tokenize / detokenize requests against a vLLM server; failures raise."""

import os


def base_url() -> str:
    """Server root: ``VLLM_API_BASE`` without a trailing "/v1"."""
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/")
    return base.removesuffix("/v1")


def _post(endpoint: str, payload: dict) -> dict:
    import requests
    resp = requests.post(f"{base_url()}/{endpoint}", json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def tokenize(model: str, text: str) -> list[int]:
    """Token ids of ``text`` without special tokens."""
    tokens = _post("tokenize", {"model": model, "prompt": text, "add_special_tokens": False}).get("tokens")
    if not tokens:
        raise ValueError(f"vLLM /tokenize returned no tokens for {text!r}")
    return tokens


def detokenize(model: str, token_ids: list[int]) -> list[str]:
    """The string of each token id, one request per id."""
    if not token_ids:
        raise ValueError("no token ids to detokenize")
    return [_post("detokenize", {"model": model, "tokens": [tid]})["prompt"] for tid in token_ids]
