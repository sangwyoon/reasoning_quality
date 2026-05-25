"""Unified LLM client for OpenAI, LiteLLM, and vLLM."""

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Logprobs normalization (Together AI -> OpenAI format)
# ---------------------------------------------------------------------------

class _TokenLogprob:
    """Matches OpenAI top_logprobs item interface."""
    __slots__ = ("token", "logprob")

    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


class _TokenInfo:
    """Matches OpenAI logprobs.content item interface."""
    __slots__ = ("token", "logprob", "top_logprobs")

    def __init__(self, token, logprob, top_logprobs):
        self.token = token
        self.logprob = logprob
        self.top_logprobs = top_logprobs


class _Logprobs:
    """Matches OpenAI choices[].logprobs interface."""
    __slots__ = ("content",)

    def __init__(self, content):
        self.content = content


def _normalize_logprobs(choice):
    """Convert Together AI logprobs format to OpenAI format.

    Together AI format:
        logprobs.tokens = ["4", "<|end|>"]
        logprobs.token_logprobs = [0.0, -0.001]
        logprobs.top_logprobs = [{"4": 0.0, "2": -26.0}, ...]

    OpenAI format:
        logprobs.content = [TokenInfo(token="4", logprob=0.0, top_logprobs=[...])]
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return

    # Skip if already OpenAI format
    content = getattr(lp, "content", None)
    if content is not None and isinstance(content, list) and len(content) > 0:
        return

    # Check Together AI format
    tokens = getattr(lp, "tokens", None)
    token_logprobs = getattr(lp, "token_logprobs", None)
    top_lps = getattr(lp, "top_logprobs", None)

    if tokens is None or token_logprobs is None:
        return

    normalized = []
    for i, (tok, tok_lp) in enumerate(zip(tokens, token_logprobs)):
        # Exclude EOS token
        if tok.startswith("<|") and tok.endswith("|>"):
            continue

        alts = []
        if top_lps and i < len(top_lps) and isinstance(top_lps[i], dict):
            for alt_tok, alt_lp in top_lps[i].items():
                if alt_tok.startswith("<|") and alt_tok.endswith("|>"):
                    continue
                alts.append(_TokenLogprob(alt_tok, alt_lp))

        normalized.append(_TokenInfo(tok, tok_lp, alts))

    choice.logprobs = _Logprobs(normalized if normalized else None)


# ---------------------------------------------------------------------------
# Client creation
# ---------------------------------------------------------------------------

def create_client(provider="openai"):
    """Return an OpenAI-compatible client for the given provider.

    Args:
        provider: "openai" | "litellm" | "vllm"

    Returns:
        client: object supporting client.chat.completions.create()
    """
    if provider == "openai":
        from openai import OpenAI
        return OpenAI()

    if provider == "litellm":
        return _LiteLLMClient()

    if provider == "vllm":
        return _VLLMClient()

    raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'litellm', or 'vllm'.")


class _VLLMClient:
    """vLLM server wrapper. Auto-injects enable_thinking=false for thinking models."""

    def __init__(self):
        from openai import OpenAI
        base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
        self._client = OpenAI(base_url=base_url, api_key="dummy")
        self.chat = self._Chat(self._client)
        self.completions = self._client.completions

    class _Chat:
        def __init__(self, client):
            self.completions = self._Completions(client)

        class _Completions:
            def __init__(self, client):
                self._client = client

            def create(self, **kwargs):
                # Preserve existing extra_body while disabling thinking
                extra_body = kwargs.pop("extra_body", {}) or {}
                if "chat_template_kwargs" not in extra_body:
                    # Only disable thinking when continue_final_message is not set
                    if "continue_final_message" not in extra_body:
                        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                kwargs["extra_body"] = extra_body
                return self._client.chat.completions.create(**kwargs)


class _LiteLLMClient:
    """Wrapper around litellm.completion() with OpenAI client interface.

    Usage is identical to OpenAI() client:
        client = _LiteLLMClient()
        resp = client.chat.completions.create(model=..., messages=..., ...)
    """

    def __init__(self):
        self.chat = self._Chat()

    class _Chat:
        def __init__(self):
            self.completions = self._Completions()

        class _Completions:
            @staticmethod
            def create(**kwargs):
                import litellm
                litellm.drop_params = True
                resp = litellm.completion(**kwargs)

                # Normalize logprobs format
                if resp.choices:
                    for choice in resp.choices:
                        _normalize_logprobs(choice)

                return resp


# ---------------------------------------------------------------------------
# Model tag (filesystem safe)
# ---------------------------------------------------------------------------

def make_model_tag(model: str) -> str:
    """Generate a filesystem-safe tag from model name.

    Examples:
        gpt-4.1-nano         -> gpt-41-nano
        openrouter/qwen/qwen3-235b-a22b -> qwen3-235b-a22b
        together_ai/Qwen/Qwen3-32B      -> Qwen3-32B
    """
    name = model.split("/")[-1]
    name = name.replace(".", "")
    return name


# ---------------------------------------------------------------------------
# Tokenizer (for teacher forcing)
# ---------------------------------------------------------------------------

def get_tokenizer(model: str):
    """Return tokenizer for the model. Falls back to cl100k_base."""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, Exception):
        return tiktoken.get_encoding("cl100k_base")
