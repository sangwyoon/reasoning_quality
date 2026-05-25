"""Unified LLM client: OpenAI 직접 호출과 LiteLLM (Together AI 등) 모두 지원."""

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Logprobs 정규화 (Together AI → OpenAI 포맷)
# ---------------------------------------------------------------------------

class _TokenLogprob:
    """OpenAI top_logprobs 개별 항목과 동일한 인터페이스."""
    __slots__ = ("token", "logprob")

    def __init__(self, token, logprob):
        self.token = token
        self.logprob = logprob


class _TokenInfo:
    """OpenAI logprobs.content 개별 항목과 동일한 인터페이스."""
    __slots__ = ("token", "logprob", "top_logprobs")

    def __init__(self, token, logprob, top_logprobs):
        self.token = token
        self.logprob = logprob
        self.top_logprobs = top_logprobs


class _Logprobs:
    """OpenAI choices[].logprobs 와 동일한 인터페이스."""
    __slots__ = ("content",)

    def __init__(self, content):
        self.content = content


def _normalize_logprobs(choice):
    """Together AI 등의 logprobs 포맷을 OpenAI 포맷으로 변환한다.

    Together AI 포맷:
        logprobs.tokens = ["4", "<|end|>"]
        logprobs.token_logprobs = [0.0, -0.001]
        logprobs.top_logprobs = [{"4": 0.0, "2": -26.0}, ...]

    OpenAI 포맷:
        logprobs.content = [TokenInfo(token="4", logprob=0.0, top_logprobs=[...])]
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return

    # 이미 OpenAI 포맷이면 (content가 list) 스킵
    content = getattr(lp, "content", None)
    if content is not None and isinstance(content, list) and len(content) > 0:
        return

    # Together AI 포맷 확인
    tokens = getattr(lp, "tokens", None)
    token_logprobs = getattr(lp, "token_logprobs", None)
    top_lps = getattr(lp, "top_logprobs", None)

    if tokens is None or token_logprobs is None:
        return

    normalized = []
    for i, (tok, tok_lp) in enumerate(zip(tokens, token_logprobs)):
        # EOS 토큰 제외
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
# Client 생성
# ---------------------------------------------------------------------------

def create_client(provider="openai"):
    """provider에 따라 OpenAI 호환 클라이언트를 반환한다.

    Args:
        provider: "openai" | "litellm" | "vllm"

    Returns:
        client: client.chat.completions.create() 호출 가능한 객체
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
    """vLLM 서버용 래퍼. thinking 모델에 자동으로 enable_thinking=false 추가."""

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
                # 기존 extra_body 유지하면서 thinking 비활성화 추가
                extra_body = kwargs.pop("extra_body", {}) or {}
                if "chat_template_kwargs" not in extra_body:
                    # continue_final_message가 설정되지 않은 경우에만 thinking 비활성화
                    if "continue_final_message" not in extra_body:
                        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                kwargs["extra_body"] = extra_body
                return self._client.chat.completions.create(**kwargs)


class _LiteLLMClient:
    """litellm.completion()을 OpenAI client 인터페이스로 감싸는 래퍼.

    사용법이 기존 OpenAI() 클라이언트와 동일하다:
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

                # logprobs 포맷 정규화
                if resp.choices:
                    for choice in resp.choices:
                        _normalize_logprobs(choice)

                return resp


# ---------------------------------------------------------------------------
# 모델 태그 (파일시스템 안전)
# ---------------------------------------------------------------------------

def make_model_tag(model: str) -> str:
    """모델 이름에서 파일시스템에 안전한 태그를 생성한다.

    예:
        gpt-4.1-nano         -> gpt-41-nano
        openrouter/qwen/qwen3-235b-a22b -> qwen3-235b-a22b
        together_ai/Qwen/Qwen3-32B      -> Qwen3-32B
    """
    name = model.split("/")[-1]
    name = name.replace(".", "")
    return name


# ---------------------------------------------------------------------------
# 토크나이저 (teacher forcing용)
# ---------------------------------------------------------------------------

def get_tokenizer(model: str):
    """모델에 맞는 토크나이저를 반환한다. 미지원 모델은 cl100k_base 폴백."""
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except (KeyError, Exception):
        return tiktoken.get_encoding("cl100k_base")
