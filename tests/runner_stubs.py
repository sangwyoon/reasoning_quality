"""Import paths and OpenAI-style fakes shared by the runner tests (no network)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
for sub in ("additional_experiments/benchmarks", "additional_experiments/semantic_and_structural"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))


def response(content, tokens=(("x", 0.9),), first=None):
    top = [SimpleNamespace(token=t, logprob=math.log(p)) for t, p in tokens]
    item = SimpleNamespace(token=first or top[0].token, logprob=top[0].logprob, top_logprobs=top)
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), logprobs=SimpleNamespace(content=[item]))])


def client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                           completions=SimpleNamespace(create=create))
