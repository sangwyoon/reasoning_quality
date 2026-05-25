"""Paraphrase 실험 핵심 로직: reasoning 분리, rule-based/LLM paraphrase, step 선택."""

import re
import random

from openai import OpenAI


# ---------------------------------------------------------------------------
# 0. Reasoning 끝 답 제거
# ---------------------------------------------------------------------------

# 최종 답을 나타내는 패턴 (마지막 40% 내에서만 매칭)
_ANSWER_INDICATOR_PATTERNS = [
    # --- separator followed by answer block
    r'\n-{3,}\s*\n+\s*(?:#{1,4}\s*)?\*{0,2}(?:Final\s+)?(?:Answer|Choice)\s*:',
    # ### Final answer: / ## Final Answer: (markdown header)
    r'\n+\s*#{1,4}\s*(?:Final\s+)?[Aa]nswer\s*:',
    # **Final answer: X** or **Final answer:**
    r'\n+\s*\*{0,2}Final\s+(?:answer|choice)\s*:?\s*\*{0,2}',
    # **Answer: X** or **Answer:**
    r'\n+\s*\*{0,2}Answer\s*:\s*\*{0,2}',
    # **Correct answer/choice: X**
    r'\n+\s*\*{0,2}Correct\s+(?:answer|choice|option)\s*:\s*\*{0,2}',
    # Therefore, the correct answer/choice is
    r'\n+\s*\*{0,2}Therefore,?\s+the\s+(?:correct\s+|best\s+|most\s+appropriate\s+)?(?:answer|choice|option)\s+is\b',
    # The best/correct answer is
    r'\n+\s*\*{0,2}[Tt]he\s+(?:correct|best|most\s+appropriate)\s+(?:answer|choice|option)\s+is\b',
]


def strip_final_answer(reasoning: str) -> tuple:
    """Reasoning 텍스트 끝에서 최종 답 섹션을 제거한다.

    모델이 생성한 reasoning 끝에 "**Answer: C**", "**Final answer: D. text**" 등의
    명시적 답이 있으면, 이를 제거하여 paraphrasing 실험에서 답 누출을 방지한다.

    Returns:
        (stripped_reasoning, removed_part)
    """
    text = reasoning.rstrip()
    if not text:
        return text, ""

    threshold = len(text) * 0.4  # 마지막 60% 내에서만 매칭

    last_pos = -1
    for pat in _ANSWER_INDICATOR_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if m.start() >= threshold and m.start() > last_pos:
                last_pos = m.start()

    if last_pos > 0:
        stripped = text[:last_pos].rstrip()
        # trailing --- separator 제거
        stripped = re.sub(r'\n-{3,}\s*$', '', stripped).rstrip()
        removed = text[last_pos:]
        return stripped, removed

    return text, ""


# ---------------------------------------------------------------------------
# 1. Reasoning → Step 분리
# ---------------------------------------------------------------------------

def split_reasoning_into_steps(reasoning: str):
    """reasoning 텍스트를 논리적 step 단위로 분리한다.

    우선순위:
      1. **Step N** 헤더 (예: **Step 1:**, ### Step 1:)
      2. 번호 매기기 (1., 2., 3. ...)
      3. 옵션/Statement 분석 (### Statement 1, **Statement 1:** 등)
      4. 단락 분리 (\\n\\n)

    Returns:
        (pattern_name, [step1, step2, ...])
    """
    text = reasoning.strip()

    # Pattern 1: **Step N** 헤더 (### Step N 또는 **Step N**)
    step_header_pattern = r'(?=(?:^|\n)(?:#{1,4}\s*)?(?:\*\*)?Step\s+\d+)'
    parts = re.split(step_header_pattern, text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        return ("step_header", parts)

    # Pattern 2: 번호 매기기 (줄 시작 1. 2. 3.)
    numbered_pattern = r'(?=(?:^|\n)\d+\.\s)'
    parts = re.split(numbered_pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        return ("numbered", parts)

    # Pattern 3: Statement/옵션 분석 (### Statement N, **Statement N**)
    statement_pattern = r'(?=(?:^|\n)(?:#{1,4}\s*)?(?:\*\*)?(?:Statement|Option)\s+\d+)'
    parts = re.split(statement_pattern, text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        return ("statement", parts)

    # Pattern 4: 단락 분리 (\n\n)
    parts = re.split(r'\n{2,}', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return ("paragraph", parts)

    # 분리 불가 → 전체를 하나의 step으로
    return ("single", [text])


# ---------------------------------------------------------------------------
# 2. Rule-Based Paraphraser
# ---------------------------------------------------------------------------

# LaTeX/수식 플레이스홀더 보호
_LATEX_PATTERNS = [
    (r'\$\$[\s\S]*?\$\$', 'DISPLAY_MATH'),     # $$...$$
    (r'\\\[[\s\S]*?\\\]', 'BRACKET_MATH'),      # \[...\]
    (r'\\\([\s\S]*?\\\)', 'PAREN_MATH'),        # \(...\)
    (r'\$[^\$\n]+?\$', 'INLINE_MATH'),          # $...$
]

# 연결어 치환 테이블 (단방향만: A→B)
_CONNECTIVE_RULES = [
    (r'\bTherefore\b', 'Thus'),
    (r'\bHowever\b', 'Nevertheless'),
    (r'\bSince\b', 'Because'),
    (r'\bMoreover\b', 'Furthermore'),
    (r'\bFurthermore\b', 'Additionally'),
    (r'\bIn other words\b', 'That is'),
    (r'\bSpecifically\b', 'In particular'),
    (r'\bFor example\b', 'For instance'),
    (r'\bSo,', 'Therefore,'),
    (r'\bConsequently\b', 'As a result'),
    (r'\bHence\b', 'Thus'),
    (r'\bNevertheless\b', 'Nonetheless'),
    (r'\bAdditionally\b', 'Also'),
]

# 동사구 치환 테이블 (단방향만)
_VERB_RULES = [
    (r'\bwe calculate\b', 'we compute'),
    (r'\bwe find\b', 'we obtain'),
    (r'\bwe need to\b', 'we must'),
    (r'\bwe can see\b', 'we observe'),
    (r'\bwe know\b', 'we recognize'),
    (r'\bwe get\b', 'we arrive at'),
    (r'\bwe determine\b', 'we establish'),
    (r'\bwe note\b', 'we observe'),
    (r'\bwe check\b', 'we verify'),
    (r'\bLet\'s analyze\b', 'Let us examine'),
    (r'\bThis means\b', 'This implies'),
    (r'\bThis gives\b', 'This yields'),
    (r'\bis equal to\b', 'equals'),
    (r'\bis given by\b', 'is expressed as'),
    (r'\bwe compute\b', 'we evaluate'),
    (r'\bwe obtain\b', 'we derive'),
    (r'\bwe must\b', 'we have to'),
    (r'\bwe observe\b', 'we notice'),
]

# 문장 시작 변환 (단방향만)
_SENTENCE_START_RULES = [
    (r'\bTo determine\b', 'In order to determine'),
    (r'\bTo find\b', 'In order to find'),
    (r'\bTo solve\b', 'In order to solve'),
    (r'\bRecall that\b', 'Note that'),
    (r'\bIt follows that\b', 'From this it follows that'),
    (r'\bFirst,\b', 'To begin,'),
    (r'\bNext,\b', 'Then,'),
    (r'\bFinally,\b', 'Lastly,'),
    (r'\bNote that\b', 'Observe that'),
]

ALL_RULES = (
    [("connective", p, r) for p, r in _CONNECTIVE_RULES]
    + [("verb", p, r) for p, r in _VERB_RULES]
    + [("sentence_start", p, r) for p, r in _SENTENCE_START_RULES]
)


class RuleBasedParaphraser:
    """Rule-based 표현 변환기. LaTeX/수식을 보호하면서 텍스트만 치환한다."""

    def paraphrase(self, text: str):
        """텍스트를 rule-based로 paraphrase한다.

        Returns:
            (paraphrased_text, [applied_rule_names])
        """
        # 1) LaTeX/수식 보호: 플레이스홀더로 치환
        protected = []
        working = text
        for pattern, tag in _LATEX_PATTERNS:
            def _replace(m, _tag=tag):
                idx = len(protected)
                placeholder = f"__PROTECTED_{_tag}_{idx}__"
                protected.append((placeholder, m.group()))
                return placeholder
            working = re.sub(pattern, _replace, working)

        # 2) Rule 적용 (chaining 방지: replacement로 삽입된 텍스트에 대한 재치환 skip)
        applied = []
        inserted_words = set()
        for rule_name, pattern, replacement in ALL_RULES:
            # 이전 rule에서 삽입한 단어가 현재 pattern에 포함되면 skip
            rep_words = set(replacement.lower().split())
            if rep_words & inserted_words:
                continue
            if re.search(pattern, working):
                new_working = re.sub(pattern, replacement, working, count=1)
                if new_working != working:
                    applied.append(f"{rule_name}:{pattern}->{replacement}")
                    working = new_working
                    inserted_words.update(rep_words)

        # 3) 플레이스홀더 복원
        for placeholder, original in reversed(protected):
            working = working.replace(placeholder, original)

        return working, applied


class LLMParaphraser:
    """LLM 기반 표현 변환기. 의미를 유지하면서 표현만 변환한다."""

    SYSTEM_PROMPT = (
        "Paraphrase the following statement. Do not change the semantic meaning. "
        "Keep all LaTeX, math notation, markdown formatting, and technical terms exactly as they are."
        "**STOP** before explicitly selecting or labeling any option as the final answer. Do NOT write phrases like 'therefore the answer is', 'the correct option is', 'this matches option X', or any equivalent conclusion that names a letter as the answer. End your response as if handing off your reasoning to someone else to make the final selection."
    )

    def __init__(self, client: OpenAI = None, model: str = "gpt-4.1-mini"):
        self.client = client or OpenAI()
        self.model = model

    def paraphrase(self, text: str):
        """텍스트를 LLM으로 paraphrase한다.

        Returns:
            (paraphrased_text, ["llm_paraphrase"])
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            paraphrased = resp.choices[0].message.content.strip()
            return paraphrased, ["llm_paraphrase"]
        except Exception as e:
            # 실패 시 원문 그대로 반환
            return text, [f"llm_error:{str(e)[:100]}"]


# ---------------------------------------------------------------------------
# 3. Token Masking (모든 step에서 k%의 토큰을 [UNK]로 대체)
# ---------------------------------------------------------------------------

class TokenMasker:
    """모든 step에서 k%의 토큰을 [UNK]로 random masking한다.

    protect_latex=True(기본값)이면 LaTeX/수식을 보호하고,
    protect_latex=False이면 LaTeX도 함께 masking한다.
    """

    UNK = "[UNK]"

    def __init__(self, mask_ratio: float = 0.1, rng: random.Random = None,
                 protect_latex: bool = True):
        """
        Args:
            mask_ratio: masking할 토큰 비율 (0.0 ~ 1.0). 예: 0.1 → 10%
            rng: 재현성을 위한 Random 인스턴스
            protect_latex: True면 LaTeX/수식을 보호, False면 LaTeX도 masking
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.rng = rng or random.Random()
        self.protect_latex = protect_latex

    def mask(self, text: str):
        """텍스트에서 k%의 토큰을 [UNK]로 대체한다.

        Returns:
            (masked_text, num_masked, num_total)
        """
        if not text.strip() or self.mask_ratio == 0.0:
            return text, 0, 0

        # 1) LaTeX/수식 보호 (protect_latex=True일 때만)
        protected = []
        working = text
        if self.protect_latex:
            for pattern, tag in _LATEX_PATTERNS:
                def _replace(m, _tag=tag):
                    idx = len(protected)
                    placeholder = f"__PROTECTED_{_tag}_{idx}__"
                    protected.append((placeholder, m.group()))
                    return placeholder
                working = re.sub(pattern, _replace, working)

        # 2) 토큰 분리 (공백 기준, 줄바꿈/공백 구조 보존)
        parts = re.split(r'(\s+)', working)
        # parts = [token, whitespace, token, whitespace, ...]
        token_indices = [i for i, p in enumerate(parts) if p.strip()]

        # 플레이스홀더 토큰 및 선택지 라벨(A, B, C, D)은 masking 대상에서 제외
        _PROTECTED_LABELS = {"A", "B", "C", "D", "A.", "B.", "C.", "D.",
                             "A:", "B:", "C:", "D:", "A)", "B)", "C)", "D)"}
        maskable = [i for i in token_indices
                    if not parts[i].startswith("__PROTECTED_")
                    and parts[i] not in _PROTECTED_LABELS]

        num_total = len(maskable)
        num_to_mask = max(1, round(num_total * self.mask_ratio)) if num_total > 0 else 0
        num_to_mask = min(num_to_mask, num_total)

        if num_to_mask == 0:
            # 복원 후 반환
            for placeholder, original in reversed(protected):
                working = working.replace(placeholder, original)
            return working, 0, num_total

        # 3) 랜덤 masking
        mask_targets = set(self.rng.sample(maskable, num_to_mask))
        for i in mask_targets:
            parts[i] = self.UNK

        masked_text = "".join(parts)

        # 4) 플레이스홀더 복원 (protect_latex=True일 때만)
        for placeholder, original in reversed(protected):
            masked_text = masked_text.replace(placeholder, original)

        return masked_text, num_to_mask, num_total


# ---------------------------------------------------------------------------
# 4. Paraphrase할 step 선택 (기존 호환용)
# ---------------------------------------------------------------------------

_INTRO_KEYWORDS = [
    "let's analyze", "let us analyze", "let's think", "let us think",
    "let's examine", "let us examine", "we are asked", "we are given",
    "let's consider", "the question asks", "we want to",
]
_CONCLUSION_KEYWORDS = [
    "final answer", "correct answer", "correct choice", "answer:",
    "answer is", "therefore, the answer",
]


def _is_intro(step: str) -> bool:
    lower = step.lower()[:200]
    return any(kw in lower for kw in _INTRO_KEYWORDS)


def _is_conclusion(step: str) -> bool:
    lower = step.lower()
    return any(kw in lower for kw in _CONCLUSION_KEYWORDS) and len(step) < 300


def select_steps_to_paraphrase(steps: list, num_paraphrase: int, rng: random.Random = None):
    """Paraphrase할 step 인덱스를 선택한다.

    intro(서론)와 conclusion(결론) step은 제외하고 body step 중에서 N개를 랜덤 선택.

    Returns:
        선택된 step 인덱스 리스트 (sorted)
    """
    if rng is None:
        rng = random.Random()

    body_indices = []
    for i, step in enumerate(steps):
        # 첫 step이 intro인 경우 제외
        if i == 0 and _is_intro(step):
            continue
        # 마지막 step이 conclusion인 경우 제외
        if i == len(steps) - 1 and _is_conclusion(step):
            continue
        body_indices.append(i)

    # body가 없으면 전체에서 선택
    if not body_indices:
        body_indices = list(range(len(steps)))

    num_select = min(num_paraphrase, len(body_indices))
    selected = sorted(rng.sample(body_indices, num_select))
    return selected
