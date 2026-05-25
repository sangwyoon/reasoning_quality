"""Core logic for paraphrase experiments: reasoning splitting, rule-based/LLM paraphrase, step selection."""

import re
import random

from openai import OpenAI


# ---------------------------------------------------------------------------
# 0. Remove final answer from reasoning
# ---------------------------------------------------------------------------

# Pattern for final answer (matched only in last 40%)
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
    """Remove final answer section from end of reasoning text.

    Removes explicit answers like "**Answer: C**", "**Final answer: D. text**"
    from the end of model-generated reasoning to prevent answer leakage.

    Returns:
        (stripped_reasoning, removed_part)
    """
    text = reasoning.rstrip()
    if not text:
        return text, ""

    threshold = len(text) * 0.4  # Match only in last 60%

    last_pos = -1
    for pat in _ANSWER_INDICATOR_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if m.start() >= threshold and m.start() > last_pos:
                last_pos = m.start()

    if last_pos > 0:
        stripped = text[:last_pos].rstrip()
        # Remove trailing --- separator
        stripped = re.sub(r'\n-{3,}\s*$', '', stripped).rstrip()
        removed = text[last_pos:]
        return stripped, removed

    return text, ""


# ---------------------------------------------------------------------------
# 1. Split reasoning into steps
# ---------------------------------------------------------------------------

def split_reasoning_into_steps(reasoning: str):
    """Split reasoning text into logical steps.

    Priority:
      1. **Step N** headers (e.g., **Step 1:**, ### Step 1:)
      2. Numbered lists (1., 2., 3. ...)
      3. Option/Statement analysis (### Statement 1, **Statement 1:** etc.)
      4. Paragraph breaks

    Returns:
        (pattern_name, [step1, step2, ...])
    """
    text = reasoning.strip()

    # Pattern 1: **Step N** headers
    step_header_pattern = r'(?=(?:^|\n)(?:#{1,4}\s*)?(?:\*\*)?Step\s+\d+)'
    parts = re.split(step_header_pattern, text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        return ("step_header", parts)

    # Pattern 2: Numbered lists
    numbered_pattern = r'(?=(?:^|\n)\d+\.\s)'
    parts = re.split(numbered_pattern, text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        return ("numbered", parts)

    # Pattern 3: Statement/option analysis
    statement_pattern = r'(?=(?:^|\n)(?:#{1,4}\s*)?(?:\*\*)?(?:Statement|Option)\s+\d+)'
    parts = re.split(statement_pattern, text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 3:
        return ("statement", parts)

    # Pattern 4: Paragraph breaks (\n\n)
    parts = re.split(r'\n{2,}', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        return ("paragraph", parts)

    # Cannot split -> treat entire text as single step
    return ("single", [text])


# ---------------------------------------------------------------------------
# 2. Rule-Based Paraphraser
# ---------------------------------------------------------------------------

# LaTeX/math placeholder protection
_LATEX_PATTERNS = [
    (r'\$\$[\s\S]*?\$\$', 'DISPLAY_MATH'),     # $$...$$
    (r'\\\[[\s\S]*?\\\]', 'BRACKET_MATH'),      # \[...\]
    (r'\\\([\s\S]*?\\\)', 'PAREN_MATH'),        # \(...\)
    (r'\$[^\$\n]+?\$', 'INLINE_MATH'),          # $...$
]

# Connector substitution table (unidirectional: A->B)
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

# Verb phrase substitution table (unidirectional)
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

# Sentence start transformations (unidirectional)
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
    """Rule-based text transformer. Substitutes text while protecting LaTeX/math."""

    def paraphrase(self, text: str):
        """Paraphrase text using rule-based substitutions.

        Returns:
            (paraphrased_text, [applied_rule_names])
        """
        # 1) Protect LaTeX/math: replace with placeholders
        protected = []
        working = text
        for pattern, tag in _LATEX_PATTERNS:
            def _replace(m, _tag=tag):
                idx = len(protected)
                placeholder = f"__PROTECTED_{_tag}_{idx}__"
                protected.append((placeholder, m.group()))
                return placeholder
            working = re.sub(pattern, _replace, working)

        # 2) Apply rules (prevent chaining)
        applied = []
        inserted_words = set()
        for rule_name, pattern, replacement in ALL_RULES:
            # Skip if previous rule inserted word matches current pattern
            rep_words = set(replacement.lower().split())
            if rep_words & inserted_words:
                continue
            if re.search(pattern, working):
                new_working = re.sub(pattern, replacement, working, count=1)
                if new_working != working:
                    applied.append(f"{rule_name}:{pattern}->{replacement}")
                    working = new_working
                    inserted_words.update(rep_words)

        # 3) Restore placeholders
        for placeholder, original in reversed(protected):
            working = working.replace(placeholder, original)

        return working, applied


class LLMParaphraser:
    """LLM-based text transformer. Changes expression while preserving meaning."""

    SYSTEM_PROMPT = (
        "Paraphrase the following statement. Do not change the semantic meaning. "
        "Keep all LaTeX, math notation, markdown formatting, and technical terms exactly as they are."
        "**STOP** before explicitly selecting or labeling any option as the final answer. Do NOT write phrases like 'therefore the answer is', 'the correct option is', 'this matches option X', or any equivalent conclusion that names a letter as the answer. End your response as if handing off your reasoning to someone else to make the final selection."
    )

    def __init__(self, client: OpenAI = None, model: str = "gpt-4.1-mini"):
        self.client = client or OpenAI()
        self.model = model

    def paraphrase(self, text: str):
        """Paraphrase text using LLM.

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
            # Return original text on failure
            return text, [f"llm_error:{str(e)[:100]}"]


# ---------------------------------------------------------------------------
# 3. Token Masking (replace k% of tokens with [UNK])
# ---------------------------------------------------------------------------

class TokenMasker:
    """Random mask k% of tokens with [UNK] across all steps.

    If protect_latex=True (default), LaTeX/math is protected.
    If protect_latex=False, LaTeX is also masked.
    """

    UNK = "[UNK]"

    def __init__(self, mask_ratio: float = 0.1, rng: random.Random = None,
                 protect_latex: bool = True):
        """
        Args:
            mask_ratio: fraction of tokens to mask (0.0-1.0). e.g., 0.1 = 10%
            rng: Random instance for reproducibility
            protect_latex: True to protect LaTeX/math, False to mask everything
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be between 0.0 and 1.0, got {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.rng = rng or random.Random()
        self.protect_latex = protect_latex

    def mask(self, text: str):
        """Replace k% of tokens in text with [UNK].

        Returns:
            (masked_text, num_masked, num_total)
        """
        if not text.strip() or self.mask_ratio == 0.0:
            return text, 0, 0

        # 1) Protect LaTeX/math (only when protect_latex=True)
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

        # 2) Tokenize (whitespace-based, preserve line breaks)
        parts = re.split(r'(\s+)', working)
        # parts = [token, whitespace, token, whitespace, ...]
        token_indices = [i for i, p in enumerate(parts) if p.strip()]

        # Exclude placeholder tokens and choice labels (A, B, C, D) from masking
        _PROTECTED_LABELS = {"A", "B", "C", "D", "A.", "B.", "C.", "D.",
                             "A:", "B:", "C:", "D:", "A)", "B)", "C)", "D)"}
        maskable = [i for i in token_indices
                    if not parts[i].startswith("__PROTECTED_")
                    and parts[i] not in _PROTECTED_LABELS]

        num_total = len(maskable)
        num_to_mask = max(1, round(num_total * self.mask_ratio)) if num_total > 0 else 0
        num_to_mask = min(num_to_mask, num_total)

        if num_to_mask == 0:
            # Restore and return
            for placeholder, original in reversed(protected):
                working = working.replace(placeholder, original)
            return working, 0, num_total

        # 3) Random masking
        mask_targets = set(self.rng.sample(maskable, num_to_mask))
        for i in mask_targets:
            parts[i] = self.UNK

        masked_text = "".join(parts)

        # 4) Restore placeholders (only when protect_latex=True)
        for placeholder, original in reversed(protected):
            masked_text = masked_text.replace(placeholder, original)

        return masked_text, num_to_mask, num_total


# ---------------------------------------------------------------------------
# 4. Select steps to paraphrase
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
    """Select step indices to paraphrase.

    Exclude intro and conclusion steps, randomly select N from body steps.

    Returns:
        Sorted list of selected step indices
    """
    if rng is None:
        rng = random.Random()

    body_indices = []
    for i, step in enumerate(steps):
        # Exclude first step if intro
        if i == 0 and _is_intro(step):
            continue
        # Exclude last step if conclusion
        if i == len(steps) - 1 and _is_conclusion(step):
            continue
        body_indices.append(i)

    # If no body steps, select from all
    if not body_indices:
        body_indices = list(range(len(steps)))

    num_select = min(num_paraphrase, len(body_indices))
    selected = sorted(rng.sample(body_indices, num_select))
    return selected
