"""Prompts. The system prompt is the hallucination guard, so it is kept explicit.

Rules are those of ТЗ slide 10:
  1. answer only from the supplied context,
  2. cite the concrete article behind every statement,
  3. answer in the language of the question,
  4. no context -> abstain and escalate to a lawyer.
"""

from __future__ import annotations

import re

LANGUAGE_NAMES = {
    "ru": "Russian (русский)",
    "zh": "Chinese (中文)",
    "en": "English",
}

# Two very different situations must not share a message. "No provision was found" is a
# statement about Russian law; saying it when retrieval actually succeeded and only the
# generator was unreachable tells the client something false about their legal position,
# and puts a misleading "nothing found" row in the lawyer's queue.
GENERATION_UNAVAILABLE_MESSAGES = {
    "ru": (
        "Сервис формирования ответа временно недоступен. Найденные нормы сохранены, "
        "вопрос передан юристу. Это техническая неполадка, а не отсутствие регулирования."
    ),
    "zh": (
        "答复生成服务暂时不可用。检索到的法律条款已保存，问题已转交律师处理。"
        "这是技术故障，并不表示法律没有相关规定。"
    ),
    "en": (
        "The answer service is temporarily unavailable. The provisions found for your "
        "question have been saved and it has been referred to a lawyer. This is a technical "
        "fault, not an absence of regulation."
    ),
}

ABSTENTION_MESSAGES = {
    "ru": (
        "В доступных материалах законодательства РФ не нашлось нормы, прямо отвечающей "
        "на этот вопрос. Чтобы не давать ответ наугад, вопрос передан юристу."
    ),
    "zh": (
        "在现有的俄罗斯联邦法律资料中，未能找到直接回答该问题的条款。"
        "为避免给出不确切的答复，该问题已转交律师处理。"
    ),
    "en": (
        "The available Russian legislation does not contain a provision that directly "
        "answers this question. Rather than guess, the question has been escalated to a lawyer."
    ),
}

SYSTEM_PROMPT = """You are a legal assistant for Russian Federation law, serving foreign \
entrepreneurs entering the Russian market.

Absolute rules — violating any of them makes the answer unusable:

1. GROUNDING. Answer **only** from the numbered context fragments provided below. Never \
use knowledge from outside them. If they do not contain the answer, say so — do not \
reason from general legal knowledge, do not extrapolate from an adjacent article.

2. CITATIONS. Every legal statement must name the article it comes from, written inline as \
«Статья N <act title>» together with the fragment number in square brackets, e.g. \
«Статья 14 Федерального закона "Об обществах с ограниченной ответственностью" [2]». \
An uncited statement is a defect.

3. LANGUAGE. Write the entire answer in {language}. Keep the Russian legal terms and act \
names in Russian in parentheses on first use, because only the Russian text is \
authoritative.

4. ABSTENTION. If the fragments are insufficient, incomplete, or only tangentially \
related, do not attempt an answer. State plainly that the available legislation does not \
cover the question and that it has been referred to a lawyer.

5. EDITIONS. Each fragment states its redaction (редакция). If a fragment's redaction \
matters to the answer, mention it.

Be concise and practical. Do not add disclaimers beyond what these rules require. Do not \
invent article numbers under any circumstance."""

USER_PROMPT = """Context fragments from Russian legislation:

{context}

---
Question: {question}

Answer using only the fragments above, following every rule."""

CONDENSE_PROMPT = """Rewrite the follow-up question into a single standalone question \
that can be understood without the conversation history. Resolve pronouns and implicit \
references using the history. Keep the original language. Output only the rewritten \
question, nothing else.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


# A question is treated as Russian or Chinese as soon as that script reaches this share of
# its letters. The scripts are not weighted equally on purpose: Latin characters turn up
# constantly inside Russian and Chinese questions (brand names, "LLC", "OOO"), while
# Cyrillic and CJK almost never turn up inside an English one. So a minority of Cyrillic is
# strong evidence, whereas a majority of Latin is weak evidence.
_SCRIPT_SHARE_THRESHOLD = 0.2


def detect_language(text: str) -> str:
    """Script-based language detection: Cyrillic -> ru, CJK -> zh, otherwise en.

    Deliberately a heuristic and not a model. It only has to pick between the three
    languages the product serves, and it must never fail or add latency. The client can
    override it with an explicit ``language`` on the request.
    """
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", text))
    cjk = len(re.findall(r"[一-鿿㐀-䶿]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = cyrillic + cjk + latin
    if not total:
        return "ru"
    if cjk / total >= _SCRIPT_SHARE_THRESHOLD:
        return "zh"
    if cyrillic / total >= _SCRIPT_SHARE_THRESHOLD:
        return "ru"
    return "en"


def build_context(fragments: list[dict]) -> str:
    """Render retrieved chunks as numbered blocks the model can cite by index."""
    blocks = []
    for index, fragment in enumerate(fragments, start=1):
        blocks.append(f"[{index}]\n{fragment['breadcrumbs']}\n\n{fragment['text']}")
    return "\n\n---\n\n".join(blocks)
