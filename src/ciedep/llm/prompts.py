"""Interpretation stage 프롬프트 (논문 III-B-2, Fig. 2).

프롬프트는 세 요소로 구성된다.
  - System rule       : 임상심리 전문가 / 참가자 본인의 역할을 명시
  - User rule & output: 출력 형식을 엄격히 제한
  - Neutrality        : 우울 증상이 있을 수도, 없을 수도 있음을 명시해 낙인 편향을 억제

CoT 프롬프트는 네 단계의 사고 흐름을 지시한다.
  1) 전체 전사를 읽는다
  2) 정서적/심리적으로 두드러진 세 순간을 고른다
  3) 각 순간에 대해 (a) 인용 (b) 정서적 근거 (c) 해석을 정리한다
  4) 이를 종합해 우울증 점수 또는 1인칭 내면 요약을 생성한다

Table II 비교를 위해 standard(non-CoT) / CoT / self-validation 변형도 제공한다.
"""

from __future__ import annotations

from typing import Dict, List

Message = Dict[str, str]

# ---------------------------------------------------------------------------
# System rule
# ---------------------------------------------------------------------------
SYSTEM_SCORE = (
    "You are a clinical psychologist and cognitive language analyst.\n"
    "Your task is to evaluate a single participant based on the interview transcript below.\n"
    "The participant may or may not exhibit signs of depression.\n"
    "Analyze the dialogue carefully and think step-by-step according to the given structure.\n"
    "Silently reason through your analysis but output only the required structured answers."
)

SYSTEM_SUMMARY = (
    "You are the inner consciousness of a participant, reflecting on your own words.\n"
    "You may or may not exhibit signs of depression.\n"
    "Your task is to silently analyze a transcript and then articulate your true inner state "
    "in a concise summary.\n"
    "You will not show your analysis steps."
)

# 네 단계 사고 흐름 (Fig. 2)
_COT_STEPS = (
    "Before writing your final answer, you must first perform these steps internally "
    "without writing them out.\n"
    "1. Read the entire transcript carefully.\n"
    "2. Identify three significant moments that reveal the participant's true feelings or core thoughts.\n"
    "3. For each moment, mentally note the (a) quoted content, (b) the emotional evidence, and "
    "(c) an explanation of its significance, considering a full range of emotions "
    "(depression, contentment, anxiety, etc.).\n"
)

SCORE_FORMAT = ">>> Depression score: 0.XX"
SUMMARY_FORMAT = ">>> Inner Monologue: [Your 3-4 sentence first-person summary here]"


# ---------------------------------------------------------------------------
# Depression score
# ---------------------------------------------------------------------------
def standard_score_prompt(transcript: str) -> List[Message]:
    """Non-CoT (Table II 의 'Standard')."""
    user = (
        "Read the transcript and output the participant's level of depression.\n"
        "The range is 0 (Non-depression) to 1 (Depression).\n"
        f"{transcript}\n\n"
        "Output format:\n"
        f"{SCORE_FORMAT}\n\n"
        "Do not include anything else. Do not rephrase or summarize. Use only the required structure."
    )
    return [{"role": "user", "content": user}]


def cot_score_prompt(transcript: str) -> List[Message]:
    """CIEDep-Net 의 CoT 우울증 점수 프롬프트."""
    user = (
        "Your mission is to analyze a single participant based on the following interview transcript.\n"
        f"{_COT_STEPS}"
        "4. Based on the three moments, estimate a depression score between 0.0 (not depressed) "
        "and 1.0 (severely depressed).\n"
        "**IMPORTANT: You must use the exact format below. No other words or variations are allowed.**\n"
        "Do not include anything else. Do not rephrase or summarize. Use only the required structure.\n"
        ">>> Depression score: <floating_point_number_between_0.00_and_1.00>\n\n"
        f"{transcript}\n\n"
        "Output format:\n"
        f"{SCORE_FORMAT}\n\n"
        "Do not include anything else. Do not rephrase or summarize. Use only the required structure."
    )
    return [{"role": "system", "content": SYSTEM_SCORE}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Inner summary
# ---------------------------------------------------------------------------
def standard_summary_prompt(transcript: str) -> List[Message]:
    """Non-CoT 내면 요약."""
    user = (
        "Below is an interview transcript.\n"
        "Using the full conversation, write a 3-4 sentence first-person inner monologue that "
        "reflects the participant's psychological and emotional state.\n"
        f"{transcript}\n\n"
        "Output only the final inner monologue."
    )
    return [{"role": "user", "content": user}]


def cot_summary_prompt(transcript: str) -> List[Message]:
    """CIEDep-Net 의 CoT 내면 요약 프롬프트."""
    user = (
        "Your mission is to produce only a final 'Inner monologue'.\n"
        "<internal_thought_process>\n"
        f"{_COT_STEPS}"
        "4. Based on these three internal reflections, synthesize them into a coherent, "
        "first-person summary of your overall inner state.\n"
        "</internal_thought_process>\n"
        "<transcript>\n"
        f"{transcript}\n"
        "</transcript>\n\n"
        "<final_output_instructions>\n"
        "Now, having completed the internal reflection, provide ONLY the final inner monologue.\n"
        "The output must be nothing else but the summary, using the following exact format. "
        "Do not include any 'Step' analysis. Do not add any words before or after the specified format.\n\n"
        f"{SUMMARY_FORMAT}\n"
        "</final_output_instructions>"
    )
    return [{"role": "system", "content": SYSTEM_SUMMARY}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Self-validation (Table II 비교군)
# ---------------------------------------------------------------------------
def self_validation_prompt(transcript: str, draft: str, target: str = "score") -> List[Message]:
    """생성된 출력을 스스로 검증/수정하게 하는 후속 프롬프트."""
    if target == "score":
        system, fmt, what = SYSTEM_SCORE, SCORE_FORMAT, "depression score"
    else:
        system, fmt, what = SYSTEM_SUMMARY, SUMMARY_FORMAT, "inner monologue"

    user = (
        f"You previously produced the following {what} for the transcript below.\n\n"
        f"<previous_output>\n{draft}\n</previous_output>\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        f"Re-examine the transcript and verify whether the {what} is well supported by the evidence. "
        "If it is accurate, repeat it unchanged. If it is not, correct it.\n"
        "Output only the final answer in the exact format below, with nothing else.\n"
        f"{fmt}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
PROMPT_BUILDERS = {
    ("standard", "score"): standard_score_prompt,
    ("standard", "summary"): standard_summary_prompt,
    ("cot", "score"): cot_score_prompt,
    ("cot", "summary"): cot_summary_prompt,
    ("cot_self_consistency", "score"): cot_score_prompt,
    ("cot_self_consistency", "summary"): cot_summary_prompt,
    ("cot_self_validation", "score"): cot_score_prompt,
    ("cot_self_validation", "summary"): cot_summary_prompt,
}


def build_prompt(transcript: str, strategy: str = "cot_self_consistency", target: str = "score") -> List[Message]:
    """전략/목표에 맞는 chat message 리스트를 만든다."""
    try:
        builder = PROMPT_BUILDERS[(strategy, target)]
    except KeyError as exc:
        raise ValueError(f"알 수 없는 조합입니다: strategy={strategy}, target={target}") from exc
    return builder(transcript)
