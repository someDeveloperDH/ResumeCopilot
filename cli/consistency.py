"""대화 일관성 검사 - jinyoung_v2 cli/main.py의 일관성 검사 블록을 별도 모듈로 추출.

규칙 기반 키워드 충돌 + NLI(자연어추론) 모순 탐지 + SBERT 의미 유사도를
이전 답변들과 pairwise로 비교한다. 모델이 설치되어 있지 않으면 해당 검사만
건너뛰고 규칙 기반 검사만으로 동작한다 (graceful degradation).
"""

from __future__ import annotations

import os

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


# 단순 휴리스틱 기반 일관성 검사 키워드
LEADER_WORDS = ["리더", "팀장", "주도", "총괄", "PM", "책임", "이끌"]
LOW_AUTHORITY_WORDS = ["결정권", "권한", "시키는", "보조", "따라", "참여만", "맡은 게 없", "기여가 적"]
SOLO_WORDS = ["혼자", "단독", "개인", "스스로"]
TEAM_WORDS = ["팀", "협업", "함께", "동료", "조원"]
VAGUE_WORDS = ["열심히", "노력", "잘", "많이", "최선을", "좋은 결과", "성공적"]

# SBERT는 맥락 유사도 검사에 사용한다. 없으면 해당 검사만 건너뛴다.
SBERT_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
SBERT_MODEL = None
CONTEXT_LOW_SIM_THRESHOLD = 0.35
CONTEXT_RELATED_THRESHOLD = 0.55

# NLI는 실제 contradiction 보조 판정에 사용한다.
# 한국어 포함 XNLI를 지원하는 multilingual 모델을 기본값으로 둔다.
# CPU가 느리면 환경변수 CONSISTENCY_USE_NLI=0 으로 끌 수 있다.
NLI_MODEL_NAME = os.environ.get("CONSISTENCY_NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
USE_NLI = os.environ.get("CONSISTENCY_USE_NLI", "1") != "0"
NLI_TOKENIZER = None
NLI_MODEL = None
NLI_DEVICE = "cpu"
NLI_CONTRADICTION_THRESHOLD = 0.65
NLI_STRONG_CONTRADICTION_THRESHOLD = 0.80


def _contains_any(text: str, words: list[str]) -> bool:
    return any(w.lower() in text.lower() for w in words)


def _get_previous_answers(session, latest_answer: str | None = None) -> list[dict]:
    """현재 답변을 제외한 이전 답변들을 비교 가능한 단위로 반환한다."""
    answers = []
    latest_text = latest_answer.strip() if latest_answer else ""

    # session.add_turn() 직후에는 현재 답변이 conversation 마지막에 들어가 있다.
    latest_is_tail = bool(
        session.conversation
        and latest_text
        and str(session.conversation[-1].get("response", "")).strip() == latest_text
    )

    initial_answer = session.answer.strip() if session.answer else ""
    # 최초 답변 검사 시에는 최초 답변이 곧 현재 답변이므로 비교 대상에서 제외한다.
    # 꼬리답변 검사 시에는 최초 답변이 중요한 이전 맥락이므로 포함한다.
    if initial_answer and (latest_is_tail or initial_answer != latest_text):
        answers.append({
            "source": "최초 답변",
            "text": initial_answer,
        })

    turns_to_compare = session.conversation[:-1] if latest_is_tail else session.conversation
    for idx, turn in enumerate(turns_to_compare, start=1):
        response = str(turn.get("response", "")).strip()
        if not response:
            continue

        answers.append({
            "source": f"꼬리답변 {idx}",
            "text": response,
        })

    return answers


def _get_sbert_model():
    """SBERT 모델을 한 번만 로딩한다. 설치되어 있지 않으면 None을 반환한다."""
    global SBERT_MODEL

    if SentenceTransformer is None or util is None:
        return None

    if SBERT_MODEL is None:
        try:
            SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")
        except Exception as e:
            print(f"  [일관성 검사] SBERT 로딩 실패 → 유사도 검사는 건너뜀: {e}")
            SBERT_MODEL = False

    return None if SBERT_MODEL is False else SBERT_MODEL


def _calc_similarity(text1: str, text2: str) -> float | None:
    """SBERT cosine similarity. SBERT 사용 불가 시 None."""
    model = _get_sbert_model()
    if model is None:
        return None

    emb = model.encode([text1, text2], convert_to_tensor=True)
    return round(float(util.cos_sim(emb[0], emb[1]).item()), 4)


def _get_nli_components():
    """NLI tokenizer/model을 한 번만 로딩한다. 사용 불가 시 None을 반환한다."""
    global NLI_TOKENIZER, NLI_MODEL

    if not USE_NLI:
        return None

    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None

    if NLI_TOKENIZER is None or NLI_MODEL is None:
        try:
            NLI_TOKENIZER = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
            NLI_MODEL = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
            NLI_MODEL.to(NLI_DEVICE)
            NLI_MODEL.eval()
        except Exception as e:
            print(f"  [일관성 검사] NLI 로딩 실패 → contradiction 검사는 건너뜀: {e}")
            NLI_TOKENIZER = False
            NLI_MODEL = False

    if NLI_TOKENIZER is False or NLI_MODEL is False:
        return None
    return NLI_TOKENIZER, NLI_MODEL


def _resolve_nli_label_map(model) -> dict:
    """모델 config의 id2label을 contradiction/entailment/neutral로 정규화한다."""
    id2label = getattr(model.config, "id2label", {}) or {}
    normalized = {}
    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if "contradiction" in label_lower or "contradict" in label_lower:
            normalized["contradiction"] = int(idx)
        elif "entailment" in label_lower or "entail" in label_lower:
            normalized["entailment"] = int(idx)
        elif "neutral" in label_lower:
            normalized["neutral"] = int(idx)

    # 일부 MNLI 계열 모델은 보통 0=contradiction, 1=neutral, 2=entailment를 쓴다.
    if "contradiction" not in normalized and getattr(model.config, "num_labels", 0) == 3:
        normalized["contradiction"] = 0
    if "neutral" not in normalized and getattr(model.config, "num_labels", 0) == 3:
        normalized["neutral"] = 1
    if "entailment" not in normalized and getattr(model.config, "num_labels", 0) == 3:
        normalized["entailment"] = 2

    return normalized


def _calc_nli_scores(premise: str, hypothesis: str) -> dict | None:
    """premise → hypothesis 관계의 NLI 확률을 계산한다. 사용 불가 시 None."""
    components = _get_nli_components()
    if components is None:
        return None

    tokenizer, model = components
    try:
        enc = tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(NLI_DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        label_map = _resolve_nli_label_map(model)
        return {
            "contradiction": round(float(probs[label_map.get("contradiction", 0)]), 4),
            "neutral": round(float(probs[label_map.get("neutral", 1)]), 4) if len(probs) > 1 else None,
            "entailment": round(float(probs[label_map.get("entailment", 2)]), 4) if len(probs) > 2 else None,
        }
    except Exception as e:
        print(f"  [일관성 검사] NLI 추론 실패 → 해당 pair는 건너뜀: {e}")
        return None


def _calc_bidirectional_nli(prev_text: str, latest_text: str) -> dict | None:
    """NLI는 방향성이 있으므로 이전→현재, 현재→이전을 모두 계산하고 contradiction 최댓값을 사용한다."""
    forward = _calc_nli_scores(prev_text, latest_text)
    backward = _calc_nli_scores(latest_text, prev_text)

    if forward is None and backward is None:
        return None

    forward_contra = forward.get("contradiction", 0.0) if forward else 0.0
    backward_contra = backward.get("contradiction", 0.0) if backward else 0.0
    max_contra = max(forward_contra, backward_contra)
    direction = "previous_to_current" if forward_contra >= backward_contra else "current_to_previous"

    return {
        "forward": forward,
        "backward": backward,
        "contradiction": round(float(max_contra), 4),
        "direction": direction,
    }


def _check_rule_conflicts(previous_text: str, latest_text: str, source: str = "이전 답변") -> tuple[list[str], list[str]]:
    """자소서 답변에서 자주 발생하는 역할/권한/협업 충돌을 검사한다."""
    issues = []
    suggestions = []

    if _contains_any(previous_text, LEADER_WORDS) and _contains_any(latest_text, LOW_AUTHORITY_WORDS):
        issues.append(f"{source}에서는 주도적 역할을 강조했지만, 현재 답변에서는 권한이나 기여도가 낮았다고 표현했습니다.")
        suggestions.append("실제로 맡은 역할, 결정권 범위, 본인 기여도를 구분해서 설명해 주세요.")

    if _contains_any(previous_text, LOW_AUTHORITY_WORDS) and _contains_any(latest_text, LEADER_WORDS):
        issues.append(f"{source}에서는 보조적 역할에 가까웠지만, 현재 답변에서는 주도적 역할로 표현이 바뀌었습니다.")
        suggestions.append("처음부터 주도한 것인지, 특정 단계에서만 주도했는지 구체적으로 정리해 주세요.")

    if _contains_any(previous_text, SOLO_WORDS) and _contains_any(latest_text, TEAM_WORDS):
        issues.append(f"{source}에서는 혼자 수행한 경험처럼 보였지만, 현재 답변에서는 팀 활동으로 표현되었습니다.")
        suggestions.append("개인 작업과 팀 협업의 범위를 나눠서 설명해 주세요.")

    if _contains_any(previous_text, TEAM_WORDS) and _contains_any(latest_text, SOLO_WORDS):
        issues.append(f"{source}에서는 팀/협업 경험으로 보였지만, 현재 답변에서는 혼자 수행한 것처럼 표현되었습니다.")
        suggestions.append("팀 전체 역할과 본인이 단독으로 맡은 부분을 구분해 주세요.")

    return issues, suggestions


def check_consistency_pairwise(session, latest_answer: str) -> dict:
    """
    이전 답변 각각과 현재 답변을 pairwise로 비교하고,
    전체 이전 답변을 합친 누적 맥락과도 비교한다.
    """
    latest_text = latest_answer.strip()
    previous_answers = _get_previous_answers(session, latest_answer=latest_text)

    issues = []
    suggestions = []
    pair_results = []

    if len(latest_text) < 30:
        issues.append("현재 답변이 너무 짧아 이전 답변과의 일관성을 충분히 판단하기 어렵습니다.")
        suggestions.append("본인의 역할, 행동, 결과를 최소 2~3문장으로 작성해 주세요.")

    vague_count = sum(1 for word in VAGUE_WORDS if word in latest_text)
    if vague_count >= 2 and len(latest_text) < 120:
        issues.append("현재 답변에 추상적인 표현이 많고 구체적인 행동이나 결과가 부족합니다.")
        suggestions.append("상황, 본인의 행동, 결과 수치나 변화가 드러나도록 보완해 주세요.")

    # 최초 답변만 있는 시점에는 비교 대상이 없으므로 현재 답변 자체의 품질만 검사한다.
    if not previous_answers:
        return {
            "status": "주의" if issues else "양호",
            "issues": list(dict.fromkeys(issues)),
            "suggestions": list(dict.fromkeys(suggestions)),
            "method": "pairwise_rule_nli_sbert",
            "pair_results": [],
            "global_result": None,
        }

    for prev in previous_answers:
        source = prev["source"]
        prev_text = prev["text"]
        sim = _calc_similarity(prev_text, latest_text)
        nli = _calc_bidirectional_nli(prev_text, latest_text)
        pair_issues, pair_suggestions = _check_rule_conflicts(prev_text, latest_text, source)

        if nli is not None and nli["contradiction"] >= NLI_CONTRADICTION_THRESHOLD:
            if nli["contradiction"] >= NLI_STRONG_CONTRADICTION_THRESHOLD:
                pair_issues.append(f"{source}과 현재 답변 사이의 NLI contradiction 점수가 높아 모순 가능성이 큽니다. (score={nli['contradiction']})")
            else:
                pair_issues.append(f"{source}과 현재 답변 사이에 NLI 기준 모순 가능성이 있습니다. (score={nli['contradiction']})")
            pair_suggestions.append(f"{source}에서 말한 내용과 현재 답변의 역할·권한·기여도 표현이 함께 성립하는지 확인해 주세요.")

        if sim is not None and sim < CONTEXT_LOW_SIM_THRESHOLD:
            pair_issues.append(f"{source}과 현재 답변의 의미 유사도가 낮아 같은 경험을 이어서 설명하는지 확인이 필요합니다.")
            pair_suggestions.append("현재 답변이 기존 경험의 같은 사례를 보완하는 내용인지 확인해 주세요.")

        if sim is not None and sim >= CONTEXT_RELATED_THRESHOLD and pair_issues:
            pair_suggestions.append("같은 경험을 말하고 있을 가능성이 높으므로, 역할·권한·기여도 표현이 충돌하지 않게 정리해 주세요.")

        pair_results.append({
            "source": source,
            "similarity": sim,
            "nli": nli,
            "issues": pair_issues,
        })
        issues.extend(pair_issues)
        suggestions.extend(pair_suggestions)

    # 전체 이전 답변을 합친 누적 맥락과도 비교한다. 단, 이 결과는 보조 신호로만 사용한다.
    global_text = "\n".join(prev["text"] for prev in previous_answers)
    global_similarity = _calc_similarity(global_text, latest_text)
    global_nli = _calc_bidirectional_nli(global_text, latest_text)
    global_issues, global_suggestions = _check_rule_conflicts(global_text, latest_text, "전체 이전 답변")

    if global_nli is not None and global_nli["contradiction"] >= NLI_CONTRADICTION_THRESHOLD:
        global_issues.append(f"전체 이전 답변과 현재 답변 사이의 NLI contradiction 점수가 높아 누적 맥락과 충돌할 가능성이 있습니다. (score={global_nli['contradiction']})")
        global_suggestions.append("전체 답변 흐름에서 역할, 권한, 기여도, 시점을 한 번에 정리해 주세요.")

    if global_similarity is not None and global_similarity < CONTEXT_LOW_SIM_THRESHOLD:
        global_issues.append("전체 이전 답변 흐름과 현재 답변의 의미 유사도가 낮아 맥락 이탈 가능성이 있습니다.")
        global_suggestions.append("현재 답변이 기존 답변의 같은 경험을 이어서 설명하는지 확인해 주세요.")

    issues.extend(global_issues)
    suggestions.extend(global_suggestions)

    return {
        "status": "주의" if issues else "양호",
        "issues": list(dict.fromkeys(issues)),
        "suggestions": list(dict.fromkeys(suggestions)),
        "method": "pairwise_rule_nli_sbert",
        "pair_results": pair_results,
        "global_result": {
            "source": "전체 이전 답변",
            "similarity": global_similarity,
            "nli": global_nli,
            "issues": global_issues,
        },
    }


def consistency_note_for_prompt(result: dict) -> str:
    """일관성 검사 결과를 꼬리질문 생성 프롬프트에 넣을 짧은 메모로 변환한다."""
    if result["status"] == "양호":
        return "현재까지 큰 모순은 없지만, 이전 답변을 반복하지 말고 더 깊은 정보를 물어볼 것."
    parts = []
    parts.extend(result.get("issues", []))
    parts.extend(result.get("suggestions", []))
    return " / ".join(parts)


def show_consistency(result: dict) -> None:
    print("\n  일관성 검사")
    if result["status"] == "양호":
        print("  ✓ 현재까지 큰 모순은 감지되지 않았습니다.\n")
        return
    print("  ⚠ 확인 필요")
    for issue in result["issues"]:
        print(f"  - {issue}")
    if result["suggestions"]:
        print("  보완 방향")
        for sug in result["suggestions"]:
            print(f"  → {sug}")
    print()
