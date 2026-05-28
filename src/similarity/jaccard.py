# STEP 3 - Jaccard 유사도
# 키워드 집합의 겹침을 측정한다.
# 의미를 모르지만 "어떤 키워드가 없는지"를 가장 명확하게 알려주는 방법


def score(job_keywords: list[str], para_tokens: list[str]) -> float:
    """
    채용공고 키워드 집합과 문단 토큰 집합의 Jaccard 유사도를 계산한다.
    두 집합이 완전히 같으면 1.0, 겹치는 게 없으면 0.0.
    """
    job_set = set(job_keywords)
    para_set = set(para_tokens)
    union = job_set | para_set
    if not union:
        return 0.0
    return len(job_set & para_set) / len(union)


def coverage(job_keywords: list[str], para_tokens: list[str]) -> dict:
    """
    채용공고 키워드 기준으로 문단에 포함된/누락된 키워드를 분류한다.
    이 결과가 priority_keywords 생성과 rewriter 프롬프트에 직접 활용된다.
    """
    job_set = set(job_keywords)
    para_set = set(para_tokens)

    covered = sorted(job_set & para_set)
    missing = sorted(job_set - para_set)

    return {
        'covered': covered,
        'missing': missing,
        'ratio': round(len(covered) / len(job_set), 3) if job_set else 0.0,
    }
