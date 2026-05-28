# STEP 7 - AI 기여도 통합 분석 리포트
# 텍스트 변화율 + 세션 행동 로그 + 언어 스타일 변화를 가중 평균해
# 최종 AI 기여도를 0~100% 수치로 산출한다

from src.analysis import edit_distance, style_analyzer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

# 사용자 선택별 AI 기여도 점수 (낮을수록 사용자가 더 개입한 것)
CHOICE_AI_SCORE = {
    'y': 0.9,   # AI 제안을 그대로 수락
    'n': 0.7,   # 재생성 후 수락 (AI 주도, 사용자 검토)
    'r': 0.4,   # 사용자 요구사항 반영 (공동 작성)
    's': 0.0,   # 원문 유지 (AI 기여 없음)
}

# 3가지 관점의 가중치
WEIGHT_EDIT    = 0.40   # 텍스트 변화율이 가장 직접적인 지표
WEIGHT_SESSION = 0.35   # 세션 행동이 두 번째로 중요
WEIGHT_STYLE   = 0.25   # 스타일 변화는 보조 지표


def generate_report(
    originals: list[str],
    finals: list[str],
    session_log: list[dict],
) -> dict:
    """
    3가지 관점을 통합해 문단별·전체 AI 기여도를 계산하고 출력한다.

    Args:
        originals:   원문 문단 리스트
        finals:      최종 문단 리스트
        session_log: STEP 5 세션 로그 (choice, idx 포함)

    Returns:
        {
            'per_paragraph': list[dict],
            'overall_ai_contribution': float,
            'verdict': str,
        }
    """
    edit_results = edit_distance.evaluate_all(originals, finals)
    choice_map = {log['idx']: log['choice'] for log in session_log}

    per_paragraph = []

    for i, (orig, final) in enumerate(zip(originals, finals)):
        # 관점 1: 텍스트 변화율
        edit_ratio = edit_results[i]['edit_ratio']

        # 관점 2: 세션 선택 기반 AI 기여도
        choice = choice_map.get(i, 's')
        session_score = CHOICE_AI_SCORE.get(choice, 0.0)

        # 관점 3: 언어 스타일 변화 (Perplexity가 낮아질수록 AI 특성)
        style = style_analyzer.analyze_delta(orig, final)
        # Perplexity 감소 비율로 AI 스타일 점수 계산 (0~1 정규화)
        ppl_drop = max(0, -style['perplexity_delta'])
        ppl_before = style['perplexity_before'] or 1
        style_score = min(ppl_drop / ppl_before, 1.0)

        # 가중 평균
        ai_score = (
            WEIGHT_EDIT    * edit_ratio
            + WEIGHT_SESSION * session_score
            + WEIGHT_STYLE   * style_score
        )
        ai_pct = round(ai_score * 100, 1)

        per_paragraph.append({
            'paragraph_idx': i,
            'edit_ratio': edit_ratio,
            'change_level': edit_results[i]['change_level'],
            'session_choice': choice,
            'session_score': session_score,
            'style_delta': style,
            'ai_contribution_pct': ai_pct,
            'verdict': _classify(ai_pct),
        })

    # 문단 길이 비율로 가중 평균해 전체 AI 기여도 산출
    total_len = sum(len(f) for f in finals) or 1
    overall = sum(
        p['ai_contribution_pct'] * len(finals[p['paragraph_idx']]) / total_len
        for p in per_paragraph
    )
    overall = round(overall, 1)

    report = {
        'per_paragraph': per_paragraph,
        'overall_ai_contribution': overall,
        'verdict': _classify(overall),
    }

    _print_report(report, session_log)
    return report


def _classify(pct: float) -> str:
    """AI 기여도 %를 4단계로 분류한다."""
    if pct < 20:
        return '자필 중심 (AI 보조)'
    elif pct < 50:
        return '혼합 작성 (사용자 + AI)'
    elif pct < 80:
        return 'AI 주도 (사용자 검토)'
    else:
        return 'AI 작성 (사용자 확인만)'


def _print_report(report: dict, session_log: list[dict]) -> None:
    """AI 기여도 분석 리포트를 Rich로 출력한다."""
    console.rule('[bold magenta]🤖 AI 작성 기여도 분석 리포트[/bold magenta]')

    table = Table(box=box.SIMPLE)
    table.add_column('문단', style='dim')
    table.add_column('변화율')
    table.add_column('변화 수준')
    table.add_column('선택')
    table.add_column('AI 기여도')
    table.add_column('판정')

    for p in report['per_paragraph']:
        table.add_row(
            str(p['paragraph_idx'] + 1),
            f"{p['edit_ratio'] * 100:.0f}%",
            p['change_level'],
            p['session_choice'],
            f"{p['ai_contribution_pct']}%",
            p['verdict'],
        )

    console.print(table)
    console.print(Panel(
        f"전체 AI 기여도: [bold]{report['overall_ai_contribution']}%[/bold]  "
        f"→  [bold magenta]{report['verdict']}[/bold magenta]",
        border_style='magenta',
    ))
