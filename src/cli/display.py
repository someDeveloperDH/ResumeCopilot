# # STEP 5 - Rich 기반 CLI 화면 출력
# # 각 문단의 분석 결과와 제안 문단을 보기 좋게 출력한다
#
# from rich.console import Console
# from rich.panel import Panel
# from rich.table import Table
# from rich import box
# from src.similarity.schema import ComparisonResult
#
# console = Console()
#
#
# def show_step_header(step: int, title: str) -> None:
#     """단계 시작 구분선을 출력한다."""
#     console.rule(f'[bold cyan]STEP {step} — {title}[/bold cyan]')
#
#
# def show_eval(label: str, data: dict) -> None:
#     """평가 지표를 테이블로 출력한다."""
#     table = Table(box=box.SIMPLE, show_header=False)
#     table.add_column('지표', style='dim')
#     table.add_column('값')
#     for k, v in data.items():
#         table.add_row(str(k), str(v))
#     console.print(Panel(table, title=f'[bold]{label}[/bold]', border_style='green'))
#
#
# def show_paragraph_panel(
#     result: ComparisonResult,
#     feedback: str,
#     rewritten: str,
#     total: int,
#     question: str | None = None,
# ) -> None:
#     """
#     CLI 루프에서 한 문단의 전체 정보를 출력한다.
#     원문 → 피드백 → 제안 순서로 표시해 사용자가 비교하기 쉽게 구성.
#     """
#     idx = result.paragraph_idx
#     label = '문항' if question else '문단'
#     console.rule(f'[bold]{label} {idx + 1}/{total}[/bold]')
#
#     if question:
#         console.print(Panel(question, title='[bold cyan]질문[/bold cyan]', border_style='cyan'))
#
#     # 피드백 (점수, 긴급도, 누락 키워드)
#     console.print(Panel(feedback, title='📊 분석 결과', border_style='yellow'))
#
#     # 원문 답변
#     console.print(Panel(
#         result.paragraph_text,
#         title='[dim]원문 답변[/dim]' if question else '[dim]원문[/dim]',
#         border_style='dim',
#     ))
#
#     # 제안 (STAR 재구성)
#     console.print(Panel(
#         rewritten,
#         title='[bold green]제안 — STAR 재구성[/bold green]',
#         border_style='green',
#     ))
#
#
# def show_session_prompt() -> None:
#     """사용자 입력 안내를 출력한다."""
#     console.print(
#         '\n [bold]> [y][/bold] 적용  '
#         '[bold][n][/bold] 재생성  '
#         '[bold][r][/bold] 요구사항 입력  '
#         '[bold][s][/bold] 건너뜀\n'
#     )
#
#
# def show_final_summary(
#     results_before: list[float],
#     results_after: list[float],
#     session_log: list[dict],
# ) -> None:
#     """STEP 6 최종 요약 화면을 출력한다."""
#     console.rule('[bold cyan]📊 최종 평가 리포트[/bold cyan]')
#
#     avg_before = sum(results_before) / len(results_before) if results_before else 0
#     avg_after = sum(results_after) / len(results_after) if results_after else 0
#     improvement = ((avg_after - avg_before) / avg_before * 100) if avg_before else 0
#
#     console.print(f'  전체 유사도: {avg_before:.2f} → {avg_after:.2f}  '
#                   f'([bold green]+{improvement:.0f}% 향상[/bold green])')
#
#     table = Table(box=box.SIMPLE)
#     table.add_column('문단', style='dim')
#     table.add_column('선택')
#     table.add_column('이전 점수')
#     table.add_column('최종 점수')
#     table.add_column('상태')
#
#     for log in session_log:
#         status = '수정됨 ✓' if log['choice'] in ('y', 'r') else '원문 유지'
#         table.add_row(
#             str(log['idx'] + 1),
#             log['choice'],
#             f"{log['score_before']:.2f}",
#             f"{log['score_after']:.2f}",
#             status,
#         )
#
#     console.print(table)
#     console.print('\n [bold][w][/bold] 저장   [bold][p][/bold] 미리보기   [bold][d][/bold] 취소\n')






# display.py
# STEP 5 - Rich 기반 CLI 화면 출력

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.similarity.schema import ComparisonResult

console = Console()


def show_step_header(step: int, title: str) -> None:
    """단계 시작 구분선을 출력한다."""
    console.rule(f'[bold cyan]STEP {step} — {title}[/bold cyan]')


def show_eval(label: str, data: dict) -> None:
    """평가 지표를 테이블로 출력한다."""
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column('지표', style='dim')
    table.add_column('값')
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(Panel(table, title=f'[bold]{label}[/bold]', border_style='green'))


def show_paragraph_panel(
    result: ComparisonResult,
    feedback: str,
    rewritten: str,
    total: int,
    question: str | None = None,
    template_score: float | None = None,  # 템플릿 매칭 점수 추가
) -> None:
    """
    CLI 루프에서 한 문단의 전체 정보를 출력한다.
    원문 → 피드백 → 최종 제안(STAR 및 템플릿 반영) 순서로 표시.
    """
    idx = result.paragraph_idx
    label = '문항' if question else '문단'
    console.rule(f'[bold]{label} {idx + 1}/{total}[/bold]')

    if question:
        console.print(Panel(question, title='[bold cyan]질문[/bold cyan]', border_style='cyan'))

    # 초기 채용공고 분석 결과
    console.print(Panel(feedback, title='📊 초기 공고 분석 결과', border_style='yellow'))

    # 원문 답변
    console.print(Panel(
        result.paragraph_text,
        title='[dim]원문 답변[/dim]' if question else '[dim]원문[/dim]',
        border_style='dim',
    ))

    # 10대 항목 심층 인터뷰 + 템플릿 규칙이 반영된 최종 제안
    title_suffix = f" (템플릿 정합성: {template_score:.2f})" if template_score is not None else ""
    console.print(Panel(
        rewritten,
        title=f'[bold green]✨ AI 최종 제안 — [임팩트 요약문 + STAR 흐름]{title_suffix}[/bold green]',
        border_style='green',
    ))


def show_session_prompt() -> None:
    """사용자 입력 안내를 출력한다."""
    console.print(
        '\n [bold]> [y][/bold] 적용(최종 저장본에 반영)  '
        '[bold][n][/bold] 템플릿 기반 즉시 재생성  '
        '[bold][r][/bold] 요구사항 입력 후 재생성  '
        '[bold][s][/bold] 원문 유지(건너뜀)\n'
    )


def show_final_summary(
    results_before: list[float],
    results_after: list[float],
    session_log: list[dict],
) -> None:
    """STEP 6 최종 요약 화면을 출력한다."""
    console.rule('[bold cyan]📊 최종 평가 리포트[/bold cyan]')

    avg_before = sum(results_before) / len(results_before) if results_before else 0
    avg_after = sum(results_after) / len(results_after) if results_after else 0
    improvement = ((avg_after - avg_before) / avg_before * 100) if avg_before else 0

    console.print(f'  전체 공고 유사도: {avg_before:.2f} → {avg_after:.2f}  '
                  f'([bold green]+{improvement:.0f}% 향상[/bold green])')

    table = Table(box=box.SIMPLE)
    table.add_column('문단', style='dim')
    table.add_column('선택')
    table.add_column('이전 점수')
    table.add_column('최종 점수')
    table.add_column('상태')

    for log in session_log:
        status = '수정됨 ✓' if log['choice'] in ('y', 'r') else '원문 유지'
        table.add_row(
            str(log['idx'] + 1),
            log['choice'],
            f"{log['score_before']:.2f}",
            f"{log['score_after']:.2f}",
            status,
        )

    console.print(table)
    console.print('\n [bold][w][/bold] 파일 저장   [bold][p][/bold] 전체 미리보기   [bold][d][/bold] 취소 후 종료\n')
