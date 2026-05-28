# STEP 5 - Rich 기반 CLI 화면 출력
# 각 문단의 분석 결과와 제안 문단을 보기 좋게 출력한다

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
) -> None:
    """
    CLI 루프에서 한 문단의 전체 정보를 출력한다.
    원문 → 피드백 → 제안 순서로 표시해 사용자가 비교하기 쉽게 구성.
    """
    idx = result.paragraph_idx
    console.rule(f'[bold]문단 {idx + 1}/{total}[/bold]')

    # 피드백 (점수, 긴급도, 누락 키워드)
    console.print(Panel(feedback, title='📊 분석 결과', border_style='yellow'))

    # 원문
    console.print(Panel(
        result.paragraph_text,
        title='[dim]원문[/dim]',
        border_style='dim',
    ))

    # 제안 (STAR 재구성)
    console.print(Panel(
        rewritten,
        title='[bold green]제안 — STAR 재구성[/bold green]',
        border_style='green',
    ))


def show_session_prompt() -> None:
    """사용자 입력 안내를 출력한다."""
    console.print(
        '\n [bold]> [y][/bold] 적용  '
        '[bold][n][/bold] 재생성  '
        '[bold][r][/bold] 요구사항 입력  '
        '[bold][s][/bold] 건너뜀\n'
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

    console.print(f'  전체 유사도: {avg_before:.2f} → {avg_after:.2f}  '
                  f'([bold green]+{improvement:.0f}% 향상[/bold green])')

    table = Table(box=box.SIMPLE)
    table.add_column('문단', style='dim')
    table.add_column('선택')
    table.add_column('이전 점수')
    table.add_column('최종 점수')
    table.add_column('상태')

    for log in session_log:
        status = '수정됨 ✓' if log['choice'] in ('y', 'r') else (
            '원문 유지' if log['choice'] == 'n' else '건너뜀 -'
        )
        table.add_row(
            str(log['idx'] + 1),
            log['choice'],
            f"{log['score_before']:.2f}",
            f"{log['score_after']:.2f}",
            status,
        )

    console.print(table)
    console.print('\n [bold][w][/bold] 저장   [bold][p][/bold] 미리보기   [bold][d][/bold] 취소\n')
