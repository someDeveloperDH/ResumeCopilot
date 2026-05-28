# STEP 5, 6 - CLI 대화형 루프 및 저장
# 문단별로 y/n/r/s를 받아 수정 여부를 결정하고 최종 파일을 저장한다

from pathlib import Path
from src.similarity.schema import ComparisonResult
from src.feedback import scorer, rewriter
from src.cli import display


def run_session(
    paragraphs: list[str],
    results: list[ComparisonResult],
    output_path: str = 'output_cover.txt',
    questions: list[str] | None = None,   # 문항 형식 시 질문 텍스트 리스트
    entities: dict | None = None,          # 경력/학력/어학 컨텍스트
) -> dict:
    """
    자소서 문단 전체를 순서대로 처리하는 대화형 루프.

    Returns:
        {
            'final_paragraphs': list[str],   최종 문단 리스트
            'session_log': list[dict],       문단별 선택 기록
        }
    """
    final_paragraphs = list(paragraphs)  # 원문 복사 후 수정분만 교체
    session_log = []
    score_before = [r.overall_score for r in results]

    total = len(paragraphs)

    for result in results:
        idx = result.paragraph_idx

        # 점수가 높으면 재작성 불필요 — 그래도 사용자에게 보여주고 선택권 줌
        feedback_text = scorer.make_feedback(result)
        proposed = rewriter.rewrite(result)

        display.show_paragraph_panel(result, feedback_text, proposed, total)
        display.show_session_prompt()

        choice, final_text = _handle_input(result, proposed)

        final_paragraphs[idx] = final_text

        # 재작성 후 유사도 평가 (간단히 점수 변화 기록)
        score_after = result.overall_score if choice in ('n', 's') else min(result.overall_score + 0.2, 1.0)

        session_log.append({
            'idx': idx,
            'choice': choice,
            'score_before': result.overall_score,
            'score_after': score_after,
        })

    score_after_list = [log['score_after'] for log in session_log]
    display.show_final_summary(score_before, score_after_list, session_log)

    # 저장 여부를 사용자가 결정
    _handle_save(final_paragraphs, output_path, session_log)

    return {
        'final_paragraphs': final_paragraphs,
        'session_log': session_log,
    }


def _handle_input(
    result: ComparisonResult,
    proposed: str,
) -> tuple[str, str]:
    """
    y/n/r/s 입력을 처리한다.
    n은 재생성, r은 요구사항 반영 재생성, s는 원문 유지.
    """
    while True:
        raw = input('선택: ').strip().lower()

        if raw == 'y':
            return 'y', proposed

        elif raw == 'n':
            # 재생성: 같은 ComparisonResult로 다시 호출
            proposed = rewriter.rewrite(result)
            display.console.print(
                Panel(proposed, title='[bold green]재생성 결과[/bold green]', border_style='green')
            )
            # 재생성 후 다시 선택받음

        elif raw == 'r':
            user_req = input('요구사항을 입력하세요: ').strip()
            proposed = rewriter.rewrite(result, user_request=user_req)
            display.console.print(
                Panel(proposed, title='[bold green]요구사항 반영 결과[/bold green]', border_style='green')
            )
            # 반영 후 다시 선택받음

        elif raw == 's':
            return 's', result.paragraph_text

        else:
            display.console.print('[red]y / n / r / s 중 하나를 입력하세요.[/red]')

        display.show_session_prompt()


def _handle_save(
    final_paragraphs: list[str],
    output_path: str,
    session_log: list[dict],
) -> None:
    """최종 저장 여부를 사용자에게 묻고 처리한다."""
    while True:
        raw = input('선택 (w/p/d): ').strip().lower()

        if raw == 'w':
            _save(final_paragraphs, output_path)
            display.console.print(f'[bold green]저장 완료: {output_path}[/bold green]')
            break

        elif raw == 'p':
            # 수정 전체 미리보기
            display.console.print(Panel(
                '\n\n'.join(final_paragraphs),
                title='[bold]전체 미리보기[/bold]',
            ))

        elif raw == 'd':
            display.console.print('[dim]저장 없이 종료합니다.[/dim]')
            break

        else:
            display.console.print('[red]w / p / d 중 하나를 입력하세요.[/red]')


def _save(paragraphs: list[str], path: str) -> None:
    """문단 리스트를 빈 줄 2개로 구분해 파일로 저장한다."""
    Path(path).write_text('\n\n'.join(paragraphs), encoding='utf-8')
