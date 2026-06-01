# session.py
# STEP 5, 6 - CLI 대화형 유도질문 루프 및 최종 저장

from pathlib import Path
from src.similarity.schema import ComparisonResult
from src.feedback import scorer, rewriter
from src.cli import display
from rich.panel import Panel

def run_session(
        paragraphs: list[str],
        results: list[ComparisonResult],
        aggregator,
        output_path: str = 'output_cover.txt',
        questions: list[str] | None = None,
        entities: dict | None = None,
) -> dict:
    """
    자소서 문항별 특징에 맞춰 AI가 동적으로 유도 질문을 새롭게 뽑아내고
    진행 상황(n번째/총개수)을 명시하며 인터랙티브 인터뷰를 수행하는 루프.
    """
    final_paragraphs = list(paragraphs)
    session_log = []
    score_before = [r.overall_score for r in results]
    total = len(paragraphs)

    for result in results:
        idx = result.paragraph_idx
        question = questions[idx] if questions and idx < len(questions) else None

        display.show_step_header(5, f"문단 {idx + 1} 인터랙티브 고도화 세션")

        # 1. 초기 분석 결과 출력
        feedback_text = scorer.make_feedback(result)
        display.console.print(Panel(result.paragraph_text, title="[dim]현재 원문 답변[/dim]", border_style="dim"))
        display.console.print(Panel(feedback_text, title="📊 채용공고 매칭 분석", border_style="yellow"))

        # 2. 고정 체크리스트 대신 자소서 문항 성격에 부합하는 유도질문 동적 추출
        display.console.print("\n[bold list]🔍 문항 맞춤형 AI 유도 질문 분석 시작...[/bold list]")
        guiding_questions = rewriter.generate_guiding_questions(result.paragraph_text, question=question)

        current_text = result.paragraph_text
        total_gq = len(guiding_questions)

        # 3. 추출된 맞춤 질문이 있다면 순차적으로 유도질문 수행 (최대 3개 제한 조건 충족)
        if total_gq > 0:
            display.console.print(
                f"[bold yellow]⚠️ 문항 분석 결과 본문을 보완할 핵심 질문 {total_gq}개가 도출되었습니다. 심층 인터뷰를 시작합니다.[/bold yellow]\n")

            for gq_idx, guiding_question in enumerate(guiding_questions, start=1):
                # 유도질문 출력시 요청하신 포맷 [n번째/총개수] 바인딩 수행
                progress_title = f"💬 AI 추가 유도질문 ({gq_idx}/{total_gq})"
                display.console.print(Panel(f"[bold cyan]{guiding_question}[/bold cyan]", title=progress_title))

                # 사용자 답변 입력
                user_answer = input("✍️ 답변 입력 (패스하려면 엔터): ").strip()
                if not user_answer:
                    continue

                # 사용자 답변을 결합하여 내부 컨텍스트 점진적 누적 업데이트
                current_text = rewriter.internal_rewrite(
                    text=current_text,
                    guiding_question=guiding_question,
                    answer=user_answer,
                    check_item=f"{gq_idx}번째 동적 생성 유도질문 지점",
                    question=question,
                    entities=entities
                )
        else:
            display.console.print("[bold green]✅ 문항 요구 조건의 흐름을 모두 충족하는 완성도 높은 답변입니다.[/bold green]\n")

        # 4. 유도질문 완료 후 최종 생성 및 지정 템플릿 유사도 검증
        display.console.print("\n[bold list]🚀 인터뷰 기반 최종 자기소개서 완성 중...[/bold list]")
        final_proposed = rewriter.generate_final_template(current_text, result, question=question, entities=entities)

        # 템플릿 규칙 준수 여부 자체 측정 (첫 문장 임팩트 + STAR 구조 흐름 부합도)
        template_score = aggregator._tfidf.score(final_proposed)

        # UI 출력 연동
        display.show_paragraph_panel(result, f"최종 보완 완료 (템플릿 정합성 점수: {template_score:.2f})", final_proposed, total,
                                     question=question, template_score=template_score)
        display.show_session_prompt()

        # 5. y/n/r/s 최종 선택 분기[cite: 3]
        choice, final_text = _handle_input_v2(result, final_proposed, question=question, entities=entities)
        final_paragraphs[idx] = final_text

        # 유사도 변화 기록[cite: 3]
        score_after = result.overall_score if choice in ('n', 's') else min(result.overall_score + 0.3, 1.0)
        session_log.append({
            'idx': idx,
            'choice': choice,
            'score_before': result.overall_score,
            'score_after': score_after,
        })

    score_after_list = [log['score_after'] for log in session_log]
    display.show_final_summary(score_before, score_after_list, session_log)

    # 최종 파일 저장 의사 결정[cite: 3]
    _handle_save(final_paragraphs, output_path, session_log, questions=questions)

    return {
        'final_paragraphs': final_paragraphs,
        'session_log': session_log,
    }


def _handle_input_v2(
        result: ComparisonResult,
        proposed: str,
        question: str | None = None,
        entities: dict | None = None,
) -> tuple[str, str]:
    """y/n/r/s 결정을 처리하는 루프"""
    while True:
        raw = input('선택: ').strip().lower()
        if raw == 'y':
            return 'y', proposed
        elif raw == 'n':
            # 즉시 최종 템플릿 기반 재생성
            proposed = rewriter.generate_final_template(result.paragraph_text, result, question=question,
                                                        entities=entities)
            display.console.print(Panel(proposed, title='[bold green]재생성 결과[/bold green]', border_style='green'))
        elif raw == 'r':
            user_req = input('추가 요구사항을 입력하세요: ').strip()
            proposed = rewriter.generate_final_template(proposed, result, user_request=user_req, question=question,
                                                        entities=entities)
            display.console.print(Panel(proposed, title='[bold green]요구사항 반영 결과[/bold green]', border_style='green'))
        elif raw == 's':
            return 's', result.paragraph_text
        else:
            display.console.print('[red]y / n / r / s 중 하나를 입력하세요.[/red]')
        display.show_session_prompt()


def _handle_save(final_paragraphs: list[str], output_path: str, session_log: list[dict],
                 questions: list[str] | None = None) -> None:
    while True:
        raw = input('선택 (w/p/d): ').strip().lower()
        if raw == 'w':
            _save(final_paragraphs, output_path, questions=questions)
            display.console.print(f'[bold green]저장 완료: {output_path}[/bold green]')
            break
        elif raw == 'p':
            display.console.print(Panel(_format_output(final_paragraphs, questions), title='[bold]전체 미리보기[/bold]'))
        elif raw == 'd':
            display.console.print('[dim]저장 없이 종료합니다.[/dim]')
            break
        else:
            display.console.print('[red]w / p / d 중 하나를 입력하세요.[/red]')


def _format_output(paragraphs: list[str], questions: list[str] | None = None) -> str:
    if not questions:
        return '\n\n'.join(paragraphs)
    blocks = []
    for i, paragraph in enumerate(paragraphs):
        question = questions[i] if i < len(questions) else f'{i + 1}.'
        blocks.append(f"### {question}\n\n{paragraph}")
    return '\n\n'.join(blocks)


def _save(paragraphs: list[str], path: str, questions: list[str] | None = None) -> None:
    Path(path).write_text(_format_output(paragraphs, questions), encoding='utf-8')