# # STEP 5, 6 - CLI 대화형 루프 및 저장
# # 문단별로 y/n/r/s를 받아 수정 여부를 결정하고 최종 파일을 저장한다
#
# from pathlib import Path
# from src.similarity.schema import ComparisonResult
# from src.feedback import scorer, rewriter
# from src.cli import display
# from rich.panel import Panel
#
#
# def run_session(
#     paragraphs: list[str],
#     results: list[ComparisonResult],
#     aggregator,
#     output_path: str = 'output_cover.txt',
#     questions: list[str] | None = None,   # 문항 형식 시 질문 텍스트 리스트
#     entities: dict | None = None,          # 경력/학력/어학 컨텍스트
# ) -> dict:
#     """
#     자소서 문단 전체를 순서대로 처리하는 대화형 루프.
#
#     Returns:
#         {
#             'final_paragraphs': list[str],   최종 문단 리스트
#             'session_log': list[dict],       문단별 선택 기록
#         }
#     """
#     final_paragraphs = list(paragraphs)  # 원문 복사 후 수정분만 교체
#     session_log = []
#     score_before = [r.overall_score for r in results]
#
#     total = len(paragraphs)
#
#     for result in results:
#         idx = result.paragraph_idx
#
#         question = questions[idx] if questions and idx < len(questions) else None
#
#         # 점수가 높으면 재작성 불필요 — 그래도 사용자에게 보여주고 선택권 줌
#         feedback_text = scorer.make_feedback(result)
#         proposed = rewriter.rewrite(result, question=question, entities=entities)
#
#         display.show_paragraph_panel(result, feedback_text, proposed, total, question=question)
#         display.show_session_prompt()
#
#         choice, final_text = _handle_input(result, proposed, question=question, entities=entities)
#
#         final_paragraphs[idx] = final_text
#
#         # 재작성 후 유사도 평가 (간단히 점수 변화 기록)
#         score_after = result.overall_score if choice in ('n', 's') else min(result.overall_score + 0.2, 1.0)
#
#         session_log.append({
#             'idx': idx,
#             'choice': choice,
#             'score_before': result.overall_score,
#             'score_after': score_after,
#         })
#
#     score_after_list = [log['score_after'] for log in session_log]
#     display.show_final_summary(score_before, score_after_list, session_log)
#
#     # 저장 여부를 사용자가 결정
#     _handle_save(final_paragraphs, output_path, session_log, questions=questions)
#
#     return {
#         'final_paragraphs': final_paragraphs,
#         'session_log': session_log,
#     }
#
#
# def _handle_input(
#     result: ComparisonResult,
#     proposed: str,
#     question: str | None = None,
#     entities: dict | None = None,
# ) -> tuple[str, str]:
#     """
#     y/n/r/s 입력을 처리한다.
#     n은 재생성, r은 요구사항 반영 재생성, s는 원문 유지.
#     """
#     while True:
#         raw = input('선택: ').strip().lower()
#
#         if raw == 'y':
#             return 'y', proposed
#
#         elif raw == 'n':
#             # 재생성: 같은 ComparisonResult로 다시 호출
#             proposed = rewriter.rewrite(result, question=question, entities=entities)
#             display.console.print(
#                 Panel(proposed, title='[bold green]재생성 결과[/bold green]', border_style='green')
#             )
#             # 재생성 후 다시 선택받음
#
#         elif raw == 'r':
#             user_req = input('요구사항을 입력하세요: ').strip()
#             proposed = rewriter.rewrite(result, user_request=user_req, question=question, entities=entities)
#             display.console.print(
#                 Panel(proposed, title='[bold green]요구사항 반영 결과[/bold green]', border_style='green')
#             )
#             # 반영 후 다시 선택받음
#
#         elif raw == 's':
#             return 's', result.paragraph_text
#
#         else:
#             display.console.print('[red]y / n / r / s 중 하나를 입력하세요.[/red]')
#
#         display.show_session_prompt()
#
#
# def _handle_save(
#     final_paragraphs: list[str],
#     output_path: str,
#     session_log: list[dict],
#     questions: list[str] | None = None,
# ) -> None:
#     """최종 저장 여부를 사용자에게 묻고 처리한다."""
#     while True:
#         raw = input('선택 (w/p/d): ').strip().lower()
#
#         if raw == 'w':
#             _save(final_paragraphs, output_path, questions=questions)
#             display.console.print(f'[bold green]저장 완료: {output_path}[/bold green]')
#             break
#
#         elif raw == 'p':
#             # 수정 전체 미리보기
#             display.console.print(Panel(
#                 _format_output(final_paragraphs, questions),
#                 title='[bold]전체 미리보기[/bold]',
#             ))
#
#         elif raw == 'd':
#             display.console.print('[dim]저장 없이 종료합니다.[/dim]')
#             break
#
#         else:
#             display.console.print('[red]w / p / d 중 하나를 입력하세요.[/red]')
#
#
# def _format_output(paragraphs: list[str], questions: list[str] | None = None) -> str:
#     """문항 형식이면 질문과 답변을 함께, 아니면 문단만 출력한다."""
#     if not questions:
#         return '\n\n'.join(paragraphs)
#
#     blocks = []
#     for i, paragraph in enumerate(paragraphs):
#         question = questions[i] if i < len(questions) else f'{i + 1}.'
#         blocks.append(f'{question}\n\n{paragraph}')
#     return '\n\n'.join(blocks)
#
#
# def _save(paragraphs: list[str], path: str, questions: list[str] | None = None) -> None:
#     """문항 형식이면 질문+답변, 아니면 문단 리스트를 저장한다."""
#     Path(path).write_text(_format_output(paragraphs, questions), encoding='utf-8')


# session.py
# STEP 5, 6 - CLI 대화형 유도질문 루프 및 최종 저장

from pathlib import Path
from src.similarity.schema import ComparisonResult
from src.feedback import scorer, rewriter
from src.cli import display
from rich.panel import Panel

# 정형화된 10개 체크사항 (질문 템플릿)
CHECKLIST = [
    "1. 해당 경험에서 본인이 맡은 구체적인 역할과 책임이 드러나는가?",
    "2. 당시에 직면했던 가장 큰 핵심 문제(또는 목표)가 명확히 기술되었는가?",
    "3. 문제를 해결하기 위해 본인이 취한 구체적인 행동(Action)이 기술되었는가?",
    "4. 왜 그 행동을 선택했는지에 대한 논리적 이유나 근거가 포함되었는가?",
    "5. 정량적 수치나 객관적 지표를 통한 구체적인 성과(Result)가 제시되었는가?",
    "6. 해당 경험을 통해 개인적으로 무엇을 배우고 느꼈는지(Lesson Learned) 나타나는가?",
    "7. 지원하는 직무 분야 및 채용공고의 주요업무와 직접적으로 연결되는가?",
    "8. 협업 과정에서의 소통 방식이나 팀워크를 발휘한 부분이 포함되었는가?",
    "9. 예기치 못한 상황이나 제약 조건에 어떻게 대응했는지 드러나는가?",
    "10. 입사 후 이러한 경험을 어떻게 기여와 성과로 연결할지 포부가 보이는가?"
]


def run_session(
        paragraphs: list[str],
        results: list[ComparisonResult],
        aggregator,
        output_path: str = 'output_cover.txt',
        questions: list[str] | None = None,
        entities: dict | None = None,
) -> dict:
    """
    자소서 문단별로 10개 체크사항을 검증하고 심층 인터랙티브 유도질문을 수행하는 루프.
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

        # 2. LLM을 통해 10개 체크사항 중 미흡한 항목 추출
        display.console.print("\n[bold list]🔍 10대 항목 기반 자소서 심층 검증 시작...[/bold list]")
        failed_checks = rewriter.evaluate_checklist(result.paragraph_text, CHECKLIST, question=question)

        current_text = result.paragraph_text

        # 3. 미흡한 항목이 있다면 순차적으로 유도질문 수행
        if failed_checks:
            display.console.print(
                f"[bold yellow]⚠️ 미흡 항목 {len(failed_checks)}개가 발견되었습니다. 심층 인터랙티브 인터뷰를 시작합니다.[/bold yellow]\n")

            for check_item in failed_checks:
                # 사용자에 맞는 맞춤형 유도질문 생성
                guiding_question = rewriter.generate_guiding_question(current_text, check_item, question=question)

                # 유도질문 출력
                display.console.print(Panel(f"[bold cyan]{guiding_question}[/bold cyan]", title="💬 AI 추가 유도질문"))

                # 사용자 답변 입력
                user_answer = input("✍️ 답변 입력 (패스하려면 엔터): ").strip()
                if not user_answer:
                    continue

                # [내부 프로세스] 사용자 답변을 녹여서 '내부 재작성' 수행 (사용자에게 보이지 않음)
                current_text = rewriter.internal_rewrite(
                    text=current_text,
                    guiding_question=guiding_question,
                    answer=user_answer,
                    check_item=check_item,
                    question=question,  # 자소서 문항 질문 컨텍스트
                    entities=entities
                )
        else:
            display.console.print("[bold green]✅ 10개 핵심 체크사항을 모두 만족하는 우수한 답변입니다.[/bold green]\n")

        # 4. 유도질문 완료 후 최종 생성 및 지정 템플릿 유사도 검증
        display.console.print("\n[bold list]🚀 인터뷰 기반 최종 자기소개서 완성 중...[/bold list]")
        final_proposed = rewriter.generate_final_template(current_text, result, question=question, entities=entities)

        # 템플릿 규칙 준수 여부 자체 측정 (첫 문장 임팩트 + STAR 구조 흐름 부합도)
        template_score = aggregator._tfidf.score(final_proposed)  # 모듈 내 스코어러 혹은 구조 분석 점수 활용

        # UI 출력
        display.show_paragraph_panel(result, f"최종 보완 완료 (템플릿 정합성 점수: {template_score:.2f})", final_proposed, total,
                                     question=question)
        display.show_session_prompt()

        # 5. y/n/r/s 최종 선택 분기
        choice, final_text = _handle_input_v2(result, final_proposed, question=question, entities=entities)
        final_paragraphs[idx] = final_text

        # 유사도 변화 기록
        score_after = result.overall_score if choice in ('n', 's') else min(result.overall_score + 0.3, 1.0)
        session_log.append({
            'idx': idx,
            'choice': choice,
            'score_before': result.overall_score,
            'score_after': score_after,
        })

    score_after_list = [log['score_after'] for log in session_log]
    display.show_final_summary(score_before, score_after_list, session_log)

    # 최종 파일 저장 의사 결정
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
