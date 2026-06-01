# JobFit v2 — test1 + jintea1 + jinyong1 통합 파이프라인
#
# test1 대비 추가/변경 사항:
#   - 키워드 추출: Rule / AI / Hybrid 3가지 비교 + P/R/F1 평가 (jintea1)
#   - 키워드 타입: Hard / Soft / Action 분류 (jintea1)
#   - 기술명 정규화: 파이썬→Python 등 (jintea1)
#   - 섹션 분리: 강화된 헤더 탐지 (jintea1)
#   - 엔티티 추출: 경력/학력/어학 조건 (jintea1)
#   - 자소서 입력: 문항(### 1.) 형식 자동 감지 (jinyong1)
#   - LLM: EXAONE-3.5-7.8B 로컬 (test1 유지, Gemini 미사용)

import time
import click
from rich.console import Console

from src.input import fetcher, parser
from src.preprocessing.cleaner import clean_job_posting, clean_cover_letter
from src.preprocessing.section_splitter import split_sections
from src.preprocessing.tokenizer import auto_split, evaluate_paragraphs
from src.keyword import rule_based, ai_based, hybrid
from src.keyword.entity_extractor import extract_entities
from src.keyword.evaluator import compare as eval_compare, print_report
from src.similarity.aggregator import Aggregator
from src.cli import display, session
from src.analysis import ai_reporter

console = Console()


def _elapsed(start: float) -> str:
    secs = time.time() - start
    if secs >= 60:
        m, s = divmod(secs, 60)
        return f'{int(m)}분 {s:.1f}초'
    return f'{secs:.2f}초'


def _done(step: int, start: float) -> None:
    console.print(f'[bold green]✓ STEP {step} 완료[/bold green] — [cyan]{_elapsed(start)}[/cyan]\n')


@click.command()
@click.option('--cover',      required=True, help='자소서 파일 (.txt/.pdf) — 문항(### 1.) 또는 문단 형식 모두 지원')
@click.option('--job',        required=True, help='채용공고 URL 또는 파일 (.txt)')
@click.option('--output',     default='output_cover.txt', help='저장 파일 경로')
@click.option('--split-mode', default='auto', type=click.Choice(['auto', 'regex', 'nlp']),
              help='문단 분리 방식 (문항 형식 자소서는 무시됨)')
@click.option('--keyword-method', default='hybrid', type=click.Choice(['rule', 'ai', 'hybrid']),
              help='키워드 추출 방법 (기본: hybrid)')
@click.option('--eval-keywords', is_flag=True, default=False,
              help='키워드 추출 방법 P/R/F1 비교 평가 실행')
def main(cover, job, output, split_mode, keyword_method, eval_keywords):
    """
    채용공고에 맞게 자기소개서를 자동으로 최적화합니다. (v2)

    사용 예시:
        python main.py --cover cover.txt --job job.txt
        python main.py --cover cover.txt --job https://... --keyword-method hybrid
        python main.py --cover cover.txt --job job.txt --eval-keywords
    """
    total_start = time.time()

    # ──────────────────────────────────────────
    # STEP 0: 입력 수집
    # ──────────────────────────────────────────
    display.show_step_header(0, '입력 수집')
    t = time.time()

    if job.startswith('http://') or job.startswith('https://'):
        job_data = fetcher.fetch(job)
        job_raw  = job_data['raw_text']
    else:
        job_data = parser.parse(job)
        job_raw  = job_data['text']

    display.show_eval('STEP 0 — 채용공고', job_data['eval'])
    if not job_data['eval']['fetch_success']:
        console.print(f'[red]채용공고 수집 실패: {job_data["eval"].get("error")}[/red]')
        return

    cover_data = parser.parse(cover)
    display.show_eval('STEP 0 — 자기소개서', cover_data['eval'])
    if not cover_data['eval']['fetch_success']:
        console.print(f'[red]자기소개서 읽기 실패: {cover_data["eval"].get("error")}[/red]')
        return

    _done(0, t)

    # ──────────────────────────────────────────
    # STEP 1: 채용공고 전처리 + 키워드 추출
    # ──────────────────────────────────────────
    display.show_step_header(1, f'채용공고 키워드 추출 [{keyword_method}]')
    t = time.time()

    # jintea1 강화 전처리: 기술명 정규화 + 섹션 분리
    job_clean    = clean_job_posting(job_raw)
    job_sections = split_sections(job_clean)

    # 엔티티 추출 (경력/학력/어학) — rewriter 프롬프트 컨텍스트로 전달
    entities = extract_entities(job_clean)
    if entities:
        console.print(f'[dim]추출된 조건: {entities}[/dim]')

    # 선택 방법으로 키워드 추출
    if keyword_method == 'rule':
        kw_result = rule_based.extract_keywords(job_sections)
        kw_eval   = rule_based.evaluate(kw_result)
    elif keyword_method == 'ai':
        ai_raw    = ai_based.extract_by_section(job_sections)
        kw_result = {
            sec: [{'keyword': kw, 'pos': '', 'type': '', 'weight': 1.0,
                   'score': sc, 'hybrid_score': sc, 'source': 'AI'}
                  for kw, sc in pairs]
            for sec, pairs in ai_raw.items()
        }
        kw_eval   = ai_based.evaluate(ai_raw)
    else:
        kw_result = hybrid.extract_keywords(job_sections)
        kw_eval   = hybrid.evaluate(kw_result)

    display.show_eval(f'STEP 1 — {keyword_method}', kw_eval)

    # P/R/F1 비교 평가 옵션
    if eval_keywords:
        console.print('[dim]키워드 추출 방법 비교 평가 중...[/dim]')
        hybrid_kws = hybrid.to_keyword_list(hybrid.extract_keywords(job_sections))
        comparison = eval_compare(job_raw, hybrid_kws)
        print_report(comparison)

    all_keywords = (
        hybrid.to_keyword_list(kw_result)
        if keyword_method == 'hybrid'
        else list({item['keyword'] for items in kw_result.values() for item in items})
    )

    _done(1, t)

    # ──────────────────────────────────────────
    # STEP 2: 자소서 처리 단위 결정
    # ──────────────────────────────────────────
    display.show_step_header(2, '자소서 처리 단위 분리')
    t = time.time()

    cover_mode = cover_data.get('mode', 'paragraph')

    if cover_mode == 'section' and cover_data.get('sections'):
        # 문항 형식 감지 (jinyong1 방식)
        sections_data = cover_data['sections']
        paragraphs    = [s['answer'] for s in sections_data]
        questions     = [f"{s['number']}. {s['question']}" for s in sections_data]
        used_method   = 'section'
        console.print(f'[dim]문항 형식 감지: {len(paragraphs)}개 문항[/dim]')
    else:
        # 문단 형식 (test1 방식)
        clean_cover = clean_cover_letter(cover_data['text'])
        questions   = None
        if split_mode == 'regex':
            from src.preprocessing.tokenizer import split_paragraphs
            paragraphs, used_method = split_paragraphs(clean_cover), 'regex'
        elif split_mode == 'nlp':
            from src.preprocessing.tokenizer import split_paragraphs_nlp
            paragraphs, used_method = split_paragraphs_nlp(clean_cover), 'nlp'
        else:
            paragraphs, used_method = auto_split(clean_cover)

    para_eval = evaluate_paragraphs(paragraphs, method=used_method)
    display.show_eval('STEP 2 — 분리 결과', para_eval)

    if para_eval['n_paragraphs'] == 0:
        console.print('[red]처리 단위를 찾을 수 없습니다.[/red]')
        return

    _done(2, t)

    # ──────────────────────────────────────────
    # STEP 3: 유사도 분석 (4가지 방법 병렬)
    # ──────────────────────────────────────────
    display.show_step_header(3, '유사도 분석')
    t = time.time()

    aggregator = Aggregator(job_sections, all_keywords)

    console.print('[dim]KoSBERT 배치 추론 중...[/dim]')
    sbert_t = time.time()
    aggregator.precompute_sbert(paragraphs)
    console.print(f'[dim]KoSBERT 완료 — {_elapsed(sbert_t)}[/dim]')

    results    = [aggregator.compare(para, idx) for idx, para in enumerate(paragraphs)]
    step3_eval = aggregator.evaluate_all(results)

    display.show_eval('STEP 3 — 비교 결과', {
        '평균 유사도':    step3_eval['avg_overall_score'],
        '수정 필요 단위': f"{step3_eval['needs_rewrite_count']}개",
        '점수 분산':      step3_eval['avg_score_variance'],
    })
    _done(3, t)

    # ──────────────────────────────────────────
    # STEP 4, 5: 피드백 + STAR 재구성 + CLI 루프
    # ──────────────────────────────────────────
    display.show_step_header(4, '피드백 & STAR 재구성 + 대화형 루프')
    t = time.time()

    # session_result = session.run_session(
    #     paragraphs=paragraphs,
    #     results=results,
    #     output_path=output,
    #     questions=questions,   # 문항 형식 시 질문 텍스트
    #     entities=entities,     # 경력/학력/어학 컨텍스트
    # )
    # _done(4, t)

    session_result = session.run_session(
        paragraphs=paragraphs,
        results=results,
        aggregator= aggregator,
        output_path=output,
        questions=questions,  # 문항 형식 시 질문 텍스트
        entities=entities,  # 경력/학력/어학 컨텍스트
    )
    _done(4, t)

    # ──────────────────────────────────────────
    # STEP 7: AI 기여도 분석
    # ──────────────────────────────────────────
    display.show_step_header(7, 'AI 기여도 분석')
    t = time.time()

    ai_reporter.generate_report(
        originals=paragraphs,
        finals=session_result['final_paragraphs'],
        session_log=session_result['session_log'],
    )
    _done(7, t)

    console.rule('[bold cyan]실행 완료[/bold cyan]')
    console.print(f'[bold]전체 소요 시간: [cyan]{_elapsed(total_start)}[/cyan][/bold]')


if __name__ == '__main__':
    main()
