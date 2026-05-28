"""
JobFit v2 시스템 아키텍처 다이어그램 생성 스크립트
실행: python draw_architecture.py
출력: architecture.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

# WSL2에서 Windows 맑은 고딕 폰트 등록
_KO_FONT = '/mnt/c/Windows/Fonts/malgun.ttf'
font_manager.fontManager.addfont(_KO_FONT)
_KO_PROP = font_manager.FontProperties(fname=_KO_FONT)
_KO_NAME = _KO_PROP.get_name()
plt.rcParams['font.family'] = _KO_NAME
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#0f1117')

# ── 색상 팔레트 ──────────────────────────────────────────
C = {
    'input':    '#1e3a5f',
    'step0':    '#1a472a',
    'step1':    '#2d1b69',
    'step2':    '#1a3a4a',
    'step3':    '#3a1a1a',
    'step4':    '#3a2a1a',
    'step5':    '#1a2a3a',
    'step7':    '#2a3a1a',
    'output':   '#1e3a5f',
    'sub':      '#1a1a2e',
    'border':   '#ffffff',
    'arrow':    '#4fc3f7',
    'text':     '#ffffff',
    'subtext':  '#b0bec5',
    'tag':      '#ffd54f',
    'new':      '#69f0ae',
    'bg':       '#0f1117',
}


def box(ax, x, y, w, h, label, color, fontsize=11, bold=True, tag=None):
    """메인 스텝 박스."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor=C['border'],
        linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, color=C['text'], fontweight=weight, zorder=4,
            multialignment='center')
    if tag:
        ax.text(x + w - 0.15, y + h - 0.15, tag, ha='right', va='top',
                fontsize=7, color=C['new'], fontweight='bold', zorder=5)


def subbox(ax, x, y, w, h, items, color='#1e2a3a', ncol=2):
    """서브 컴포넌트 박스."""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='#4a5568',
        linewidth=0.8, zorder=3
    )
    ax.add_patch(rect)
    n = len(items)
    per_col = (n + ncol - 1) // ncol
    col_w = w / ncol
    for i, item in enumerate(items):
        col   = i // per_col
        row   = i % per_col
        cx    = x + col * col_w + col_w / 2
        cy    = y + h - 0.22 - row * (h / per_col)
        ax.text(cx, cy, item, ha='center', va='center',
                fontsize=7.5, color=C['subtext'], zorder=4)


def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arrow'],
                                lw=2.0, connectionstyle='arc3,rad=0.0'),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.2, my, label, fontsize=7.5,
                color=C['arrow'], va='center', zorder=5)


def side_label(ax, x, y, text, color='#546e7a'):
    ax.text(x, y, text, fontsize=7, color=color, va='center',
            style='italic', zorder=4)


# ── 제목 ─────────────────────────────────────────────────
ax.text(10, 27.5, 'JobFit v2 — 시스템 아키텍처',
        ha='center', va='center', fontsize=17,
        color='white', fontweight='bold')
ax.text(10, 27.1,
        'test1 (기반) + jintea1 (키워드 고도화) + jinyong1 (문항 파싱)',
        ha='center', va='center', fontsize=9, color='#90a4ae')

# ── 범례 ─────────────────────────────────────────────────
ax.text(0.4, 27.4, '■ NEW (jintea1/jinyong1)', fontsize=8,
        color=C['new'], fontweight='bold')

# ══════════════════════════════════════════════════════════
# 입력
# ══════════════════════════════════════════════════════════
box(ax, 3.5, 25.6, 13, 1.0,
    '사용자 입력', C['input'], fontsize=12)
subbox(ax, 3.6, 25.65, 6.2, 0.85,
       ['자소서 파일  (.txt / .pdf)', '문항(### 1.) 또는 문단 형식'],
       ncol=1)
subbox(ax, 10.0, 25.65, 6.3, 0.85,
       ['채용공고  (URL 또는 .txt)', '잡코리아 · 사람인 · 원티드 등'],
       ncol=1)

arrow(ax, 10, 25.6, 10, 25.1)

# ══════════════════════════════════════════════════════════
# STEP 0
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 23.5, 17, 1.45,
    'STEP 0  ·  입력 수집', C['step0'], fontsize=11)
subbox(ax, 1.6, 23.55, 7.8, 1.1,
       ['fetcher.py  |  URL → BeautifulSoup 스크래핑',
        '섹션 헤더 자동 탐지 (주요업무 / 자격요건 / 우대사항)'],
       ncol=1)
subbox(ax, 9.6, 23.55, 8.7, 1.1,
       ['parser.py  |  txt / pdf 파싱  ★NEW',
        '문항(### 1.) 형식 자동 감지 → section 모드'],
       ncol=1)
side_label(ax, 0.3, 24.2, 'fetcher.py\nparser.py')
arrow(ax, 10, 23.5, 10, 23.0)

# ══════════════════════════════════════════════════════════
# STEP 1
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 20.3, 17, 2.55,
    'STEP 1  ·  채용공고 키워드 추출  ★NEW', C['step1'],
    fontsize=11, tag='jintea1')

subbox(ax, 1.6, 22.1, 8.2, 0.65,
       ['cleaner.py  |  HTML·이모지 제거 + 기술명 정규화 (파이썬→Python)'],
       ncol=1)
subbox(ax, 9.9, 22.1, 8.4, 0.65,
       ['section_splitter.py  |  ■·◆·▶ 등 다양한 헤더 탐지'],
       ncol=1)

subbox(ax, 1.6, 21.35, 5.3, 0.65,
       ['rule_based.py', 'POS 필터 + Hard/Soft/Action 분류'],
       ncol=1)
subbox(ax, 7.05, 21.35, 5.2, 0.65,
       ['ai_based.py (KeyBERT)', 'KoSBERT 임베딩 기반 중요도 추출'],
       ncol=1)
subbox(ax, 12.4, 21.35, 5.9, 0.65,
       ['hybrid.py  ★', 'Rule후보+KeyBERT점수+SafeNet'],
       ncol=1)

subbox(ax, 1.6, 20.38, 7.8, 0.85,
       ['entity_extractor.py  ★  |  경력 3년↑ / 학사↑ / TOEIC 800↑ 추출',
        '→ rewriter 프롬프트 컨텍스트로 전달'],
       ncol=1)
subbox(ax, 9.6, 20.38, 8.7, 0.85,
       ['evaluator.py  ★  |  Precision / Recall / F1 비교',
        'Rule: F1=0.793  AI: F1=0.802  Hybrid: F1=0.765'],
       ncol=1)

side_label(ax, 0.3, 21.7, 'keyword/\ncleaner\nsection_\nsplitter')
arrow(ax, 10, 20.3, 10, 19.85)

# ══════════════════════════════════════════════════════════
# STEP 2
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 18.4, 17, 1.3,
    'STEP 2  ·  자소서 처리 단위 분리', C['step2'], fontsize=11)
subbox(ax, 1.6, 18.48, 7.8, 1.05,
       ['문항 모드  (jinyong1 방식)',
        '### 1. 질문 감지 → 문항별 answer 리스트'],
       ncol=1)
subbox(ax, 9.6, 18.48, 8.7, 1.05,
       ['문단 모드  (test1 방식)',
        'regex(\\n{2,}) / kss+KoSBERT NLP / auto 자동 선택'],
       ncol=1)
side_label(ax, 0.3, 19.0, 'tokenizer.py')
arrow(ax, 10, 18.4, 10, 17.95)

# ══════════════════════════════════════════════════════════
# STEP 3
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 15.8, 17, 2.0,
    'STEP 3  ·  유사도 분석  (4가지 병렬)', C['step3'], fontsize=11)

subbox(ax, 1.6, 16.65, 3.6, 1.0,
       ['jaccard.py', '키워드 집합 겹침',
        'covered / missing', '가중치: 0.10'],
       ncol=1)
subbox(ax, 5.4, 16.65, 3.6, 1.0,
       ['tfidf.py', 'TF-IDF Cosine',
        '전체 텍스트 유사도', '가중치: 0.25'],
       ncol=1)
subbox(ax, 9.2, 16.65, 3.6, 1.0,
       ['bm25.py', '섹션별 관련도',
        '주요업무/자격/우대', '가중치: 0.30'],
       ncol=1)
subbox(ax, 13.0, 16.65, 4.8, 1.0,
       ['sbert.py  (GPU)', 'KoSBERT 의미 임베딩',
        '전체 배치 추론→캐시', '가중치: 0.35'],
       ncol=1)

subbox(ax, 1.6, 15.88, 16.7, 0.7,
       ['aggregator.py  |  ThreadPoolExecutor 병렬  →  ComparisonResult '
        '(overall_score / section_scores / covered·missing·priority_keywords / fix_urgency)'],
       ncol=1)

side_label(ax, 0.3, 16.8, 'similarity/\naggregator')
arrow(ax, 10, 15.8, 10, 15.35)

# ══════════════════════════════════════════════════════════
# STEP 4
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 13.2, 17, 2.0,
    'STEP 4  ·  피드백 생성 & STAR 재구성', C['step4'], fontsize=11)

subbox(ax, 1.6, 14.3, 7.8, 0.75,
       ['scorer.py  |  ComparisonResult → 피드백 텍스트',
        '★☆☆☆☆ 시각화 / 누락 키워드 / 수정 긴급도'],
       ncol=1)
subbox(ax, 9.6, 14.3, 8.7, 0.75,
       ['ollama_manager.py  |  Ollama 자동 설치·서버 시작·모델 다운로드',
        '첫 호출 시 1회만 실행'],
       ncol=1)

subbox(ax, 1.6, 13.28, 16.7, 0.95,
       ['rewriter.py  |  EXAONE-3.5-7.8B  (RTX 5070 Ti, ~5GB VRAM, 완전 오프라인)',
        '프롬프트: 원문 + 누락키워드 + 관련섹션 + 엔티티조건 + 문항질문(★) + STAR규칙'],
       ncol=1)

side_label(ax, 0.3, 14.2, 'feedback/\nsetup/')
arrow(ax, 10, 13.2, 10, 12.75)

# ══════════════════════════════════════════════════════════
# STEP 5 / 6
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 11.1, 17, 1.5,
    'STEP 5 / 6  ·  CLI 대화형 루프  &  최종 저장', C['step5'], fontsize=11)
subbox(ax, 1.6, 11.18, 7.8, 1.15,
       ['display.py  |  Rich 기반 화면 출력',
        '문단별 원문 / 피드백 / STAR 제안 나란히 표시'],
       ncol=1)
subbox(ax, 9.6, 11.18, 8.7, 1.15,
       ['session.py  |  y(적용) / n(재생성) / r(요구사항) / s(건너뜀)',
        '최종: w(저장) / p(미리보기) / d(취소)'],
       ncol=1)
side_label(ax, 0.3, 11.85, 'cli/')
arrow(ax, 10, 11.1, 10, 10.65)

# ══════════════════════════════════════════════════════════
# STEP 7
# ══════════════════════════════════════════════════════════
box(ax, 1.5, 8.7, 17, 1.8,
    'STEP 7  ·  AI 기여도 분석', C['step7'], fontsize=11)
subbox(ax, 1.6, 9.6, 5.0, 0.85,
       ['edit_distance.py', 'Levenshtein 변화율',
        '0%→원문 / 80%+→전면재작성'],
       ncol=1)
subbox(ax, 6.85, 9.6, 5.2, 0.85,
       ['style_analyzer.py', 'KoGPT2 Perplexity',
        'TTR 어휘다양성 / 문장길이'],
       ncol=1)
subbox(ax, 12.3, 9.6, 5.9, 0.85,
       ['ai_reporter.py', '3관점 가중평균',
        '변화율40%+세션35%+스타일25%'],
       ncol=1)

subbox(ax, 1.6, 8.78, 16.7, 0.75,
       ['AI 기여도 0~100%  →  자필중심(<20%) / 혼합(21~50%) / AI주도(51~80%) / AI작성(81%+)'],
       ncol=1)

side_label(ax, 0.3, 9.6, 'analysis/')
arrow(ax, 10, 8.7, 10, 8.25)

# ══════════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════════
box(ax, 3.5, 7.0, 13, 1.1,
    '출력', C['output'], fontsize=12)
subbox(ax, 3.6, 7.08, 6.0, 0.88,
       ['output_cover.txt', '공고 맞춤 수정 자기소개서'],
       ncol=1)
subbox(ax, 9.8, 7.08, 6.5, 0.88,
       ['AI 기여도 리포트', '문단별 판정 + 전체 % 표시'],
       ncol=1)

# ══════════════════════════════════════════════════════════
# 우측 옵션 설명
# ══════════════════════════════════════════════════════════
opt_items = [
    ('--keyword-method', 'rule / ai / hybrid'),
    ('--split-mode',     'auto / regex / nlp'),
    ('--eval-keywords',  'P/R/F1 비교 평가 출력'),
]
for i, (opt, desc) in enumerate(opt_items):
    y = 22.5 - i * 0.65
    ax.text(18.4, y, opt,  fontsize=7.5, color='#ffd54f',
            fontweight='bold', va='center')
    ax.text(18.4, y - 0.28, desc, fontsize=7,
            color='#90a4ae', va='center')

ax.text(18.4, 23.3, 'CLI 옵션', fontsize=8.5,
        color='white', fontweight='bold', va='center')

# 옵션 박스
opt_rect = FancyBboxPatch((18.1, 20.9), 1.7, 2.6,
                           boxstyle="round,pad=0.1",
                           facecolor='#1a1a2e', edgecolor='#4fc3f7',
                           linewidth=1.0, zorder=2)
ax.add_patch(opt_rect)

# ══════════════════════════════════════════════════════════
# 하단 기술 스택
# ══════════════════════════════════════════════════════════
stack = [
    'kiwipiepy  |  KeyBERT  |  KoSBERT  |  rank-bm25  |  sklearn',
    'sentence-transformers  |  EXAONE-3.5-7.8B (Ollama)  |  KoGPT2  |  kss',
    'Rich  |  Click  |  python-Levenshtein  |  BeautifulSoup4  |  pdfminer',
]
for i, line in enumerate(stack):
    ax.text(10, 6.6 - i * 0.35, line,
            ha='center', va='center', fontsize=7.5, color='#546e7a')

ax.text(10, 6.95, '기술 스택', ha='center', fontsize=8.5,
        color='#78909c', fontweight='bold')

plt.tight_layout(pad=0.3)
plt.savefig('/home/killy/School/NLP/test2/architecture.png',
            dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("저장 완료: architecture.png")
