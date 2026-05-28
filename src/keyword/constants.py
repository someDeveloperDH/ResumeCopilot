# jintea1/part1/constants.py 기반 — Hard/Soft/Action 분류 및 기술명 정규화 사전
# test1에는 없던 키워드 타입 분류와 섹션 가중치 체계를 추가

SECTION_NAMES = ("주요업무", "자격요건", "우대사항")

# 섹션별 가중치 — 주요업무가 가장 중요
WEIGHTS = {
    "주요업무": 1.5,
    "자격요건": 1.0,
    "우대사항": 0.7,
}

# 섹션 헤더 탐지 패턴 (section_splitter.py에서 사용)
SECTION_PATTERNS = {
    "주요업무": [
        r"주요\s*업무", r"담당\s*업무", r"업무\s*내용",
        r"주요\s*직무", r"하는\s*일",
    ],
    "자격요건": [
        r"자격\s*요건", r"지원\s*자격", r"필수\s*요건",
        r"필수\s*사항", r"자격\s*조건", r"요구\s*역량",
    ],
    "우대사항": [
        r"우대\s*사항", r"우대\s*조건", r"우대\s*요건",
        r"preferred\s*qualification",
    ],
}

# 기술명 정규화 (한글/약어 → 표준 표기)
TECH_NORMALIZE = {
    ".NET": ".NET", "닷넷": ".NET",
    "파이썬": "Python", "python": "Python",
    "자바스크립트": "JavaScript", "javascript": "JavaScript", "js": "JavaScript",
    "타입스크립트": "TypeScript", "typescript": "TypeScript", "ts": "TypeScript",
    "리액트": "React", "react.js": "React", "reactjs": "React", "react": "React",
    "노드": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    "뷰": "Vue", "vue.js": "Vue",
    "장고": "Django", "스프링": "Spring", "스프링부트": "Spring Boot",
    "쿠버네티스": "Kubernetes", "k8s": "Kubernetes",
    "도커": "Docker", "깃": "Git", "깃허브": "GitHub", "깃랩": "GitLab",
    "에어플로우": "Airflow", "스파크": "Spark", "카프카": "Kafka",
    "마이SQL": "MySQL", "mysql": "MySQL",
    "postgresql": "PostgreSQL", "포스트그레SQL": "PostgreSQL",
    "몽고디비": "MongoDB",
    "aws": "AWS", "gcp": "GCP", "azure": "Azure",
}

# 불용어 (키워드 추출 시 제외)
STOPWORDS = {
    "등", "기반", "관련", "경험", "능력", "담당", "업무", "가능",
    "이상", "이하", "위한", "통한", "대한", "및", "또는", "그리고",
    "에서", "으로", "에게", "하며", "합니다", "있는", "우리", "회사",
    "서비스", "프로젝트", "경력", "자격", "요건", "사항",
    "TOEIC", "토익", "담당합니다", "담당하다", "보유", "보유자",
}

# 사용자 정의 복합어 사전
USER_DICT_WORDS = [
    "클라우드 네이티브", "머신러닝", "딥러닝", "데이터 파이프라인",
    "마이크로서비스", "자연어처리", "강화학습", "실시간 처리",
    "분산 처리", "추천 시스템", "검색 엔진", "데이터 웨어하우스",
]

# Hard Skills (기술 스킬)
HARD_SKILLS = {
    "Python", "Java", "JavaScript", "TypeScript", "React", "Vue",
    "Node.js", "Django", "Spring", "Spring Boot", "SQL", "MySQL",
    "PostgreSQL", "MongoDB", "Docker", "Kubernetes", "AWS", "GCP",
    "Azure", "Linux", "Git", "GitHub", "GitLab", "Spark", "Kafka",
    "Airflow", "ETL", "CI/CD", "데이터 파이프라인", "머신러닝", "딥러닝", "자연어처리",
}

# Soft Skills (소프트 스킬)
SOFT_SKILLS = {
    "커뮤니케이션", "소통", "협업", "문제 해결", "리더십",
    "주도적", "책임감", "문서화", "코드 리뷰",
}

# Action Verbs (동사형 핵심 키워드)
ACTION_VERBS = {
    "설계", "구축", "개발", "운영", "분석", "개선", "최적화",
    "자동화", "검증", "관리", "도입", "마이그레이션", "리팩토링", "모니터링",
}
