# SIES — Structure Insight, Engineer Serendipity

망각된-가치 검색기. 활성도와 분리된 **'가치'**를 일차 신호로 삼아
**잊었지만 통찰 있는 과거의 나**를 끌어올리는 개인용 검색 엔진.

전체 설계와 단계별 로드맵은 [`PLAN.md`](PLAN.md) 참고.

## 현재 상태 — Phase 0 (파이프라인 증명)

코퍼스 → 청킹 → 한국어 임베딩 → sqlite-vec 저장 → 의미검색까지 도는 토대.
지금은 **파이프라인이 한국어로 "그럴듯하게" 작동하는지** 증명하는 단계.
역전 재순위(활성도·밴드패스)는 Phase 1, 코퍼스가 커진 뒤 시작.

## 설치

[uv](https://docs.astral.sh/uv/) 사용 (Python 3.12 자동 관리, sudo 불필요):

```bash
uv sync           # 의존성 설치 (torch CPU, sentence-transformers, sqlite-vec ...)
```

## 사용

```bash
# 1) 코퍼스 점검 — 무엇이 로드되고 날짜가 어떻게 잡히는지
uv run python -m sies.corpus corpus

# 2) 인덱싱 — 청킹·임베딩·저장 (기본 모델 KURE, 모델별 임베딩 테이블 분리)
uv run python -m sies.index
uv run python -m sies.index --model bge-m3   # 비교용 베이스라인

# 3) 의미검색 (베이스라인: 순수 유사도, 기본 KURE)
uv run python -m sies.search "할머니에 대한 기억"

# 4) 모델 비교 벤치 — 같은 질의를 여러 모델에 나란히
uv run python -m sies.bench --models kure bge-m3
```

## 구조

```
corpus/          # 개인 글 (git 미추적). 소스별 하위 폴더. 형식: .md/.txt 선호
sies/
  corpus.py      # 로딩·Notion 메타 파싱·타임스탬프 결정(내용→파일명→mtime)
  chunk.py       # 문단 단위 청킹
  embed.py       # 임베딩 백엔드 (기본 kure / bge-m3 / minilm)
  store.py       # sqlite-vec 저장·KNN 검색
  index.py       # 인덱싱 CLI
  search.py      # 검색 CLI (베이스라인)
  bench.py       # Phase 0 모델 비교
sies.db          # sqlite-vec DB (git 미추적)
```

## 원칙 (PLAN에서)

- **AI는 검색 결정에 절대 닿지 않는다.** 검색 점수 = 결정론적 산수(벡터+가중치).
  LLM은 분해·태깅·해설 등 앞뒤 단계에만.
- 각 Phase는 *그 자체로 쓸 물건*으로 끝나게 잘려 있다.
