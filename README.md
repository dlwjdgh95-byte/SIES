# SIES — Structure Insight, Engineer Serendipity

망각된-가치 검색기. 활성도와 분리된 **'가치'**를 일차 신호로 삼아
**잊었지만 통찰 있는 과거의 나**를 끌어올리는 개인용 검색 엔진.

전체 설계와 단계별 로드맵은 [`PLAN.md`](PLAN.md) 참고.

## 현재 상태 — Phase 1 (역전 재순위)

코퍼스 → 청킹 → 한국어 임베딩 → sqlite-vec 저장 → 의미검색 위에,
**역전 재순위**(`점수 = 관련성 × (1−활성도) × 밴드패스`)를 얹어
*적당히 관련되면서 오래 잊힌* 글을 끌어올린다. 검색 점수는 순수 결정론적 산수 —
**LLM은 검색 결정에 절대 닿지 않는다(원칙 2).**

킬 테스트: A/B 하니스로 베이스라인 vs 역전을 블라인드 판정해 로그에 쌓고,
역전의 적중률이 베이스라인을 이기는지 본다(`sies.stats`).

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

# 3) 의미검색 — 베이스라인(순수 유사도) 또는 역전 재순위
uv run python -m sies.search "할머니에 대한 기억"
uv run python -m sies.search "관성에 대하여" --invert            # 잊힌 글 끌어올리기
uv run python -m sies.search "관성에 대하여" --invert --half-life 180

# 4) A/B 하니스 — 베이스라인 vs 역전, 블라인드 판정 + 로그 (킬 테스트)
uv run python -m sies.ab "관성에 대하여" --judge
uv run python -m sies.stats                                     # 누적 적중률 집계

# 5) 모델 비교 벤치 — 같은 질의를 여러 모델에 나란히
uv run python -m sies.bench --models kure bge-m3
```

## 테스트

임베딩 모델 없이 도는 단위 테스트 (corpus 파싱·청킹·sqlite-vec 왕복·역전 산수·A/B 집계):

```bash
uv run pytest
```

## 구조·원칙

모듈 지도와 작업 규칙은 [`CLAUDE.md`](CLAUDE.md), 설계 원칙과 Phase 로드맵은 [`PLAN.md`](PLAN.md) 참고.
`corpus/`(개인 글)·`sies.db`·`search_log.jsonl`은 git 미추적 개인 데이터.
