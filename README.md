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

# 3) 의미검색 — 베이스라인(순수 유사도) 또는 재순위(--invert, 기본 도전자 relz_fv=E)
uv run python -m sies.search "할머니에 대한 기억"
uv run python -m sies.search "관성에 대하여" --invert            # 잊힌 글 끌어올리기
uv run python -m sies.search "관성에 대하여" --invert --ranker gated   # 구 랭커
uv run python -m sies.search "관성에 대하여" --invert --lam 1.5  # 망각 보너스 강하게

# 4) A/B 하니스 — 베이스라인 vs 도전자(기본 relz_fv), 블라인드 판정 + 로그 (킬 테스트)
uv run python -m sies.ab "관성에 대하여" --judge
uv run python -m sies.stats                                     # 누적 적중률 집계
uv run python -m sies.replay                                    # 오프라인 점수식 재실행 비교

# 5) 모델 비교 벤치 — 같은 질의를 여러 모델에 나란히
uv run python -m sies.bench --models kure bge-m3
```

## 테스트

임베딩 모델 없이 도는 단위 테스트 (corpus 파싱·청킹·sqlite-vec 왕복·역전 산수·A/B 집계):

```bash
uv run pytest
```

## 구조

```
corpus/          # 개인 글 (git 미추적). 소스별 하위 폴더. 형식: .md/.txt 선호
sies/
  corpus.py      # 로딩(.md/.txt/.pdf/.hwp)·Notion 메타 파싱·타임스탬프(내용→파일명→mtime)
  chunk.py       # 문단 단위 청킹
  embed.py       # 임베딩 백엔드 (기본 kure / bge-m3 / minilm)
  store.py       # sqlite-vec 저장·KNN 검색
  retrieve.py    # 질의 임베딩 + 전체 후보 풀 조회 (정규화·밴드패스용)
  rank.py        # ★ 재순위 — 제품의 뇌, ML 없음. 기본 relz_fv(E: rel_z + λ·fv)
                 #   rel_z=질의내 유사도 z정규화 · fv=비대칭 망각밴드 (구: inverted/gated)
  index.py       # 인덱싱 CLI
  search.py      # 검색 CLI (베이스라인 / --invert 역전)
  ab.py          # A/B 하니스 — 블라인드 판정 + JSONL 로그
  stats.py       # A/B 로그 적중률 집계 (킬 테스트 판정)
  bench.py       # 모델 비교
sies.db          # sqlite-vec DB (git 미추적)
search_log.jsonl # A/B 판정 로그 (git 미추적)
```

## 원칙 (PLAN에서)

- **AI는 검색 결정에 절대 닿지 않는다.** 검색 점수 = 결정론적 산수(벡터+가중치).
  LLM은 분해·태깅·해설 등 앞뒤 단계에만.
- 각 Phase는 *그 자체로 쓸 물건*으로 끝나게 잘려 있다.
