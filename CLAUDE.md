# SIES — 에이전트용 지도

잊힌 글을 끌어올리는 개인용 의미검색 엔진. 이 파일이 레포의 압축 지도다 —
작업 전에 README/PLAN/모듈 전체를 다시 읽지 말고, 필요한 파일만 골라 읽어라.

## 불변 원칙 (어기면 안 됨)

1. 검색 점수는 순수 결정론적 산수. **LLM·ML은 검색 결정에 절대 닿지 않는다.**
2. 활성도 낮다는 이유로 삭제 금지. 삭제는 오직 `가치 낮음 ∧ 중복`.
3. 킬 테스트(A/B 로그로 역전 vs 베이스라인 적중률)가 프로젝트의 생사를 정한다.
4. 상수·점수식의 단일 소스는 `rank.py` (`LOW_ACTIVITY`, `gated_score` 등) — 사본 만들지 마라.

## 모듈 맵 (sies/, 데이터 흐름 순)

| 파일 | 역할 | 언제 읽나 |
|---|---|---|
| corpus.py | .md/.txt/.pdf/.hwp/.docx 로딩, Notion 메타·타임스탬프 파싱 | 로딩/날짜 버그 |
| normalize.py | PDF 띄어쓰기 교정(Kiwi 교집합), 앞머리 마커 제거 | 텍스트 정제 |
| chunk.py | 문단 단위 청킹 (`MIN_CHARS=40` 미만은 병합) | 청킹 |
| embed.py | sentence-transformers 레지스트리 (kure★/bge-m3/minilm) | 모델 추가 |
| dedup.py | 인덱싱 시 임베딩 코사인 ≥0.97 근접중복 제거 | 중복 |
| store.py | sqlite-vec 저장·KNN. 모델별 vec 테이블, 공용 chunks 테이블 | DB |
| retrieve.py | `query_pool(db, model, 질의)` — 질의→임베딩→전체 후보 풀 | CLI 진입로 |
| **rank.py** | ★ 제품의 뇌. 활성도(시간·볼륨 min), 소프트 밴드패스, `rank_inverted`/`rank_gated`/`rank_baseline` | 점수 로직 |
| index.py / search.py | 인덱싱·검색 CLI | CLI |
| ab.py | A/B 하니스: 블라인드 판정 → search_log.jsonl | 킬 테스트 |
| stats.py | 로그 집계(매크로/마이크로/I-only/잊힘도) | 킬 테스트 |
| replay.py | 기존 로그로 점수식 오프라인 비교 | 점수식 실험 |
| bench.py | Phase 0 모델 나란히 비교 (유물) | 거의 안 읽음 |
| util.py | preview/fmt_pct/read_log — CLI 공용 소품 | 거의 안 읽음 |

테스트는 `tests/test_<모듈>.py` 1:1 대응, 임베딩 모델 없이 돈다.
(예외: `test_extract.py`는 corpus.py의 추출 경로 — PDF/HWP 분기·마커 제거 — 를 검증.)

## 모듈 맵 (people/ — 잊힌 인물 발굴→쓰레드, 같은 산수·다른 대상)

`점수 = 정점_저명도 × (1−활성도) × 밴드패스` — `sies.rank`의 산수 재사용.
외부 API(wikidata/wikimedia) 필요, 테스트는 오프라인. 상세: `people/README.md`.

| 파일 | 역할 | 언제 읽나 |
|---|---|---|
| wikidata.py | SPARQL 후보 풀 (직업 QID, sitelink 문턱) | 후보 수집 |
| pageviews.py | 위키 월간 조회수 시계열 + 파일 캐시 | 활성도 신호 |
| score.py / discover.py | 점수 계산 · 발굴 CLI | 점수/CLI |
| trends.py | Google Trends 참고 신호 (점수 미반영) | 거의 안 읽음 |
| enrich.py | 팩트 수집: Commons 사진·위키 발췌·정점 사유 | 카드 재료 |
| **story.py** | ★ 레포 유일 LLM 지점 — 서사 작성 (랭킹 아님, 원칙 1 유지) | 서사 |
| cards.py / threads.py | Markdown 카드 렌더 · 포스트 검증+Graph API 업로드 | 발행 |
| daily.py | 일간 루틴 CLI: prepare/publish. 루틴 프롬프트는 `people/ROUTINE.md` | 루틴 |

## 명령어

```bash
uv sync                                 # 설치 (torch CPU)
uv run pytest                           # 테스트 (모델 다운로드 없음, 빠름)
uv run python -m sies.index             # 인덱싱 (corpus/ → sies.db)
uv run python -m sies.search "질의" --invert
uv run python -m sies.ab "질의" --judge # A/B + 블라인드 판정 → 로그
uv run python -m sies.stats             # 킬 테스트 집계
```

## 문서 (필요할 때만)

- `PLAN.md` — Phase 로드맵·설계 철학. 방향성 결정 때만.
- `docs/genre-aware-bandpass.md` — 보류된 Phase 3 설계. 평소엔 읽지 마라.
- `corpus/`, `sies.db`, `search_log.jsonl` — 개인 데이터, git 미추적. 절대 커밋 금지.

## 작업 규칙

- 수정 후 `uv run pytest` 필수. 공개 API(테스트가 import하는 이름)는 유지.
- 커밋 메시지는 한국어, `영역: 요약` 형식 (예: `랭커: gated B 추가`).
- 주석·문서는 한국어. 기존 밀도를 따라라 — 설계 이유가 docstring에 산다.
