# 일간 쓰레드 업로드 루틴

Claude Code 루틴(매일 1회)에 아래 프롬프트를 등록한다. Anthropic API 키 불필요 —
서사·포스트는 루틴 세션 안에서 작성되고, 스크립트는 큐·검증·업로드만 한다.

## 루틴 프롬프트 (복사해서 등록)

```
SIES 저장소에서 오늘의 잊힌 인물을 쓰레드에 올려줘.

1. `uv run python -m people.daily prepare` 실행.
2. 출력 지시대로:
   - 서사가 없으면 Opus 서브에이전트로 people_out/prompts/_SYSTEM.txt 원칙에 따라
     서사를 작성해 stories.json에 저장.
   - 쓰레드 포스트 4~6개(각 450자 이하, (n/총) 넘버링, 1번=훅, 마지막=오늘의 액션+위키 링크)를
     작성해 people_out/threads/{QID}.json 에 {"posts":[...]} 형식으로 저장.
     톤은 카드와 동일: 재치 있는 해요체, 팩트에 없는 사실 추가 금지.
3. `uv run python -m people.daily publish` 실행. 검증 에러가 나면 지적된 포스트를
   고쳐 다시 실행. 발행 결과(인물·루트 포스트 id)를 보고해줘.
```

## 사전 준비 (1회)

1. **Threads 토큰**: Meta 개발자 콘솔에서 앱 생성 → Threads API 활성화 →
   `threads_basic`, `threads_content_publish` 권한으로 장기 액세스 토큰 발급.
2. 환경변수 등록(루틴 환경 설정 또는 셸 프로필):
   ```
   export THREADS_ACCESS_TOKEN=...
   export THREADS_USER_ID=me        # 기본값 me, 보통 생략 가능
   ```
   토큰이 없으면 publish는 자동으로 dry-run(포스트 출력만)이 되어 안전하다.
3. 큐 채우기: `people_out/enriched.json`이 큐다. 소진되면
   `uv run python -m people.enrich --ranking people_out/all100.json --top 60` 처럼
   범위를 늘려 재실행.

## 동작 요약

- 상태: `people_out/publish_state.json` — posted 기록, pending(작성 중) 표시.
  실발행 성공 시에만 posted로 확정되므로 중간 실패는 다음 루틴이 이어서 처리한다.
- 사진: enrich가 수집한 Wikimedia Commons 공개 URL을 1번 포스트에 첨부(자유 라이선스).
- 검증: 포스트 3~8개·각 500자 이하·빈 포스트 금지(`people/threads.py`)를 통과해야 발행.
