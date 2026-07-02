"""잊힌 인물 서사 생성 — 유일하게 LLM을 쓰는 단계(콘텐츠 제작, 랭킹 아님).

SIES 원칙 2(랭킹은 결정론적 산수)는 그대로다: 누구를 고를지는 discover/score가
LLM 없이 정하고, 여기서는 이미 골라진 인물의 '이야기'만 쓴다. enrich가 모은
팩트(위키 발췌·정점 사유·수상 등)를 근거로 넣어 환각을 누르고, 스키마 고정
JSON(structured outputs)으로 받아 카드 템플릿에 바로 꽂는다.

ANTHROPIC_API_KEY가 없으면: 인물별 프롬프트 팩(people_out/prompts/*.txt)을 떨궈
사람이 검토하거나 다른 곳에서 실행할 수 있게 한다 — 같은 프롬프트, 같은 스키마.

사용:
    uv run python -m people.story --enriched people_out/enriched.json
    uv run python -m people.story --limit 3          # 소량 검증
    uv run python -m people.story --prompts-only     # 프롬프트 팩만 생성
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MODEL = "claude-opus-4-8"

STORY_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {
            "type": "string",
            "description": "인물을 한 문장으로 소개하는 헤드라인(한국어, 40자 이내)",
        },
        "peak_story": {
            "type": "string",
            "description": "이 인물이 왜 화제가 되었는지의 서사(한국어, 2~3문단). "
                           "정점 시점에 무슨 일이 있었고 왜 세상이 주목했는지.",
        },
        "modern_inspiration": {
            "type": "string",
            "description": "현대 지식인에게 어떤 영감을 주는지(한국어, 1~2문단). "
                           "설교조가 아니라 구체적 긴장/질문 중심.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "카드 분류용 태그 3~5개(한국어)",
        },
    },
    "required": ["headline", "peak_story", "modern_inspiration", "tags"],
    "additionalProperties": False,
}

SYSTEM = """\
너는 '잊힌 인물 재발견' 카드의 작가다. 아래 원칙을 지킨다.

- 제공된 팩트(발췌·정점 사유·수상·데이터)만 근거로 쓴다. 팩트에 없는 사건·숫자·
  인용을 지어내지 않는다. 확실하지 않으면 단정 대신 신중한 표현을 쓴다.
- 위인전 어투 금지. 인물의 모순과 논쟁도 숨기지 않는다(발췌에 있다면).
- '현대적 영감'은 교훈 설교가 아니라, 이 인물의 삶이 지금의 지식인에게 던지는
  구체적인 질문이나 긴장을 짚는다.
- 한국어로 쓴다. 이름 첫 등장 시 원어 병기.
"""


def build_prompt(e: dict) -> str:
    """enrich가 모은 한 인물의 팩트를 근거 블록으로 조립한다."""
    name = e.get("name_ko") or e.get("name_en") or e["qid"]
    awards = ", ".join(
        f"{a['label']}({a['month'] or '시점미상'})" for a in e.get("awards", [])
    ) or "없음"
    reason = e.get("peak_reason") or {}
    reason_txt = {
        "death": f"사망 직후 조회수 정점 — {reason.get('detail')}",
        "award": f"수상 시점에 조회수 정점 — {reason.get('detail')}",
    }.get(reason.get("type"), "미상(정점 월의 뉴스 이벤트를 발췌에서 추정하되, 근거 없으면 단정하지 말 것)")

    return f"""\
다음 인물의 '잊힌 인물 재발견' 카드 텍스트를 작성하라.

<인물 팩트>
이름: {name} ({e.get('name_en') or '-'})
직업(위키데이터 분류): {e.get('occupation')}
한 줄 설명: {e.get('description') or '-'}
위키백과 발췌({e.get('summary_lang') or '-'}):
{e.get('summary') or '(발췌 없음 — 아는 사실이 확실한 경우에만 일반 상식 수준으로 서술)'}

수상: {awards}
사망: {e.get('death_month') or '생존 또는 미상'}
</인물 팩트>

<주목도 데이터>
위키백과 조회수 정점: {e.get('peak_month')} (월 {e.get('peak_views'):,}회)
정점 사유(휴리스틱): {reason_txt}
최근 12개월 월평균: {e.get('recent_monthly_views', 0):,.0f}회 — 정점 대비 크게 잊힘
</주목도 데이터>
"""


def write_prompt_pack(enriched: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_SCHEMA.json").write_text(json.dumps(STORY_SCHEMA, ensure_ascii=False, indent=2))
    (out_dir / "_SYSTEM.txt").write_text(SYSTEM)
    for e in enriched:
        (out_dir / f"{e['qid']}.txt").write_text(build_prompt(e))
    print(f"프롬프트 팩 저장: {out_dir} ({len(enriched)}명 + _SYSTEM/_SCHEMA)")


def generate_stories(enriched: list[dict], out_path: Path) -> None:
    import anthropic

    client = anthropic.Anthropic()
    # 이미 생성된 인물은 건너뛴다 — 중단 후 재실행해도 비용이 이중으로 들지 않게.
    stories: dict[str, dict] = {}
    if out_path.exists():
        stories = json.loads(out_path.read_text())

    for i, e in enumerate(enriched, 1):
        qid = e["qid"]
        name = e.get("name_ko") or e.get("name_en") or qid
        if qid in stories:
            print(f"[{i}/{len(enriched)}] {name} — 캐시됨, 건너뜀")
            continue
        print(f"[{i}/{len(enriched)}] {name} 생성 중...")
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=[{"type": "text", "text": SYSTEM,
                          "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": build_prompt(e)}],
                output_config={"format": {"type": "json_schema", "schema": STORY_SCHEMA}},
            )
        except anthropic.APIStatusError as exc:
            print(f"  ! API 오류({exc.status_code}) — 건너뜀: {exc.message}", file=sys.stderr)
            continue
        if resp.stop_reason == "refusal":
            print(f"  ! 거부됨 — 건너뜀", file=sys.stderr)
            continue
        text = next(b.text for b in resp.content if b.type == "text")
        stories[qid] = json.loads(text)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stories, ensure_ascii=False, indent=2))

    print(f"\n저장: {out_path} ({len(stories)}명)")


def main() -> None:
    ap = argparse.ArgumentParser(description="잊힌 인물 서사 생성(Claude API 또는 프롬프트 팩)")
    ap.add_argument("--enriched", default="people_out/enriched.json")
    ap.add_argument("--out", default="people_out/stories.json")
    ap.add_argument("--limit", type=int, help="상위 N명만(소량 검증용)")
    ap.add_argument("--prompts-only", action="store_true",
                     help="API를 부르지 않고 프롬프트 팩만 생성")
    args = ap.parse_args()

    enriched = json.load(open(args.enriched))
    if args.limit:
        enriched = enriched[: args.limit]

    have_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if args.prompts_only or not have_key:
        if not args.prompts_only:
            print("ANTHROPIC_API_KEY 없음 — 프롬프트 팩으로 대체합니다.", file=sys.stderr)
        write_prompt_pack(enriched, Path("people_out/prompts"))
        return

    generate_stories(enriched, Path(args.out))


if __name__ == "__main__":
    main()
