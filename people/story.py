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
        "concept": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "이 서사를 관통하는 '이름 붙은 개념' 하나(한국어, "
                                   "원어 병기). 경영·전략·조직·경제학·심리학에서 실제로 "
                                   "통용되는 개념을 우선한다.",
                },
                "explanation": {
                    "type": "string",
                    "description": "그 개념의 정의와, 이 인물의 서사가 그 개념의 사례가 "
                                   "되는 이유(한국어, 1~2문단). 개념을 모르는 독자가 "
                                   "이 칸만 읽고도 남에게 설명할 수 있는 수준으로.",
                },
            },
            "required": ["name", "explanation"],
            "additionalProperties": False,
        },
        "business_insight": {
            "type": "string",
            "description": "이 서사가 비즈니스/커리어/조직 운영에 주는 교훈(한국어, "
                           "1~2문단). 일반론 금지 — 이 인물 사례의 특수성에서 출발해, "
                           "의사결정·전략·브랜딩·거버넌스 중 하나로 번역할 것.",
        },
        "today_action": {
            "type": "string",
            "description": "읽은 사람이 '오늘' 실행할 수 있는 행동 또는 스스로에게 "
                           "던질 질문 하나(한국어, 1~3문장). 회의·업무·의사결정 장면이 "
                           "구체적으로 그려져야 한다.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "카드 분류용 태그 3~5개(한국어)",
        },
    },
    "required": ["headline", "peak_story", "concept", "business_insight",
                 "today_action", "tags"],
    "additionalProperties": False,
}

SYSTEM = """\
너는 '잊힌 인물 재발견' 뉴스레터의 에디터다. 독자는 출근길 지하철에서 폰으로 이걸
읽는 지식노동자·창업가 — 3분 안에 한 번 피식하고, 개념 하나 건지고, 오늘 써먹을 것
하나 들고 내려야 한다.

톤 — 가장 중요하다:
- 재치 있는 해요체. 친한 선배가 커피 마시며 들려주는 이야기처럼.
- 첫 문장은 훅. 스크롤을 멈추게 만들 것 — 질문, 반전, 의외의 사실, 이름의 아이러니,
  뭐든 좋다. "OO는 ~였다"로 시작하는 백과사전 문장 금지.
- 유머 환영. 이름·상황·데이터의 아이러니를 그냥 지나치지 말 것. 단 조롱이 아니라
  위트여야 하고, 웃기려고 팩트를 비틀면 안 된다.
- 문장은 짧게, 리듬 있게. 괄호 드립·구어체 감탄 적당히 섞기.
- 보고서 문체 금지: "~로 전락한다", "~해야 할 것이다", "~인 것이다" 연발 금지.
- 어렵게 쓰면 실패다. 개념 설명도 예능 자막처럼 쉽게.

내용 원칙:
- 인물 사실은 제공된 팩트(발췌·정점 사유·수상·데이터) + 널리 알려진 확실한 사실만.
  지어내지 않는다. 애매하면 "~였다고 해요" 대신 아예 빼거나 모른다고 솔직하게.
  단, 개념(concept)과 비즈니스 해석은 네 지식으로 채우는 칸이다.
- 위인전 어투 금지. 인물의 모순·논쟁·흑역사도 숨기지 않는다(있다면 그게 더 재밌다).
- concept: 서사를 관통하는 이름 붙은 개념 하나. 실제로 통용되는 개념(권한과 권력의
  분리, 선점자의 저주, 주인-대리인 문제, 서사 자본 등)을 우선하고, 없으면 새로
  명명하되 방금 지었다고 밝힌다. 설명은 이 칸만 읽고 회식 자리에서 아는 척할 수
  있는 수준으로.
- business_insight: "겸손하라" 류 덕담 금지. 이 인물이 아니면 나올 수 없는 교훈을
  의사결정·전략·브랜딩·거버넌스의 언어로 — 근데 재밌게.
- today_action: 오늘 실행 가능한 한 가지. 구체적 장면(오늘 회의에서, 다음 채용에서)
  + 동사. 가볍게 툭 건네듯, 근데 해보면 뜨끔하게.
- 한국어. 이름 첫 등장 시 원어 병기.
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


def generate_one(client, e: dict) -> dict | None:
    """한 인물의 서사를 생성한다. 거부/오류 시 None (호출자가 건너뜀 판단)."""
    import anthropic

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
        print(f"  ! API 오류({exc.status_code}): {exc.message}", file=sys.stderr)
        return None
    if resp.stop_reason == "refusal":
        print("  ! 거부됨", file=sys.stderr)
        return None
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


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
        story = generate_one(client, e)
        if story is None:
            continue
        stories[qid] = story
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
