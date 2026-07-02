"""잊힌 인물 Markdown 카드 렌더링 — enrich 팩트 + (있으면) story 서사를 합친다.

서사(stories.json)가 아직 없으면 해당 칸을 프롬프트 팩 안내 플레이스홀더로 채운다
— 수집과 생성을 분리해 두었으므로, 서사만 나중에 채워 재렌더링하면 된다.

사용:
    uv run python -m people.cards
    uv run python -m people.cards --enriched people_out/enriched.json --stories people_out/stories.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt_views(v: float | int | None) -> str:
    return f"{v:,.0f}" if v is not None else "-"


def render_card(e: dict, story: dict | None) -> str:
    name = e.get("name_ko") or e.get("name_en") or e["qid"]
    name_en = e.get("name_en")
    title = f"{name}" + (f" ({name_en})" if name_en and name_en != name else "")

    lines: list[str] = [f"# {title}", ""]
    if story and story.get("headline"):
        lines += [f"> **{story['headline']}**", ""]
    elif e.get("description"):
        lines += [f"> {e['description']}", ""]

    if e.get("image_path"):
        img_rel = Path(e["image_path"]).name
        lines += [f"![{name}](images/{img_rel})", ""]
    elif e.get("image_url"):
        lines += [f"![{name}]({e['image_url']})", ""]

    reason = (e.get("peak_reason") or {}).get("detail")
    lines += [
        "## 기본 정보",
        "",
        f"- **분류**: {e.get('occupation')}",
        f"- **조회수 정점**: {e.get('peak_month')} — 월 {_fmt_views(e.get('peak_views'))}회"
        + (f" ({reason})" if reason else ""),
        f"- **지금**: 월평균 {_fmt_views(e.get('recent_monthly_views'))}회 — 정점 대비 "
        f"{(e.get('recent_monthly_views') or 0) / max(e.get('peak_views') or 1, 1):.1%}",
    ]
    if e.get("awards"):
        awards = ", ".join(
            f"{a['label']}" + (f"({a['month']})" if a.get("month") else "")
            for a in e["awards"][:5]
        )
        lines += [f"- **수상**: {awards}"]
    links = " · ".join(
        f"[{lab}]({e[key]})" for lab, key in (("한국어 위키", "wiki_ko"), ("영어 위키", "wiki_en"))
        if e.get(key)
    )
    if links:
        lines += [f"- **링크**: {links}"]
    lines += [""]

    lines += ["## 왜 화제가 되었나", ""]
    if story and story.get("peak_story"):
        lines += [story["peak_story"], ""]
    else:
        lines += ["_(서사 미생성 — `people_out/prompts/" + e["qid"] + ".txt` 프롬프트로 생성하거나 "
                  "`uv run python -m people.story` 실행)_", ""]
        if e.get("summary"):
            lines += ["**위키 발췌**: " + e["summary"], ""]

    concept = (story or {}).get("concept")
    if concept:
        lines += [f"## 개념 렌즈 — {concept['name']}", "", concept["explanation"], ""]

    if story and story.get("business_insight"):
        lines += ["## 일하는 사람에게", "", story["business_insight"], ""]
    elif story and story.get("modern_inspiration"):
        # 구버전 스키마(영감 칸) 호환 — 재생성 전까지는 이 칸으로 렌더링한다.
        lines += ["## 현대 지식인에게", "", story["modern_inspiration"], ""]
    else:
        lines += ["## 현대 지식인에게", "", "_(서사 미생성)_", ""]

    if story and story.get("today_action"):
        lines += ["## 오늘 바꿔볼 한 가지", "", f"> {story['today_action']}", ""]

    if story and story.get("tags"):
        lines += ["`" + "` `".join(story["tags"]) + "`", ""]
    return "\n".join(lines)


def render_index(enriched: list[dict], stories: dict) -> str:
    lines = ["# 잊힌 인물 카드", "", "| # | 인물 | 분류 | 정점 | 서사 |", "|---|---|---|---|---|"]
    for i, e in enumerate(enriched, 1):
        name = e.get("name_ko") or e.get("name_en") or e["qid"]
        done = "✅" if e["qid"] in stories else "—"
        lines.append(
            f"| {i} | [{name}]({e['qid']}.md) | {e.get('occupation')} "
            f"| {e.get('peak_month')} | {done} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="잊힌 인물 Markdown 카드 렌더링")
    ap.add_argument("--enriched", default="people_out/enriched.json")
    ap.add_argument("--stories", default="people_out/stories.json")
    ap.add_argument("--out-dir", default="people_out/cards")
    args = ap.parse_args()

    enriched = json.load(open(args.enriched))
    stories: dict = {}
    if Path(args.stories).exists():
        stories = json.loads(Path(args.stories).read_text())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 카드가 images/ 상대경로를 참조하므로, enrich가 받아둔 이미지 위치를 맞춰준다.
    img_src = Path(args.enriched).parent / "images"
    img_dst = out_dir / "images"
    if img_src.exists() and not img_dst.exists():
        img_dst.symlink_to(img_src.resolve())

    for e in enriched:
        (out_dir / f"{e['qid']}.md").write_text(render_card(e, stories.get(e["qid"])))
    (out_dir / "README.md").write_text(render_index(enriched, stories))
    n_story = sum(1 for e in enriched if e["qid"] in stories)
    print(f"카드 {len(enriched)}장 렌더링 → {out_dir} (서사 채워짐 {n_story}/{len(enriched)})")


if __name__ == "__main__":
    main()
