"""잊힌 인물 팩트 수집 — 사진·위키 발췌·정점 사유 휴리스틱. LLM 없음(결정론).

discover가 뽑은 랭킹 JSON을 받아, 카드/서사 생성에 필요한 근거 팩트를 모은다:
  - 사진: Wikidata P18 → Commons Special:FilePath 썸네일 URL (+로컬 다운로드).
    자유 라이선스(Commons)라 재사용 안전 — 사진은 이 경로만 쓴다.
  - 발췌: Wikipedia REST summary(ko 우선, en 폴백) → 첫 문단 + 한 줄 설명.
  - 정점 사유: peak_month를 사망일(P570)·수상일(P166/P585)과 대조해
    '부고 정점'/'수상 정점'을 결정론적으로 태깅. 못 맞추면 미상으로 남긴다
    (서사 생성 단계가 발췌를 근거로 채우거나, 사람이 직접 판단).

사용:
    uv run python -m people.enrich --ranking people_out/all100.json --top 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote

import httpx

USER_AGENT = "SIES-people-spike/0.1 (feasibility spike; personal project)"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
COMMONS_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath"
REQUEST_INTERVAL = 0.2

_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True
        )
    return _CLIENT


def _get_with_retry(url: str, params: dict | None = None, retries: int = 4) -> httpx.Response | None:
    """GET + 지수 백오프. 소진하면 None — 한 명의 실패로 전체 실행이 죽지 않게."""
    for attempt in range(retries):
        backoff = 2.0 ** (attempt + 1)
        try:
            resp = _client().get(url, params=params)
        except httpx.HTTPError as exc:
            if attempt == retries - 1:
                print(f"  ! GET 실패({url[:80]}) — 건너뜀: {exc}", file=sys.stderr)
                return None
            time.sleep(backoff)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == retries - 1:
                print(f"  ! HTTP {resp.status_code}({url[:80]}) — 건너뜀", file=sys.stderr)
                return None
            retry_after = resp.headers.get("retry-after", "")
            delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else backoff
            time.sleep(max(delay, backoff))
            continue
        return resp
    return None


# ── Wikidata 클레임 ──────────────────────────────────────────
def fetch_entities(qids: list[str]) -> dict:
    """wbgetentities로 여러 인물의 클레임을 한 번에(최대 50개) 가져온다."""
    out: dict = {}
    for i in range(0, len(qids), 50):
        chunk = qids[i : i + 50]
        resp = _get_with_retry(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims",
                "format": "json",
            },
        )
        if resp is not None:
            out.update(resp.json().get("entities", {}))
        time.sleep(REQUEST_INTERVAL)
    return out


def _claim_time_month(snak: dict) -> str | None:
    """Wikidata time 스냅("+2016-02-19T00:00:00Z")에서 "YYYY-MM"만 뽑는다."""
    try:
        t = snak["datavalue"]["value"]["time"]
    except (KeyError, TypeError):
        return None
    t = t.lstrip("+")
    return t[:7] if len(t) >= 7 and t[4] == "-" else None


def _first_claim(claims: dict, prop: str) -> dict | None:
    arr = claims.get(prop) or []
    return arr[0].get("mainsnak") if arr else None


def extract_facts(claims: dict) -> dict:
    """한 인물의 클레임에서 사진 파일명·사망월·(수상QID, 수상월) 목록을 뽑는다."""
    image = None
    snak = _first_claim(claims, "P18")
    if snak:
        try:
            image = snak["datavalue"]["value"]
        except (KeyError, TypeError):
            image = None

    death_month = None
    snak = _first_claim(claims, "P570")
    if snak:
        death_month = _claim_time_month(snak)

    awards: list[tuple[str, str | None]] = []  # (award_qid, "YYYY-MM"|None)
    for c in claims.get("P166", []) or []:
        try:
            award_qid = c["mainsnak"]["datavalue"]["value"]["id"]
        except (KeyError, TypeError):
            continue
        month = None
        for q in (c.get("qualifiers", {}) or {}).get("P585", []):
            month = _claim_time_month(q)
            if month:
                break
        awards.append((award_qid, month))
    return {"image": image, "death_month": death_month, "awards": awards}


def fetch_labels(qids: list[str]) -> dict[str, str]:
    """수상 QID들의 라벨(ko 우선, en 폴백)을 일괄 조회."""
    labels: dict[str, str] = {}
    unique = sorted(set(qids))
    for i in range(0, len(unique), 50):
        chunk = unique[i : i + 50]
        resp = _get_with_retry(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels",
                "languages": "ko|en",
                "format": "json",
            },
        )
        if resp is None:
            continue
        for qid, ent in resp.json().get("entities", {}).items():
            lab = ent.get("labels", {})
            val = (lab.get("ko") or lab.get("en") or {}).get("value")
            if val:
                labels[qid] = val
        time.sleep(REQUEST_INTERVAL)
    return labels


# ── Wikipedia 발췌 ───────────────────────────────────────────
def _title_from_url(url: str | None) -> str | None:
    if not url or "/wiki/" not in url:
        return None
    return unquote(url.rsplit("/wiki/", 1)[-1])


def fetch_summary(lang: str, title: str) -> dict | None:
    """REST summary — 첫 문단(extract)·한 줄 설명(description)."""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    resp = _get_with_retry(url)
    if resp is None or resp.status_code != 200:
        return None
    d = resp.json()
    return {"extract": d.get("extract"), "description": d.get("description")}


# ── 정점 사유 휴리스틱 ────────────────────────────────────────
def _months_apart(a: str, b: str) -> int:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return abs((ya * 12 + ma) - (yb * 12 + mb))


def peak_reason(peak_month: str | None, death_month: str | None,
                awards: list[tuple[str, str | None]],
                award_labels: dict[str, str]) -> dict:
    """정점 월과 사망·수상 시점의 일치(±1개월)로 사유를 태깅. 못 맞추면 unknown.

    사망이 수상보다 우선 — 같은 달에 둘 다 걸리는 경우는 드물지만, 부고가
    조회수 스파이크의 더 직접적인 원인인 게 보통이다.
    """
    if not peak_month:
        return {"type": "unknown", "detail": None}
    if death_month and _months_apart(peak_month, death_month) <= 1:
        return {"type": "death", "detail": f"사망({death_month})"}
    for award_qid, month in awards:
        if month and _months_apart(peak_month, month) <= 1:
            label = award_labels.get(award_qid, award_qid)
            return {"type": "award", "detail": f"수상: {label}({month})"}
    return {"type": "unknown", "detail": None}


# ── 사진 다운로드 ────────────────────────────────────────────
def photo_url(image_name: str, width: int = 512) -> str:
    return f"{COMMONS_FILEPATH}/{quote(image_name)}?width={width}"


def download_photo(image_name: str, dest: Path) -> bool:
    resp = _get_with_retry(photo_url(image_name))
    if resp is None or resp.status_code != 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return True


# ── 메인 ─────────────────────────────────────────────────────
def enrich(ranked: list[dict], out_dir: Path, download_photos: bool = True) -> list[dict]:
    qids = [r["qid"] for r in ranked]
    print(f"[1/3] Wikidata 클레임 조회 ({len(qids)}명)...")
    entities = fetch_entities(qids)
    facts_by_qid = {
        qid: extract_facts((entities.get(qid) or {}).get("claims", {}) or {}) for qid in qids
    }
    all_award_qids = [a for f in facts_by_qid.values() for a, _ in f["awards"]]
    award_labels = fetch_labels(all_award_qids) if all_award_qids else {}

    print("[2/3] Wikipedia 발췌 조회...")
    enriched: list[dict] = []
    for r in ranked:
        facts = facts_by_qid[r["qid"]]
        summary = None
        lang_used = None
        for lang, url_key in (("ko", "wiki_ko"), ("en", "wiki_en")):
            title = _title_from_url(r.get(url_key))
            if title:
                summary = fetch_summary(lang, title)
                if summary and summary.get("extract"):
                    lang_used = lang
                    break
            time.sleep(REQUEST_INTERVAL)
        reason = peak_reason(r.get("peak_month"), facts["death_month"],
                             facts["awards"], award_labels)
        enriched.append({
            **r,
            "image_name": facts["image"],
            "image_url": photo_url(facts["image"]) if facts["image"] else None,
            "death_month": facts["death_month"],
            "awards": [
                {"label": award_labels.get(a, a), "month": m} for a, m in facts["awards"]
            ],
            "summary_lang": lang_used,
            "summary": (summary or {}).get("extract"),
            "description": (summary or {}).get("description"),
            "peak_reason": reason,
        })

    if download_photos:
        print("[3/3] 사진 다운로드...")
        img_dir = out_dir / "images"
        for e in enriched:
            if not e["image_name"]:
                continue
            dest = img_dir / f"{e['qid']}.jpg"
            if dest.exists():
                e["image_path"] = str(dest)
                continue
            ok = download_photo(e["image_name"], dest)
            e["image_path"] = str(dest) if ok else None
            time.sleep(REQUEST_INTERVAL)
    return enriched


def main() -> None:
    ap = argparse.ArgumentParser(description="잊힌 인물 팩트 수집(사진·발췌·정점 사유)")
    ap.add_argument("--ranking", default="people_out/all100.json")
    ap.add_argument("--top", type=int, default=30, help="점수>0 상위 N명")
    ap.add_argument("--out", default="people_out/enriched.json")
    ap.add_argument("--no-photos", action="store_true")
    args = ap.parse_args()

    ranked = [r for r in json.load(open(args.ranking)) if r["score"] > 0][: args.top]
    out_path = Path(args.out)
    enriched = enrich(ranked, out_path.parent, download_photos=not args.no_photos)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))

    n_photo = sum(1 for e in enriched if e.get("image_name"))
    n_sum = sum(1 for e in enriched if e.get("summary"))
    n_reason = sum(1 for e in enriched if e["peak_reason"]["type"] != "unknown")
    print(f"\n저장: {args.out} — {len(enriched)}명 "
          f"(사진 {n_photo} · 발췌 {n_sum} · 정점사유 태깅 {n_reason})")


if __name__ == "__main__":
    main()
