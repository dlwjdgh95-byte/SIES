# people/ — 잊힌 인물 발굴 (feasibility spike)

SIES의 스코어링 산수(`sies.rank.activity` / `bandpass_weight`)를 그대로 재사용해,
**공개 인물** 중 "한때 확실히 유명했지만 지금은 조용한" 사람을 끌어올린다.

    점수 = 정점_저명도(풀 내 상대) × (1 − 활성도) × 밴드패스가중치

- 후보 풀: Wikidata SPARQL (직업 QID로 좁힘, en/ko 위키 sitelink ≥ 문턱값)
- 정점/활성도: Wikimedia Pageviews (월간 조회수 시계열, 2015-07~)
- Google Trends: 참고용 부가 신호(`--with-trends`), 점수엔 미반영

## 실행

```bash
uv sync
uv run python -m people.discover --occupation politician --limit 20
uv run python -m people.discover --occupation all --limit 200 --out people_out/result.json
```

플래그: `--occupation {politician,entrepreneur,actor,musician,influencer,all}`,
`--limit`, `--min-sitelinks`(기본 15), `--refresh`(pageviews 캐시 무시), `--with-trends`.

## ⚠️ 네트워크 egress 요구사항

CLI는 외부 API를 호출하므로 **아웃바운드 HTTPS가 열린 환경**에서만 돈다.
`uv run pytest`(스코어 산수 단위테스트)는 네트워크 없이 돌지만, `discover`는 아래
호스트로 나갈 수 있어야 한다:

| 호스트 | 용도 | 필수 |
|---|---|---|
| `query.wikidata.org` | SPARQL 후보 풀 | ✅ |
| `wikimedia.org` | Pageviews REST API | ✅ |
| `trends.google.com`, `www.google.com` | Google Trends(pytrends) | `--with-trends`일 때만 |

**Claude Code on the web에서 돌릴 경우:** 이 도메인들은 기본 네트워크 정책의
패키지 레지스트리 허용목록(pypi/npm 등)에 없어 403(정책 차단)으로 막힌다.
환경을 만들 때 일반 인터넷 접근을 허용하는 정책(또는 위 호스트를 포함한 커스텀
허용목록)을 선택해야 한다. 자세한 정책 옵션:
https://code.claude.com/docs/en/claude-code-on-the-web

**로컬에서 돌릴 경우:** 별도 설정 없이 위 명령 그대로 실행.
