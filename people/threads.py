"""Threads(Meta) 업로드 — 포스트 검증 + Graph API 클라이언트. LLM 없음.

포스트 '작성'은 Claude Code 루틴 세션이 한다(people/ROUTINE.md 참고). 이 모듈은
그 결과물(JSON)을 검증하고 글타래로 올리는 결정론 부분만 담당한다 — 그래서
Anthropic API 키가 필요 없고, 여기 꽂을 것은 THREADS_ACCESS_TOKEN 하나다.

Threads Graph API 흐름(공식 문서 기준):
  1) POST /{user}/threads          → 미디어 컨테이너 생성(id)
  2) GET  /{container}?fields=status → 이미지 처리 대기(FINISHED까지 폴링)
  3) POST /{user}/threads_publish  → 발행(media id)
  답글 체인은 reply_to_id=직전 발행 media id.
"""
from __future__ import annotations

import os
import sys
import time

import httpx

THREADS_BASE = "https://graph.threads.net/v1.0"
POST_CHAR_LIMIT = 500       # Threads 하드 한도
MIN_POSTS, MAX_POSTS = 3, 8  # 글타래 상식선 — 작성 지시는 4~6을 요구하지만 검증은 느슨하게
CONTAINER_POLL_INTERVAL = 3.0
CONTAINER_POLL_TIMEOUT = 60.0


def validate_posts(posts: list) -> list[str]:
    """루틴 세션이 쓴 포스트를 업로드 전에 기계 검증. 위반은 ValueError로 즉시 실패
    — 루틴이 에러 메시지를 보고 해당 포스트를 고쳐 재시도하는 흐름을 전제한다."""
    if not isinstance(posts, list) or not all(isinstance(p, str) for p in posts):
        raise ValueError("posts는 문자열 리스트여야 한다")
    if not (MIN_POSTS <= len(posts) <= MAX_POSTS):
        raise ValueError(f"포스트 개수 {len(posts)} — {MIN_POSTS}~{MAX_POSTS}개여야 한다")
    for i, p in enumerate(posts, 1):
        if not p.strip():
            raise ValueError(f"{i}번 포스트가 비어 있다")
        if len(p) > POST_CHAR_LIMIT:
            raise ValueError(f"{i}번 포스트 {len(p)}자 — {POST_CHAR_LIMIT}자 초과")
    return posts


class ThreadsClient:
    """Meta Threads Graph API 클라이언트. 토큰은 THREADS_ACCESS_TOKEN 환경변수."""

    def __init__(self, access_token: str, user_id: str = "me"):
        self.token = access_token
        self.user_id = user_id
        self._client = httpx.Client(timeout=30.0)

    @classmethod
    def from_env(cls) -> "ThreadsClient | None":
        token = os.environ.get("THREADS_ACCESS_TOKEN")
        if not token:
            return None
        return cls(token, os.environ.get("THREADS_USER_ID", "me"))

    def _post_with_retry(self, url: str, data: dict, retries: int = 4) -> dict:
        last: Exception | None = None
        for attempt in range(retries):
            backoff = 2.0 ** (attempt + 1)
            try:
                resp = self._client.post(url, data={**data, "access_token": self.token})
            except httpx.HTTPError as exc:
                last = exc
                time.sleep(backoff)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(backoff)
                continue
            if resp.status_code >= 400:
                # 4xx는 요청 자체 문제 — 재시도 무의미, 바로 실패시켜 원인 노출.
                raise RuntimeError(f"Threads API {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        assert last is not None
        raise last

    def _wait_container(self, container_id: str) -> None:
        """이미지 컨테이너 처리 대기. 텍스트는 보통 즉시 FINISHED."""
        deadline = time.monotonic() + CONTAINER_POLL_TIMEOUT
        while time.monotonic() < deadline:
            resp = self._client.get(
                f"{THREADS_BASE}/{container_id}",
                params={"fields": "status", "access_token": self.token},
            )
            status = resp.json().get("status")
            if status == "FINISHED":
                return
            if status in ("ERROR", "EXPIRED"):
                raise RuntimeError(f"컨테이너 {container_id} 상태 {status}")
            time.sleep(CONTAINER_POLL_INTERVAL)
        raise RuntimeError(f"컨테이너 {container_id} 처리 타임아웃")

    def _publish_one(self, text: str, image_url: str | None, reply_to: str | None) -> str:
        data: dict = {"text": text}
        if image_url:
            data["media_type"] = "IMAGE"
            data["image_url"] = image_url
        else:
            data["media_type"] = "TEXT"
        if reply_to:
            data["reply_to_id"] = reply_to
        container = self._post_with_retry(f"{THREADS_BASE}/{self.user_id}/threads", data)["id"]
        if image_url:
            self._wait_container(container)
        media = self._post_with_retry(
            f"{THREADS_BASE}/{self.user_id}/threads_publish", {"creation_id": container}
        )["id"]
        return media

    def publish_thread(self, posts: list[str], image_url: str | None = None) -> list[str]:
        """글타래 발행 — 1번 포스트(사진 첨부) 뒤에 나머지를 답글로 체인."""
        ids: list[str] = []
        for i, text in enumerate(posts):
            media = self._publish_one(
                text,
                image_url=image_url if i == 0 else None,
                reply_to=ids[-1] if ids else None,
            )
            ids.append(media)
            print(f"  발행 {i + 1}/{len(posts)}: {media}", file=sys.stderr)
            time.sleep(1.0)  # 연속 발행 예의상 간격
        return ids
