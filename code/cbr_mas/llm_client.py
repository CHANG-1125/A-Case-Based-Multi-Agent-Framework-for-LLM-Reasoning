"""OpenAI-compatible chat client (minimal JSON via httpx for proxy compatibility)."""

from __future__ import annotations

import os
import re
import time

import httpx

from cbr_mas.config import LLMConfig


class ChatLLM:
    """
    Uses a small JSON body that matches the official Chat Completions API.
    The openai>=2.x SDK can send fields that some gateways (e.g. UniAPI) mishandle,
    which triggers "could not parse the JSON body" from the upstream provider.
    """

    def __init__(self, cfg: LLMConfig):
        self._api_key = cfg.api_key
        default_base = "https://api.openai.com/v1"
        self._base = (cfg.base_url or default_base).rstrip("/")
        self._model = cfg.model
        self._read_timeout_s = float(
            os.environ.get("OPENAI_READ_TIMEOUT")
            or os.environ.get("OPENAI_TIMEOUT")
            or "600"
        )
        connect_s = float(os.environ.get("OPENAI_CONNECT_TIMEOUT", "60"))
        self._timeout = httpx.Timeout(
            connect=connect_s,
            read=self._read_timeout_s,
            write=120.0,
            pool=60.0,
        )
        self._max_retries = max(1, int(os.environ.get("LLM_MAX_RETRIES", "3")))
        self._single_user = os.environ.get("LLM_SINGLE_USER_MESSAGE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        # httpx 默认会读 HTTP_PROXY / HTTPS_PROXY；不少本机或公司代理在 CONNECT+TLS 上会握手失败
        # （SSL: UNEXPECTED_EOF_WHILE_READING）。调 API 时默认直连；若必须走代理再设 HTTPX_TRUST_ENV=1
        self._trust_env = os.environ.get("HTTPX_TRUST_ENV", "").lower() in (
            "1",
            "true",
            "yes",
        )

    def complete(self, system: str, user: str, temperature: float) -> tuple[str, int | None]:
        sys = self._sanitize_text(system)
        usr = self._sanitize_text(user)
        payload = self._build_payload(sys, usr, temperature, single_user=self._single_user)
        url = f"{self._base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        r = self._post_with_network_retry(url, headers, payload)

        if r.status_code >= 400:
            msg = self._extract_error_message(r)
            # Some gateways intermittently fail to parse long/multi-message JSON.
            # Retry once with merged single-user message and further sanitized text.
            if r.status_code == 400 and "parse the JSON body" in str(msg):
                payload_fallback = self._build_payload(
                    self._sanitize_text(sys, hard=True),
                    self._sanitize_text(usr, hard=True),
                    temperature,
                    single_user=True,
                )
                r2 = self._post_with_network_retry(url, headers, payload_fallback)
                if r2.status_code < 400:
                    return self._parse_success(r2)
                msg2 = self._extract_error_message(r2)
                raise RuntimeError(f"LLM HTTP {r2.status_code}: {msg2}") from None
            raise RuntimeError(f"LLM HTTP {r.status_code}: {msg}") from None

        return self._parse_success(r)

    def _build_payload(self, system: str, user: str, temperature: float, *, single_user: bool) -> dict:
        if single_user:
            merged = f"{system}\n\n---\n\n{user}"
            messages = [{"role": "user", "content": merged}]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        return {
            "model": self._model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False,
        }

    def _post_with_network_retry(self, url: str, headers: dict, payload: dict) -> httpx.Response:
        r: httpx.Response | None = None
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout, trust_env=self._trust_env) as client:
                    r = client.post(url, headers=headers, json=payload)
                break
            except (
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                httpx.ConnectError,
            ) as e:
                if attempt >= self._max_retries - 1:
                    raise RuntimeError(
                        f"LLM 请求在 {self._max_retries} 次尝试后仍失败（网络/超时）。"
                        f"可调高 OPENAI_READ_TIMEOUT（当前 read≈{self._read_timeout_s:.0f}s）或检查网络。"
                    ) from e
                time.sleep(min(2.0**attempt, 30.0))
        assert r is not None
        return r

    def _extract_error_message(self, r: httpx.Response) -> str:
        try:
            body = r.json()
            err = body.get("error", body)
            return str(err.get("message", err) if isinstance(err, dict) else err)
        except Exception:
            return r.text[:500]

    def _parse_success(self, r: httpx.Response) -> tuple[str, int | None]:
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices: {data!r}")
        msg_out = choices[0].get("message") or {}
        text = (msg_out.get("content") or "").strip()
        usage = data.get("usage") or {}
        tokens = usage.get("total_tokens")
        return text, tokens if isinstance(tokens, int) else None

    @staticmethod
    def _sanitize_text(s: str, hard: bool = False) -> str:
        out = s.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
        # Remove invalid surrogate code points that may break downstream JSON parsers.
        out = re.sub(r"[\ud800-\udfff]", " ", out)
        if hard:
            # Keep printable + newline/tab to avoid odd control characters from model text.
            out = re.sub(r"[^\x09\x0A\x20-\x7E]", " ", out)
            out = re.sub(r"[ ]{2,}", " ", out)
        return out
