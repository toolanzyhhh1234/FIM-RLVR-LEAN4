import json
import os
import re
import socket
from typing import Any, Dict, Tuple
from urllib import error, request


class AxleLeanVerifier:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        environment: str | None = None,
        *,
        no_sorries: bool = False,
        timeout_seconds: float = 60.0,
    ):
        resolved_api_key = api_key or os.environ.get("AXLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("AXLE_API_KEY environment variable or axle_api_key config is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        self.api_key = resolved_api_key
        self.base_url = (base_url or os.environ.get("AXLE_API_URL") or "https://axle.axiommath.ai/api/v1").rstrip("/")
        self.environment = environment or os.environ.get("AXLE_ENVIRONMENT") or "lean-4.28.0"
        self.no_sorries = bool(no_sorries)
        self.timeout_seconds = float(timeout_seconds)

    def _contains_sorry_or_admit(self, code: str) -> bool:
        return bool(re.search(r"\bsorry\b", code) or re.search(r"\badmit\b", code))

    def _serialize_output(self, payload: Dict[str, Any]) -> str:
        message_parts: list[str] = []
        for key in ("message", "detail", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                message_parts.append(value.strip())

        lean_messages = payload.get("lean_messages")
        if isinstance(lean_messages, dict):
            for category in ("errors", "warnings", "infos"):
                items = lean_messages.get(category)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str) and item.strip():
                            message_parts.append(item.strip())
                        elif isinstance(item, dict):
                            text = item.get("message") or item.get("text") or item.get("data")
                            if isinstance(text, str) and text.strip():
                                message_parts.append(text.strip())

        if message_parts:
            return "\n".join(message_parts)
        return json.dumps(payload, ensure_ascii=False)

    def verify(self, full_code: str) -> Tuple[bool, str]:
        if self.no_sorries and self._contains_sorry_or_admit(full_code):
            return False, "Verification failed: code contains 'sorry' or 'admit' (no_sorries=True)"

        payload = {
            "content": full_code,
            "environment": self.environment,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/check",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed_error = json.loads(response_body)
            except json.JSONDecodeError:
                parsed_error = None
            if isinstance(parsed_error, dict):
                return False, self._serialize_output(parsed_error)
            return False, response_body or f"HTTP {exc.code}: {exc.reason}"
        except (error.URLError, socket.timeout, TimeoutError) as exc:
            return False, f"Verification request failed: {exc}"

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            return False, response_body

        if not isinstance(parsed, dict):
            return False, response_body

        success = bool(parsed.get("okay"))
        return success, self._serialize_output(parsed)
