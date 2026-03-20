import io
import json
from unittest.mock import patch
from urllib import error

import pytest

from tinker_integration.axle_verifier import AxleLeanVerifier


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_init_requires_api_key():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="AXLE_API_KEY"):
            AxleLeanVerifier()


def test_verify_success():
    verifier = AxleLeanVerifier(api_key="axle-key")

    with patch("tinker_integration.axle_verifier.request.urlopen", return_value=_MockResponse({"okay": True, "lean_messages": {"errors": []}})):
        success, output = verifier.verify("import Mathlib\ntheorem t : True := by trivial")

    assert success is True
    assert '"okay": true' in output.lower()


def test_verify_failure_includes_lean_error():
    verifier = AxleLeanVerifier(api_key="axle-key")
    payload = {
        "okay": False,
        "lean_messages": {"errors": [{"message": "type mismatch"}]},
    }

    with patch("tinker_integration.axle_verifier.request.urlopen", return_value=_MockResponse(payload)):
        success, output = verifier.verify("import Mathlib\ntheorem t : False := by trivial")

    assert success is False
    assert "type mismatch" in output


def test_verify_http_error_uses_error_payload():
    verifier = AxleLeanVerifier(api_key="axle-key")
    http_error = error.HTTPError(
        url="https://example.test/api/v1/check",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=io.BytesIO(b'{"error": "bad request"}'),
    )

    with patch("tinker_integration.axle_verifier.request.urlopen", side_effect=http_error):
        success, output = verifier.verify("import Mathlib")

    assert success is False
    assert "bad request" in output


def test_verify_respects_no_sorries():
    verifier = AxleLeanVerifier(api_key="axle-key", no_sorries=True)

    with patch("tinker_integration.axle_verifier.request.urlopen") as mock_urlopen:
        success, output = verifier.verify("theorem t : True := by\n  sorry")

    assert success is False
    assert "sorry" in output
    mock_urlopen.assert_not_called()
