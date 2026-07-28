"""Unit tests for OpenAICompatProvider — no GPU / network / server.

The httpx client is replaced with a scripted fake, so these exercise the full
retry / drop / parse state machine against canned HTTP outcomes:
  * payload + auth-header construction (explicit key, $OPENAI_API_KEY, none)
  * completion order + the len(out) == len(prompts) contract
  * retries: 408/429/5xx + transport errors -> backoff -> success or None
  * loud failures: non-retryable HTTP status, unexpected finish_reason
  * drops: content_filter, empty completion, retry exhaustion
  * bounded concurrency (the semaphore is held across retries)
  * sidecar provenance attrs (model / max_tokens / temperature) for stage 2

Runs standalone (`python tests/test_openai_compat_provider.py`) or under
pytest. Needs nothing beyond the stdlib — `anthropic` and `httpx` are stubbed
into sys.modules when absent so `nla.datagen.providers` imports anywhere.
"""
import asyncio
import contextlib
import io
import json as _json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# --- make nla.datagen.providers importable without the real SDKs installed ---
try:
    import anthropic  # noqa: F401
except ImportError:
    _a = types.ModuleType("anthropic")
    for _n in ("RateLimitError", "InternalServerError", "APIConnectionError"):
        setattr(_a, _n, type(_n, (Exception,), {}))
    _a.AsyncAnthropic = object
    sys.modules["anthropic"] = _a
try:
    import httpx  # noqa: F401
except ImportError:
    _h = types.ModuleType("httpx")
    _h.TransportError = type("TransportError", (Exception,), {})
    _h.AsyncClient = None  # replaced per-test by _fake_client
    sys.modules["httpx"] = _h

import nla.datagen.providers as P  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (_json.dumps(payload) if payload is not None else "")

    def json(self):
        return self._payload


def _ok(text, reason="stop"):
    return FakeResponse(200, {"choices": [{"finish_reason": reason, "message": {"content": text}}]})


class FakeClient:
    """Scripted stand-in for httpx.AsyncClient.

    `script` maps prompt -> list of outcomes (FakeResponse, or an exception to
    raise). Each post() pops the next outcome for its prompt; running dry
    fails the test loudly. Tracks max concurrent post() calls in flight.
    """

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []  # (url, payload, headers) in arrival order
        self.in_flight = 0
        self.max_in_flight = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)  # yield so concurrent posts overlap deterministically
        self.in_flight -= 1
        queue = self.script.get(json["messages"][0]["content"])
        assert queue, f"unexpected extra call for prompt {json['messages'][0]['content']!r}"
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@contextlib.contextmanager
def _fake_client(script, *, env_key=None):
    """Install a FakeClient + no-op backoff; pin $OPENAI_API_KEY to env_key (None = unset)."""
    fake = FakeClient(script)
    saved_client, saved_sleep = P.httpx.AsyncClient, P._backoff_sleep
    saved_env = os.environ.pop("OPENAI_API_KEY", None)
    if env_key is not None:
        os.environ["OPENAI_API_KEY"] = env_key

    async def _nosleep(seconds):
        return None

    P.httpx.AsyncClient = lambda timeout=None: fake
    P._backoff_sleep = _nosleep
    try:
        yield fake
    finally:
        P.httpx.AsyncClient, P._backoff_sleep = saved_client, saved_sleep
        os.environ.pop("OPENAI_API_KEY", None)
        if saved_env is not None:
            os.environ["OPENAI_API_KEY"] = saved_env


def test_happy_path_order_payload_and_no_auth():
    prompts = ["p0", "p1", "p2"]
    with _fake_client({p: [_ok(f"answer-{p}")] for p in prompts}) as fake:
        prov = P.OpenAICompatProvider(
            model="m14b", base_url="http://h:8000/v1", max_tokens=123, temperature=0.7, max_retries=0
        )
        out = prov.complete(prompts)
    assert isinstance(prov, P.CompletionProvider)
    assert out == ["answer-p0", "answer-p1", "answer-p2"], "order must be preserved"
    assert len(fake.calls) == len(prompts)
    for url, payload, headers in fake.calls:
        assert url == "http://h:8000/v1/chat/completions"
        assert payload["model"] == "m14b" and payload["max_tokens"] == 123
        assert payload["temperature"] == 0.7
        [msg] = payload["messages"]
        assert msg["role"] == "user" and msg["content"] in prompts
        assert "Authorization" not in headers, "no key given and env unset -> no auth header"


def test_auth_header_key_param_and_base_url_slash():
    with _fake_client({"p": [_ok("t")]}) as fake:
        assert P.OpenAICompatProvider(api_key="sk-test", base_url="http://h:1/v1/").complete(["p"]) == ["t"]
    url, _, headers = fake.calls[0]
    assert url == "http://h:1/v1/chat/completions", "trailing slash must be normalized"
    assert headers["Authorization"] == "Bearer sk-test"


def test_auth_header_env_fallback():
    with _fake_client({"p": [_ok("t")]}, env_key="sk-env") as fake:
        assert P.OpenAICompatProvider().complete(["p"]) == ["t"]
    assert fake.calls[0][2]["Authorization"] == "Bearer sk-env"


def test_retryable_statuses_and_transport_then_success():
    script = {"p": [FakeResponse(429), FakeResponse(503), P.httpx.TransportError("boom"), FakeResponse(408), _ok("t")]}
    with _fake_client(script) as fake:
        assert P.OpenAICompatProvider(max_retries=4).complete(["p"]) == ["t"]
    assert len(fake.calls) == 5, "each retryable outcome consumes exactly one attempt"


def test_retry_exhausted_drops_row_others_survive():
    script = {"dead": [FakeResponse(500), FakeResponse(500)], "ok": [_ok("t")]}
    with _fake_client(script) as fake:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = P.OpenAICompatProvider(max_retries=1).complete(["dead", "ok"])
    assert out == [None, "t"], "exhausted row -> None in place; sibling unaffected"
    assert sum(1 for _, p, _ in fake.calls if p["messages"][0]["content"] == "dead") == 2
    assert "1 retry-exhausted of 2" in buf.getvalue()


def test_non_retryable_status_raises():
    with _fake_client({"p": [FakeResponse(401, text="bad key")]}) as fake:
        try:
            P.OpenAICompatProvider(max_retries=5).complete(["p"])
        except RuntimeError as e:
            assert "401" in str(e) and "bad key" in str(e)
        else:
            raise AssertionError("401 must raise, not retry/drop")
    assert len(fake.calls) == 1, "auth errors must not be retried"


def test_content_filter_empty_whitespace_length():
    script = {
        "f": [_ok("whatever", reason="content_filter")],
        "e": [_ok("")],
        "w": [_ok(" \n\t ")],
        "s": [_ok("  kept  ")],
        "l": [_ok("cut off mid-thought", reason="length")],
    }
    with _fake_client(script):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = P.OpenAICompatProvider(max_retries=0).complete(["f", "e", "w", "s", "l"])
    # length-truncations pass through — stage2's <analysis> tag regex is the filter for those
    assert out == [None, None, None, "kept", "cut off mid-thought"]
    assert "1 filtered + 2 empty" in buf.getvalue()


def test_unexpected_finish_reason_raises():
    with _fake_client({"p": [_ok("x", reason="tool_calls")]}):
        try:
            P.OpenAICompatProvider().complete(["p"])
        except AssertionError as e:
            assert "tool_calls" in str(e)
        else:
            raise AssertionError("unexpected finish_reason must raise")


def test_concurrency_bounded_by_semaphore():
    prompts = [f"p{i}" for i in range(6)]
    with _fake_client({p: [_ok("t")] for p in prompts}) as fake:
        assert P.OpenAICompatProvider(concurrency=2).complete(prompts) == ["t"] * 6
    assert fake.max_in_flight == 2, ">1 proves overlap, <=2 proves the cap"


def test_sidecar_provenance_attrs_and_defaults():
    """stage2 records provider config via getattr(provider, 'model'|'max_tokens'|'temperature')."""
    prov = P.OpenAICompatProvider()
    assert (prov.model, prov.max_tokens, prov.temperature) == ("Qwen/Qwen2.5-14B-Instruct", 300, 1.0)
    prov = P.OpenAICompatProvider(model="x", max_tokens=7, temperature=0.2)
    assert (prov.model, prov.max_tokens, prov.temperature) == ("x", 7, 0.2)


def test_unknown_result_kind_guard():
    """The defensive guard for a _one() result outside the known kinds must blow up loud."""

    async def _bogus(self, sem, client, prompt):
        return ("bogus",)

    saved = P.OpenAICompatProvider._one
    P.OpenAICompatProvider._one = _bogus
    try:
        with _fake_client({}):
            try:
                P.OpenAICompatProvider().complete(["p"])
            except AssertionError as e:
                assert "bogus" in str(e)
            else:
                raise AssertionError("unknown result kind must raise")
    finally:
        P.OpenAICompatProvider._one = saved


def test_backoff_sleep_really_sleeps():
    asyncio.run(P._backoff_sleep(0.001))  # the real coroutine body (stubbed everywhere else)


if __name__ == "__main__":
    tests = [
        test_happy_path_order_payload_and_no_auth,
        test_auth_header_key_param_and_base_url_slash,
        test_auth_header_env_fallback,
        test_retryable_statuses_and_transport_then_success,
        test_retry_exhausted_drops_row_others_survive,
        test_non_retryable_status_raises,
        test_content_filter_empty_whitespace_length,
        test_unexpected_finish_reason_raises,
        test_concurrency_bounded_by_semaphore,
        test_sidecar_provenance_attrs_and_defaults,
        test_unknown_result_kind_guard,
        test_backoff_sleep_really_sleeps,
    ]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nall {len(tests)} OpenAICompatProvider unit tests passed")
