"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import os
from abc import ABC, abstractmethod

import anthropic
import httpx


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


async def _backoff_sleep(seconds: float) -> None:
    """Isolated so tests can stub the retry wait without touching asyncio globally."""
    await asyncio.sleep(seconds)


class OpenAICompatProvider(CompletionProvider):
    """OpenAI-compatible `/chat/completions` provider — local vLLM/SGLang or hosted.

    Speaks the OpenAI chat-completions HTTP protocol directly over httpx (no
    vendor SDK), so one class covers a local `vllm serve` explainer, OpenAI
    itself, or Gemini's OpenAI-compat endpoint — set `base_url` (plus
    `api_key` or $OPENAI_API_KEY for hosted APIs; local servers need neither).
    Written for the keyless local-explainer path: serve e.g.
    Qwen2.5-14B-Instruct with vLLM and point stage 2 here via `--provider-cls`.

    Same contract as AnthropicProvider (`prompts[i] -> completion[i] | None`),
    with retry/drop semantics adapted to raw HTTP + local models:

    - 408/429/5xx and transport errors (connect/read/timeout) retry with
      exponential backoff. The concurrency slot is held while backing off, so
      a struggling server sees *less* traffic, not a stampede. Exhausting
      retries yields None (caller drops the row).
    - Any other HTTP status raises immediately — auth/schema/payload problems
      are code bugs, not transients.
    - `finish_reason == "content_filter"` -> None (hosted-refusal analog).
    - Empty/whitespace completions -> None instead of AnthropicProvider's
      assert: small local models occasionally emit blanks; that is a
      data-quality drop, not an API-contract violation.

    Sends the standard `max_tokens` field (vLLM / SGLang / llama.cpp / classic
    OpenAI). Hosted OpenAI *reasoning* models want `max_completion_tokens`
    instead — not this class's target use.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    """

    _RETRYABLE_STATUS = (408, 429)  # plus every 5xx

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-14B-Instruct",
        base_url: str = "http://127.0.0.1:8000/v1",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
        api_key: str | None = None,
        timeout: float = 180.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def _parse(self, data: dict) -> tuple:
        choice = data["choices"][0]
        reason = choice.get("finish_reason")
        if reason == "content_filter":
            return ("refused",)
        assert reason in ("stop", "length"), (
            f"unexpected finish_reason={reason!r} (want stop/length/content_filter)"
        )
        text = ((choice.get("message") or {}).get("content") or "").strip()
        return ("text", text) if text else ("empty",)

    async def _one(self, sem: asyncio.Semaphore, client, prompt: str) -> tuple:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                except httpx.TransportError:
                    resp = None  # connect/read/timeout — retryable
                if resp is not None:
                    if resp.status_code == 200:
                        return self._parse(resp.json())
                    if resp.status_code not in self._RETRYABLE_STATUS and resp.status_code < 500:
                        raise RuntimeError(
                            f"non-retryable HTTP {resp.status_code} from {url}: {resp.text[:500]}"
                        )
                if attempt < self.max_retries:
                    await _backoff_sleep(min(2.0**attempt, 30.0))
        return ("exhausted",)

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list:
            sem = asyncio.Semaphore(self.concurrency)
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await asyncio.gather(
                    *(self._one(sem, client, p) for p in prompts),
                    return_exceptions=True,
                )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        drops = {"refused": 0, "empty": 0, "exhausted": 0}
        for i, r in enumerate(raw):
            if isinstance(r, BaseException):
                raise r  # non-retryable / contract violation — blow up loud
            kind = r[0]
            if kind == "text":
                out.append(r[1])
            elif kind in drops:
                drops[kind] += 1
                out.append(None)
            else:
                raise AssertionError(f"gather returned unexpected kind at [{i}]: {r!r}")
        if any(drops.values()):
            print(
                f"  [OpenAICompatProvider] dropped {drops['refused']} filtered "
                f"+ {drops['empty']} empty + {drops['exhausted']} retry-exhausted of {len(prompts)}"
            )
        return out
