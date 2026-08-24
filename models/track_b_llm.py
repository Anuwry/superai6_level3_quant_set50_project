from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar

import tiktoken
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "gpt-5.6-terra"
PROMPT_VERSION = "track-b-terra-v1"
OPENAI_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}")


class SentimentVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant: bool
    positive_probability: float = Field(ge=0.0, le=1.0)
    neutral_probability: float = Field(ge=0.0, le=1.0)
    negative_probability: float = Field(ge=0.0, le=1.0)
    predicted_label: Literal["positive", "neutral", "negative"]
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class WorkerArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stance: Literal["bullish", "bearish"]
    relevance_probability: float = Field(ge=0.0, le=1.0)
    claim_strength: int = Field(ge=0, le=100)
    evidence: list[str] = Field(min_length=1, max_length=2)
    counterevidence: str


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    article_id: str
    ticker: str
    date: str
    text: str
    text_sha256: str

    @classmethod
    def from_values(
        cls,
        *,
        article_id: object,
        ticker: object,
        date: object,
        text: object,
    ) -> BenchmarkItem:
        article = str(text).strip()
        article_id_value = str(article_id).strip()
        ticker_value = str(ticker).strip().upper()
        if not article or not article_id_value or not ticker_value:
            raise ValueError("Article ID, ticker, and text must be non-empty")
        return cls(
            item_id=f"{article_id_value}::{ticker_value}",
            article_id=article_id_value,
            ticker=ticker_value,
            date=str(date),
            text=article,
            text_sha256=hashlib.sha256(article.encode("utf-8")).hexdigest(),
        )


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class Pricing:
    input_usd_per_million: float = 2.5
    output_usd_per_million: float = 15.0


ParsedOutput = SentimentVerdict | WorkerArgument


@dataclass(frozen=True)
class LLMCall:
    role: str
    response_id: str
    parsed: ParsedOutput
    usage: TokenUsage
    runtime_seconds: float
    cost_usd: float


@dataclass(frozen=True)
class DebateResult:
    item: BenchmarkItem
    single: LLMCall | None
    bull: LLMCall
    bear: LLMCall
    leader: LLMCall

    @property
    def total_cost_usd(self) -> float:
        calls = [self.bull, self.bear, self.leader]
        if self.single is not None:
            calls.append(self.single)
        return float(sum(call.cost_usd for call in calls))

    @property
    def total_runtime_seconds(self) -> float:
        parallel_stage = max(
            call.runtime_seconds
            for call in (self.bull, self.bear, self.single)
            if call is not None
        )
        return parallel_stage + self.leader.runtime_seconds


class StructuredClient(Protocol):
    def generate(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        response_type: type[BaseModel],
        max_output_tokens: int,
    ) -> LLMCall: ...


def calculate_cost_usd(usage: TokenUsage, pricing: Pricing) -> float:
    input_cost = usage.input_tokens * pricing.input_usd_per_million / 1_000_000
    output_cost = usage.output_tokens * pricing.output_usd_per_million / 1_000_000
    return float(input_cost + output_cost)


class BudgetExceededError(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, limit_usd: float) -> None:
        if limit_usd <= 0:
            raise ValueError("Budget limit must be positive")
        self.limit_usd = float(limit_usd)
        self.spent_usd = 0.0
        self.reserved_usd = 0.0
        self._reservations: dict[str, float] = {}
        self._lock = threading.Lock()

    def reserve(self, estimated_cost_usd: float) -> str:
        if estimated_cost_usd < 0:
            raise ValueError("Estimated cost cannot be negative")
        with self._lock:
            projected = self.spent_usd + self.reserved_usd + estimated_cost_usd
            if projected > self.limit_usd + 1e-12:
                raise BudgetExceededError(
                    f"Cost guard blocked request at ${projected:.4f}; "
                    f"limit is ${self.limit_usd:.2f}"
                )
            reservation = uuid.uuid4().hex
            self._reservations[reservation] = estimated_cost_usd
            self.reserved_usd += estimated_cost_usd
            return reservation

    def commit(self, reservation: str, *, actual_cost_usd: float) -> None:
        if actual_cost_usd < 0:
            raise ValueError("Actual cost cannot be negative")
        with self._lock:
            reserved = self._reservations.pop(reservation)
            self.reserved_usd -= reserved
            self.spent_usd += actual_cost_usd
            if self.spent_usd > self.limit_usd + 1e-12:
                raise BudgetExceededError("Actual API usage exceeded the cost guard")

    def cancel(self, reservation: str) -> None:
        with self._lock:
            reserved = self._reservations.pop(reservation, None)
            if reserved is not None:
                self.reserved_usd -= reserved


def extract_api_key(path: str | Path) -> str:
    raw = Path(path).read_text(encoding="utf-8")
    match = OPENAI_KEY_PATTERN.search(raw)
    if match is None:
        raise ValueError("No OpenAI API key pattern found in the supplied file")
    return match.group(0)


def resolve_api_key(key_file: str | Path | None = None) -> str:
    environment_key = os.getenv("OPENAI_API_KEY", "").strip()
    if environment_key:
        return environment_key
    if key_file is None:
        raise ValueError("OPENAI_API_KEY is unset and no key file was supplied")
    return extract_api_key(key_file)


def _article_block(item: BenchmarkItem) -> str:
    return (
        f"Publication date: {item.date}\n"
        f"Target ticker: {item.ticker}\n"
        f"Article:\n{item.text}"
    )


def build_single_messages(item: BenchmarkItem) -> list[dict[str, str]]:
    system = (
        "You are a Thai equity sentiment classifier. Assess only the likely "
        "effect of the supplied article on the target ticker at publication "
        "time. Do not infer facts absent from the article. Neutral means no "
        "clear directional effect. Probabilities must sum to 1. Keep the "
        "rationale to one short sentence."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _article_block(item)},
    ]


def build_worker_messages(
    item: BenchmarkItem,
    *,
    stance: Literal["bullish", "bearish"],
) -> list[dict[str, str]]:
    direction = (
        "positive or price-supportive"
        if stance == "bullish"
        else "negative or price-damaging"
    )
    system = (
        f"You are the {stance} debate worker for Thai equities. Build the "
        f"strongest evidence-grounded case that the article is {direction} "
        "for the target ticker. Quote at most two short evidence fragments, "
        "acknowledge the strongest counterevidence, and never invent facts."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _article_block(item)},
    ]


def build_leader_messages(
    item: BenchmarkItem,
    bull: WorkerArgument,
    bear: WorkerArgument,
) -> list[dict[str, str]]:
    system = (
        "You are an impartial leader adjudicating two constrained arguments. "
        "Re-read the original Thai article, reject unsupported worker claims, "
        "and decide the likely effect on the target ticker at publication "
        "time. Neutral is valid when evidence is balanced or non-directional. "
        "Probabilities must sum to 1. Keep the rationale to one short sentence."
    )
    reports = json.dumps(
        {"bull_worker": bull.model_dump(), "bear_worker": bear.model_dump()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    user = f"{_article_block(item)}\nWorker reports:\n{reports}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _message_tokens(messages: list[dict[str, str]]) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    return sum(
        len(encoding.encode(message["role"]))
        + len(encoding.encode(message["content"]))
        + 4
        for message in messages
    )


TModel = TypeVar("TModel", bound=BaseModel)


class OpenAIStructuredClient:
    def __init__(
        self,
        *,
        api_key: str,
        budget: BudgetLedger,
        model: str = MODEL_ID,
        pricing: Pricing | None = None,
        reasoning_effort: str = "low",
        timeout_seconds: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            max_retries=2,
            timeout=timeout_seconds,
        )
        self.budget = budget
        self.model = model
        self.pricing = pricing or Pricing()
        self.reasoning_effort = reasoning_effort

    def _estimated_cost(
        self,
        messages: list[dict[str, str]],
        max_output_tokens: int,
    ) -> float:
        usage = TokenUsage(
            input_tokens=_message_tokens(messages) + 100,
            output_tokens=max_output_tokens,
        )
        return calculate_cost_usd(usage, self.pricing)

    @staticmethod
    def _usage(response) -> TokenUsage:
        usage = response.usage
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        return TokenUsage(
            input_tokens=int(usage.input_tokens),
            output_tokens=int(usage.output_tokens),
            cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
            reasoning_tokens=int(getattr(output_details, "reasoning_tokens", 0) or 0),
        )

    def generate(
        self,
        *,
        role: str,
        messages: list[dict[str, str]],
        response_type: type[TModel],
        max_output_tokens: int,
    ) -> LLMCall:
        estimate = self._estimated_cost(messages, max_output_tokens)
        reservation = self.budget.reserve(estimate)
        started = time.perf_counter()
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=messages,
                text_format=response_type,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": self.reasoning_effort},
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError(f"{role} returned no parsed structured output")
            usage = self._usage(response)
            cost = calculate_cost_usd(usage, self.pricing)
            self.budget.commit(reservation, actual_cost_usd=cost)
            return LLMCall(
                role=role,
                response_id=str(response.id),
                parsed=parsed,
                usage=usage,
                runtime_seconds=time.perf_counter() - started,
                cost_usd=cost,
            )
        except Exception:
            self.budget.cancel(reservation)
            raise


class DebateOrchestrator:
    def __init__(
        self,
        *,
        client: StructuredClient,
        single_max_output_tokens: int = 384,
        worker_max_output_tokens: int = 384,
        leader_max_output_tokens: int = 512,
    ) -> None:
        self.client = client
        self.single_max_output_tokens = single_max_output_tokens
        self.worker_max_output_tokens = worker_max_output_tokens
        self.leader_max_output_tokens = leader_max_output_tokens

    def _single(self, item: BenchmarkItem) -> LLMCall:
        return self.client.generate(
            role="single",
            messages=build_single_messages(item),
            response_type=SentimentVerdict,
            max_output_tokens=self.single_max_output_tokens,
        )

    def _worker(
        self,
        item: BenchmarkItem,
        stance: Literal["bullish", "bearish"],
    ) -> LLMCall:
        role = "bull" if stance == "bullish" else "bear"
        return self.client.generate(
            role=role,
            messages=build_worker_messages(item, stance=stance),
            response_type=WorkerArgument,
            max_output_tokens=self.worker_max_output_tokens,
        )

    def process(
        self,
        item: BenchmarkItem,
        *,
        include_single: bool = True,
    ) -> DebateResult:
        with ThreadPoolExecutor(max_workers=3) as executor:
            bull_future = executor.submit(self._worker, item, "bullish")
            bear_future = executor.submit(self._worker, item, "bearish")
            single_future = (
                executor.submit(self._single, item) if include_single else None
            )
            bull = bull_future.result()
            bear = bear_future.result()
            single = single_future.result() if single_future is not None else None
        leader = self.client.generate(
            role="leader",
            messages=build_leader_messages(
                item,
                bull.parsed,
                bear.parsed,
            ),
            response_type=SentimentVerdict,
            max_output_tokens=self.leader_max_output_tokens,
        )
        return DebateResult(
            item=item,
            single=single,
            bull=bull,
            bear=bear,
            leader=leader,
        )
