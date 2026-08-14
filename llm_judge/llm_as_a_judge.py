"""LLM-as-a-Judge: rewrites dense insurance-policy text into plain language,
using a second LLM as a judge that must approve it before the loop stops.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The model used for both roles (rewriter and judge).
DEFAULT_MODEL = "openai/gpt-oss-120b"

# How many independent times to ask the judge about the SAME draft, each
# round. The draft only advances if a strict majority say it's ready.
N_JUDGE_SAMPLES = 3

# Safety cap so the loop always terminates even if the judge never approves.
MAX_ROUNDS = 5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JudgeVote:
    """One raw call to the judge about one draft, before any vote-counting."""

    sample_index: int
    raw_output: str
    clear: bool | None
    faithful: bool | None
    feedback: str | None


@dataclass
class RoundResult:
    """Everything that happened in one iteration of the refine/judge loop."""

    round_number: int
    draft_text: str
    votes: list[JudgeVote] = field(default_factory=list)
    clear_votes: int = 0
    faithful_votes: int = 0
    approved: bool = False
    combined_feedback: str = ""


@dataclass
class RefinementRun:
    """The full trace of one paragraph being refined until it's approved
    (or the round cap is hit)."""

    original_text: str
    rounds: list[RoundResult] = field(default_factory=list)
    final_text: str = ""
    approved: bool = False


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

REWRITER_SYSTEM_PROMPT = """You are an expert at rewriting insurance policy language into plain, \
everyday English that a person with no legal or insurance background can \
understand. You must keep every important detail from the original -- \
every condition, exclusion, dollar amount, time limit, and requirement -- \
just express it in simple, direct language. Do not add information that \
was not in the original, and do not drop or soften any condition that \
limits what is covered.

Respond with ONLY the rewritten paragraph. No preamble, no headings, no \
explanation of what you changed -- just the plain-language paragraph itself."""

REWRITER_INITIAL_TEMPLATE = """Rewrite the following insurance policy paragraph in plain language:

{original_text}"""

REWRITER_REVISION_TEMPLATE = """Original insurance policy paragraph (for reference -- do not lose any of its meaning):
{original_text}

Your previous rewrite:
{previous_draft}

A reviewer said this rewrite still isn't good enough:
{feedback}

Write an improved plain-language version that fixes this specific problem. \
Respond with ONLY the rewritten paragraph."""


JUDGE_SYSTEM_PROMPT = """You are an impartial judge checking whether a plain-language rewrite of an \
insurance policy paragraph is actually ready to show to a customer. Check \
two separate things:

1. clear: Would someone with NO insurance or legal background fully \
understand this on a single read? Watch for leftover jargon, long \
run-on sentences, or vague phrasing that a layman would stumble on.

2. faithful: Does the rewrite still convey every important condition, \
exclusion, dollar amount, time limit, and requirement from the original? \
A rewrite that is easy to read but has quietly dropped or softened a \
real limitation is NOT faithful, even if it reads well.

Be strict on both. A rewrite that fails either check is not ready.

You must respond with ONLY a single JSON object, no other text, matching \
exactly this schema:
{"clear": true or false, "faithful": true or false, "feedback": "<one sentence on the single biggest remaining problem, or empty string if both are true>"}
"""

JUDGE_USER_TEMPLATE = """Original insurance policy paragraph:
{original_text}

Plain-language rewrite to check:
{draft_text}

Evaluate this rewrite. Respond with the JSON object only."""


# ---------------------------------------------------------------------------
# The two model roles: rewriter and judge
# ---------------------------------------------------------------------------


class Rewriter:
    """The agent that produces each successive draft. A thin wrapper around
    one model call -- all of the looping and decision-making about whether
    to call this again lives in `refine_until_approved` below."""

    def __init__(self, client: groq.Groq, model: str = DEFAULT_MODEL):
        self._client = client
        self._model = model

    def _call(self, user: str, retries: int = 2) -> str:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_completion_tokens=1024,
                    messages=[
                        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    return text
                # The provider returned no content for this call (e.g. cut
                # off, or an empty completion). Retry rather than silently
                # handing back an empty draft that's a guaranteed judge fail.
                last_error = RuntimeError(
                    f"Rewriter returned no content (finish_reason={response.choices[0].finish_reason!r})"
                )
            except Exception as exc:
                last_error = exc
        raise last_error

    def write_first_draft(self, original_text: str) -> str:
        return self._call(REWRITER_INITIAL_TEMPLATE.format(original_text=original_text))

    def revise(self, original_text: str, previous_draft: str, feedback: str) -> str:
        user = REWRITER_REVISION_TEMPLATE.format(
            original_text=original_text, previous_draft=previous_draft, feedback=feedback
        )
        return self._call(user)


class Judge:
    """The agent that decides whether a draft is ready. Also a thin wrapper
    around one model call -- the multi-sample voting logic that turns
    several of these raw calls into one trustworthy decision lives in
    `judge_round` below, not in this class."""

    def __init__(self, client: groq.Groq, model: str = DEFAULT_MODEL):
        self._client = client
        self._model = model

    def call(self, original_text: str, draft_text: str) -> str:
        user = JUDGE_USER_TEMPLATE.format(original_text=original_text, draft_text=draft_text)
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Turning the judge's raw text into a usable answer
# ---------------------------------------------------------------------------


def parse_judge_output(raw: str) -> tuple[bool | None, bool | None, str | None]:
    """
    Pull (clear, faithful, feedback) out of the judge's raw reply. The
    judge is told to answer with plain JSON, but models sometimes wrap the
    JSON in a code fence or add a stray sentence, so this tries several
    strategies before giving up:
      1. parse the whole reply as JSON
      2. strip a ```json ... ``` fence and try again
      3. find the first {...} anywhere in the text and try that
    Returns (None, None, None) if nothing could be extracted.
    """
    candidate_json = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate_json, re.DOTALL)
    if fence_match:
        candidate_json = fence_match.group(1)

    for text in (candidate_json, raw):
        try:
            obj = json.loads(text)
            clear, faithful = obj.get("clear"), obj.get("faithful")
            if isinstance(clear, bool) and isinstance(faithful, bool):
                return clear, faithful, obj.get("feedback")
        except (json.JSONDecodeError, AttributeError):
            pass

    brace_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if brace_match:
        try:
            obj = json.loads(brace_match.group(0))
            clear, faithful = obj.get("clear"), obj.get("faithful")
            if isinstance(clear, bool) and isinstance(faithful, bool):
                return clear, faithful, obj.get("feedback")
        except (json.JSONDecodeError, AttributeError):
            pass

    return None, None, None


# ---------------------------------------------------------------------------
# The reliability layer: turning raw judge calls into a trustworthy verdict
# ---------------------------------------------------------------------------


def judge_round(judge: Judge, original_text: str, draft_text: str, round_number: int, n_samples: int, verbose: bool, callback=None) -> RoundResult:
    """
    Judge one draft with a majority vote instead of trusting a single call.

      ask the judge n_samples times, independently
      count how many said "clear" and how many said "faithful"
      approved = (strict majority said clear) AND (strict majority said faithful)
      combined_feedback = feedback text from one of the calls that voted "not ready"
                          (or empty, if approved)
    """
    result = RoundResult(round_number=round_number, draft_text=draft_text)
    majority_threshold = n_samples // 2 + 1

    for sample_index in range(n_samples):
        try:
            raw_output = judge.call(original_text, draft_text)
            clear, faithful, feedback = parse_judge_output(raw_output)
        except Exception as exc:
            # A single failed call (network error, or the provider rejecting
            # its own malformed JSON generation) shouldn't crash the whole
            # round -- treat it the same as a reply we couldn't parse, so
            # the other samples still get a chance to form a majority.
            raw_output = f"<call failed: {exc}>"
            clear, faithful, feedback = None, None, None

        result.votes.append(
            JudgeVote(sample_index=sample_index, raw_output=raw_output, clear=clear, faithful=faithful, feedback=feedback)
        )
        if clear is True:
            result.clear_votes += 1
        if faithful is True:
            result.faithful_votes += 1

        if verbose:
            print(f"    [judge #{sample_index + 1}] clear={clear} faithful={faithful}  ({feedback!r})")

        if callback:
            callback({
                "type": "judge_vote",
                "round_number": round_number,
                "sample_index": sample_index + 1,
                "clear": clear,
                "faithful": faithful,
                "feedback": feedback
            })

        # Keep the most recent piece of critical feedback in reserve, in
        # case the round as a whole is not approved.
        if feedback and (clear is False or faithful is False):
            result.combined_feedback = feedback

    result.approved = result.clear_votes >= majority_threshold and result.faithful_votes >= majority_threshold

    if result.approved:
        result.combined_feedback = ""
    elif not result.combined_feedback:
        # Every vote failed to parse, or approved votes gave no feedback text.
        result.combined_feedback = "The rewrite is still too complex or may have altered the original meaning -- simplify further while keeping every condition intact."

    if callback:
        callback({
            "type": "round_result",
            "round_number": round_number,
            "approved": result.approved,
            "clear_votes": result.clear_votes,
            "faithful_votes": result.faithful_votes,
            "combined_feedback": result.combined_feedback
        })

    return result


def refine_until_approved(
    rewriter: Rewriter,
    judge: Judge,
    original_text: str,
    max_rounds: int = MAX_ROUNDS,
    n_judge_samples: int = N_JUDGE_SAMPLES,
    verbose: bool = True,
    callback=None,
) -> RefinementRun:
    """
    The main agentic loop: write a draft, judge it, and if it's not
    approved, feed the judge's feedback into another rewrite -- up to
    max_rounds times. The judge is not a one-shot grader here, it is the
    stopping condition that decides whether the loop keeps going.
    """
    run = RefinementRun(original_text=original_text)

    if callback:
        callback({"type": "status", "message": "Drafting initial version..."})
    draft = rewriter.write_first_draft(original_text)

    for round_number in range(1, max_rounds + 1):
        if verbose:
            print(f"\n--- Round {round_number}/{max_rounds} ---")
            print(f"  draft: {draft}")

        if callback:
            callback({"type": "draft", "round_number": round_number, "text": draft})
            callback({"type": "status", "message": f"Judging round {round_number}..."})

        result = judge_round(judge, original_text, draft, round_number, n_judge_samples, verbose, callback)
        run.rounds.append(result)

        if verbose:
            status = "APPROVED" if result.approved else "needs another round"
            print(
                f"  -> {status}  (clear votes {result.clear_votes}/{n_judge_samples}, "
                f"faithful votes {result.faithful_votes}/{n_judge_samples})"
            )

        if result.approved:
            run.final_text = draft
            run.approved = True
            if callback:
                callback({"type": "final", "approved": True, "text": draft})
            return run

        if round_number < max_rounds:
            if verbose:
                print(f"  feedback for next round: {result.combined_feedback}")
            if callback:
                callback({"type": "status", "message": f"Drafting revision {round_number+1} based on feedback..."})
            try:
                draft = rewriter.revise(original_text, draft, result.combined_feedback)
            except Exception as exc:
                # The rewriter failed even after its own retries. Keep the
                # previous draft rather than losing it to an empty/failed
                # revision -- the next round's judge call will just see the
                # same text again instead of a guaranteed-fail blank draft.
                if verbose:
                    print(f"  revision failed, keeping previous draft: {exc}")
                if callback:
                    callback({"type": "status", "message": f"Revision failed ({exc}); retrying with previous draft."})

    # Ran out of rounds without approval -- return the last draft produced,
    # clearly marked as not approved rather than silently pretending success.
    run.final_text = draft
    run.approved = False
    if callback:
        callback({"type": "final", "approved": False, "text": draft})
    return run
