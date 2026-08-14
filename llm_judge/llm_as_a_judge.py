"""
LLM-as-a-Judge, applied to simplifying insurance policy text
================================================================

This script uses one LLM to *rewrite* dense insurance policy text into
plain language, and a second LLM role -- the judge -- to decide whether
each rewrite is actually good enough yet. If the judge says no, the draft
gets rewritten again using the judge's feedback, and the cycle repeats
until the judge approves or a maximum number of rounds is reached.

The idea comes from the paper "A Survey on LLM-as-a-Judge" (Gu, Jiang,
Shi, et al., arXiv:2411.15594v6) -- the PDF is in the repo root as
`2411.15594v6.pdf`.

The applied task, in plain terms
-----------------------------------
You give the program one paragraph of real insurance-policy legalese. It
then runs a loop:

    draft  ->  judge checks it  ->  not good enough?  ->  rewrite  ->  repeat

On each round, the judge looks at the *original* legal paragraph and the
*current* simplified draft, and answers two yes/no questions:

  1. Is this understandable to a layman with no insurance or legal
     background?
  2. Does it still mean the same thing as the original -- did the rewrite
     accidentally drop or distort an important condition, exclusion, or
     number?

If both are "yes", the loop stops and that draft is the final answer. If
either is "no", the judge also writes one sentence explaining what's still
wrong, and that feedback is fed straight into the next rewrite attempt.

This is a real use of the "LLM-as-a-Judge" pattern where the judge isn't
just picking a winner between two fixed options -- it's acting as the
*stopping condition* for an iterative loop. The paper itself calls out
exactly this style of use: a Yes/No judgment used as a feedback signal
that decides whether another iteration is needed (Section 2.1.2, citing
Reflexion's "Modification needed." / "No modification needed." pattern).

The two reliability ideas borrowed from the paper
-----------------------------------------------------
A single raw judge call is a bit noisy and easy to fool, so two ideas from
the paper's reliability discussion are used to make the stop/continue
decision trustworthy rather than a coin flip:

1. **Ask the judge more than once per round and require agreement.**
   The paper recommends running several independent judgments on the same
   thing and combining them rather than trusting one call (Section 3.3.1,
   "perform multiple runs of evaluation ... and summarize these
   results"). Here, each round asks the judge N times independently; the
   draft only advances to "approved" if a strict majority of those calls
   say yes on *both* clarity and faithfulness. This stops one lucky or
   unlucky judge call from ending the loop too early or too late.

2. **Force the judge to answer in structured JSON, not free text.**
   The paper calls this out directly as a reliability technique (Section
   3.1.2, "Standardizing LLMs' Output Format"). It's much easier to
   reliably extract a yes/no answer plus feedback text from
   `{"clear": true, "faithful": true, "feedback": "..."}` than from a
   sentence buried in prose. Because real models sometimes still wrap
   JSON in extra text anyway, the parser below has a few fallback
   strategies before it gives up (Section 2.3.1, "Extracting specific
   tokens").

Everything above -- the refine/judge loop, the majority-vote gate, the
parsing -- is plain, hand-written Python in this file. The only external
library used is the official `groq` SDK (Groq hosts open models like
Llama and GPT-OSS behind a fast inference API), and it is used only to
make individual API calls; no agent framework or orchestration library is
involved anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The model used for both roles (rewriter and judge). Override with --model.
# Groq hosts several open models; gpt-oss-120b is a strong general default
# for instruction-following / structured JSON output like this script needs.
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

    def _call(self, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=1024,
            messages=[
                {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

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
            # Ask Groq to constrain the reply to valid JSON. This is the same
            # "standardize the output format" reliability technique the
            # paper describes (Section 3.1.2) -- the parser below still
            # keeps its fallback strategies in case a model doesn't honor it.
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

    This is the actual decision logic of the whole program -- plain
    Python control flow, not a call into an agent framework.
    """
    result = RoundResult(round_number=round_number, draft_text=draft_text)
    majority_threshold = n_samples // 2 + 1

    for sample_index in range(n_samples):
        raw_output = judge.call(original_text, draft_text)
        clear, faithful, feedback = parse_judge_output(raw_output)

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
    max_rounds times.

        draft = rewriter.write_first_draft(original_text)
        for round in range(max_rounds):
            verdict = judge_round(draft)        # majority vote, see above
            if verdict.approved:
                return draft as final answer
            draft = rewriter.revise(original_text, draft, verdict.feedback)
        return the last draft produced, marked as not approved

    This is the whole point of the assignment made concrete: the judge is
    not a one-shot grader here, it is the stopping condition that decides
    whether the loop keeps going.
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
            draft = rewriter.revise(original_text, draft, result.combined_feedback)

    # Ran out of rounds without approval -- return the last draft produced,
    # clearly marked as not approved rather than silently pretending success.
    run.final_text = draft
    run.approved = False
    if callback:
        callback({"type": "final", "approved": False, "text": draft})
    return run


# ---------------------------------------------------------------------------
# The applied task: real insurance-policy paragraphs to simplify
# ---------------------------------------------------------------------------


def build_examples() -> list[str]:
    """A few real-style dense insurance-policy paragraphs, covering
    different kinds of jargon: exclusions, conditional coverage, and
    procedural/time-limit language."""
    return [
        (
            "Notwithstanding any other provision of this Policy, the Company shall not be liable "
            "for any loss or damage occurring subsequent to the Insured's failure to notify the "
            "Company in writing within thirty (30) days of the occurrence giving rise to such loss, "
            "and any claim submitted beyond such period shall be deemed forfeited absent a showing "
            "of good cause acceptable to the Company in its sole discretion."
        ),
        (
            "Coverage under Section 4(b) is contingent upon the Insured maintaining the subject "
            "premises in a condition free of any pre-existing defect known or reasonably "
            "discoverable by the Insured prior to the inception of this Policy, and the Company "
            "reserves the right to deny indemnification for any claim arising, in whole or in part, "
            "from such a pre-existing defect, irrespective of whether said defect was the proximate "
            "cause of the loss."
        ),
        (
            "The aggregate limit of liability for all claims arising out of a single occurrence "
            "shall not exceed the lesser of (i) the Limit of Insurance stated in the Declarations, "
            "or (ii) the actual cash value of the covered property at the time of loss, less the "
            "applicable deductible set forth in Item 4 of the Declarations, and in no event shall "
            "the Company be obligated to pay any amount in excess of said limit regardless of the "
            "number of claimants or causes of action asserted."
        ),
    ]


def run_all_examples(rewriter: Rewriter, judge: Judge, examples: list[str], max_rounds: int, n_judge_samples: int) -> list[RefinementRun]:
    runs = []
    for i, text in enumerate(examples, start=1):
        print(f"\n{'#' * 60}\n# EXAMPLE {i}/{len(examples)}\n{'#' * 60}")
        print(f"Original:\n{text}\n")
        run = refine_until_approved(rewriter, judge, text, max_rounds=max_rounds, n_judge_samples=n_judge_samples)
        print(f"\n{'=' * 60}")
        status = "APPROVED" if run.approved else f"NOT approved after {max_rounds} rounds (showing best attempt)"
        print(f"RESULT ({status}), {len(run.rounds)} round(s) used:")
        print(run.final_text)
        print("=" * 60)
        runs.append(run)
    return runs


def summarize(runs: list[RefinementRun]) -> None:
    total = len(runs)
    approved = sum(1 for r in runs if r.approved)
    total_rounds = sum(len(r.rounds) for r in runs)
    total_judge_calls = sum(len(round_.votes) for r in runs for round_ in r.rounds)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Approved:                {approved}/{total}")
    print(f"Average rounds per item: {total_rounds / total:.1f}")
    print(f"Total judge calls made:  {total_judge_calls}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge applied to iteratively simplifying insurance policy text."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model used for both roles (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help=f"Max refine/judge rounds per paragraph (default: {MAX_ROUNDS})")
    parser.add_argument(
        "--n-judge-samples",
        type=int,
        default=N_JUDGE_SAMPLES,
        help=f"How many times to ask the judge per round, for majority voting (default: {N_JUDGE_SAMPLES})",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Simplify this exact paragraph instead of running the three built-in examples.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "ERROR: GROQ_API_KEY is not set. Export your Groq API key, e.g.\n"
            "    export GROQ_API_KEY=gsk_...\n"
            "Get one at https://console.groq.com/keys\n",
            file=sys.stderr,
        )
        sys.exit(1)

    client = groq.Groq(api_key=api_key)
    rewriter = Rewriter(client, model=args.model)
    judge = Judge(client, model=args.model)

    start = time.time()
    if args.text:
        runs = run_all_examples(rewriter, judge, [args.text], args.max_rounds, args.n_judge_samples)
    else:
        runs = run_all_examples(rewriter, judge, build_examples(), args.max_rounds, args.n_judge_samples)
    summarize(runs)
    print(f"\n(total time: {time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
