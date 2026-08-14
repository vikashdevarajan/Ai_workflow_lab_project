# LLM-as-a-Judge — simplifying insurance policy text

This program takes a paragraph of dense insurance-policy legalese and
rewrites it into plain language a normal person can understand — not in
one shot, but through a loop: write a draft, have a judge check it, and if
the judge isn't satisfied, rewrite it again using the judge's feedback.
This repeats until the judge approves, or a maximum number of rounds is
reached.

It's a from-scratch implementation of the **LLM-as-a-Judge** pattern,
based on the paper **"A Survey on LLM-as-a-Judge"** (Gu, Jiang, Shi, et
al., arXiv:2411.15594v6) — the PDF is in the repo root
(`2411.15594v6.pdf`).

No agent framework or orchestration library is used. The refine/judge
loop, the majority-vote gate, and the output parsing are all plain,
hand-written Python in `llm_as_a_judge.py`. The only third-party
dependency is the official `groq` SDK, used only to make individual API
calls. Groq hosts fast inference for several open models (Llama,
GPT-OSS, Kimi K2, etc.) — this script defaults to `openai/gpt-oss-120b`.

## How the loop works

```
draft = rewriter writes a first plain-language attempt
loop up to MAX_ROUNDS times:
    judge checks the current draft against the original text
    if the judge approves -> done, return this draft
    otherwise -> rewriter revises the draft using the judge's feedback
if we run out of rounds -> return the last draft, marked "not approved"
```

The judge doesn't just grade the draft once — every round it's asked
**two separate yes/no questions**:

1. **clear** — would someone with no insurance or legal background fully
   understand this?
2. **faithful** — does the rewrite still say everything the original said
   (every condition, exclusion, dollar amount, time limit), or did
   simplifying it quietly drop or soften something important?

A draft only passes if *both* checks pass. This matters because it's easy
for a naive simplifier to just delete the confusing parts instead of
actually explaining them — the faithfulness check exists specifically to
catch that.

## What the paper says, and what this code does about it

The paper's basic formula for LLM-as-a-Judge is:

```
E <- P_LLM(x + C)
```

In plain terms: give the judge model an input `x` wrapped in a prompt `C`,
and it produces an evaluation `E`. That's the core idea, and it's what
`Judge.call()` does — one call in, one raw answer out.

The paper also points out that using a Yes/No judgment as a **feedback
signal that decides whether to keep iterating** is itself a documented
pattern (Section 2.1.2), citing systems like Reflexion that output
`"Modification needed."` or `"No modification needed."` to drive a
self-improvement loop. That's exactly the role the judge plays here — not
a one-off grade, but the thing deciding whether the loop continues.

But a single raw judge call isn't something you should trust blindly — it
can be inconsistent from one call to the next. So this script adds one
reliability technique straight from the paper's reliability section
(Section 3.3.1, "perform multiple runs of evaluation ... and summarize
these results"):

- **Ask the judge several times per round, and require a majority.**
  Instead of asking the judge once and trusting whatever it says, each
  round asks it `N` times independently (default 3). The draft only
  advances to "approved" if a strict majority of those calls say **yes**
  on *both* clarity and faithfulness. One lucky or unlucky single call
  can't end the loop early or keep it going forever.

Two more details borrowed from the paper:

- The judge is forced to answer in **strict JSON**, not free text (Section
  3.1.2, "Standardizing LLMs' Output Format"), so its yes/no answer and
  feedback sentence can be parsed reliably. Because real models sometimes
  wrap JSON in extra text anyway, `parse_judge_output()` has a few
  fallback strategies before giving up (Section 2.3.1).
- The judge's feedback (why it said no) is fed directly into the next
  rewrite attempt, so each round is a targeted fix for a specific problem
  rather than a blind re-roll.

## Setup

```bash
cd llm_judge
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install groq
```

Set your Groq API key (get one at https://console.groq.com/keys):

```bash
export GROQ_API_KEY=gsk_...
```

## Running it

```bash
# Run all 3 built-in example paragraphs through the refine/judge loop
python llm_as_a_judge.py

# Simplify your own paragraph instead
python llm_as_a_judge.py --text "Notwithstanding any provision herein..."

# More/fewer judge samples per round (default 3 -- must be an odd-ish
# number for a clean majority; 3 or 5 are good choices)
python llm_as_a_judge.py --n-judge-samples 5

# Allow more/fewer refinement rounds before giving up (default 5)
python llm_as_a_judge.py --max-rounds 8

# Use a different Groq-hosted model
python llm_as_a_judge.py --model llama-3.3-70b-versatile
```

### What the output looks like

For each example, you'll see the original paragraph, then every round of
the loop as it happens — the current draft, each of the judge's votes with
its reasoning, whether that round was approved, and (if not) the feedback
that gets fed into the next rewrite:

```
--- Round 1/5 ---
  draft: If you don't tell us in writing within 30 days after something happens...
    [judge #1] clear=True faithful=False  ('drops the "sole discretion" forfeiture clause')
    [judge #2] clear=True faithful=False  ('same issue, missing forfeiture language')
    [judge #3] clear=True faithful=True  ('')
  -> needs another round  (clear votes 3/3, faithful votes 1/3)
  feedback for next round: drops the "sole discretion" forfeiture clause

--- Round 2/5 ---
  draft: If you don't tell us in writing within 30 days...your claim may be
  ...
  -> APPROVED  (clear votes 3/3, faithful votes 3/3)
```

followed by the final plain-language paragraph for that example, and a
summary at the end reporting how many examples were approved, the average
number of rounds it took, and the total number of judge calls made.

## The examples it runs on

Three real-style dense insurance-policy paragraphs (`build_examples()` in
the code), each testing a different kind of jargon:

1. A **time-limit / forfeiture clause** ("shall be deemed forfeited absent
   a showing of good cause...") — easy to accidentally lose the forfeiture
   condition while simplifying.
2. A **pre-existing-condition exclusion** ("Coverage... is contingent
   upon...") — easy to simplify into something that sounds like broader
   coverage than the original actually promises.
3. A **liability-limit clause** with nested conditions ("the lesser of...
   less the applicable deductible...") — dense with numbers and
   conditionals that are easy to drop.

## A few design decisions worth knowing about

- **The judge checks two things, not one.** Checking only "is this
  clear?" would reward a simplifier that just deletes the hard parts.
  Checking both clarity and faithfulness, and requiring both to pass,
  keeps the loop honest.
- **Majority vote, not a single call, decides approval.** `judge_round()`
  tallies votes separately for "clear" and "faithful" and only approves if
  a strict majority (`n_samples // 2 + 1`) agree on both — see the doc
  comment on that function for the exact logic.
- **A call that can't be parsed doesn't silently count as a "no".** If the
  judge's JSON can't be extracted at all, that vote is recorded but
  doesn't count toward either total — see `parse_judge_output()` returning
  `(None, None, None)` and how `judge_round()` only increments a counter
  when the value is exactly `True`.
- **The loop always terminates.** If the judge never approves, the loop
  stops after `--max-rounds` and returns the best attempt so far, clearly
  labeled as not approved rather than silently pretending success.
- **JSON mode on the judge call.** The Groq API accepts a
  `response_format={"type": "json_object"}` parameter that constrains the
  model's output to valid JSON — this is used on the judge call as an
  extra layer on top of the prompt instruction, and is the same
  "standardize the output format" idea the paper describes (Section
  3.1.2). The rewriter call doesn't use it since it's meant to return
  plain prose, not JSON.
- **Model choice affects reliability.** `openai/gpt-oss-120b` is used by
  default for its strong instruction-following, which matters both for
  producing faithful rewrites and for the judge reliably returning valid
  JSON. Smaller Groq models may need more of the fallback parsing paths
  in `parse_judge_output()` to kick in.

## File layout

```
llm_judge/
├── llm_as_a_judge.py   # the whole implementation (single file, per the assignment)
└── README.md           # this file
```
