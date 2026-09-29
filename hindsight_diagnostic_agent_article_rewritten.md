# How Hindsight Stopped My Diagnostic Agent From Hallucinating Repairs

A maintenance-agent test exposed a grounding problem: it cited a work order that did not exist, including a technician name and part number. The response looked like a genuine CMMS record, but the model had generated it because the pipeline did not verify the reference.

Retrieval-based agents can blur evidence with plausible patterns. I tried two fixes; the first improved the output, but did not enforce the required boundary.

## What the system does

The agent connects technicians with plant maintenance history. A symptom is checked against earlier events on the same or related equipment. Relevant history can identify failed parts, unsuccessful repairs, successful fixes, and recurring causes; missing history is reported rather than guessed.

Each repair becomes retained memory containing warning signals, the failure, attempted repairs, the successful fix, and the systemic cause.

The updated `demo_agent.py` calls Hindsight's `reflect()` with `RESPONSE_SCHEMA` and `include_facts=True`. It uses retained memories, machine-specific Mental Models, and plant directives to produce triage, root cause, actions, and supporting citations.

The key requirement is traceability: each claim must point to real evidence.

## The technical story: grounding as a filter, not a prompt

The first version relied on a prompt requiring citations from retrieved memory, then accepted the model's output. Unsupported references could still pass because no application-level validation existed.

A stricter prompt helped only slightly. Instructions encourage grounded behavior, but they do not guarantee that the model will avoid inventing plausible identifiers.

The stronger fix had to happen in the application pipeline.

The revised pipeline receives the response and supporting evidence from `reflect()`. It builds the valid record-ID set and compares every generated citation against it.

```python
def check_before_repair(symptoms, lang="English", include_live=True):
    # ...

    r, err = reflect(
        query,
        context=ctx,
        response_schema=RESPONSE_SCHEMA
    )

    # ...

    facts = (r.based_on.memories if r.based_on else None) or []
    mental_models = (
        (r.based_on.mental_models if r.based_on else None) or []
    )

    texts = [f.text for f in facts]
    id_blob = texts + [f.context for f in facts if f.context]

    if machine and any(
        m.id == twin_id(machine)
        for m in mental_models
    ):
        id_blob += [
            f"[{rid}] machine {machine}"
            for rid in BY_ID
            if BY_ID[rid]["machine_id"] == machine
        ]

    srcs = sources_from(id_blob)
    cross = cross_fleet_from(id_blob, machine)

    allowed = (
        {s["id"] for s in srcs}
        | {c["id"] for c in cross}
        | {x["id"] for x in led}
    )

    cites = [
        c for c in (a.get("citations") or [])
        if isinstance(c, str)
        and any(c.startswith(i) for i in allowed)
    ]

    a["citations"] = [
        next(i for i in allowed if c.startswith(i))
        for c in cites
    ]

    return {
        "agent": a,
        "confidence": conf,
        "sources": srcs,
        ...
    }
```

Each citation is checked against identifiers present in retrieved evidence. Invented identifiers are removed before the technician receives the response.

Grounding becomes an enforced application rule rather than a model request.

## The dead end: recall without a budget is just a bigger prompt

Citation filtering exposed another issue: retrieval could also be too broad.

The first recall implementation had no result cap or token budget. A query could return many irrelevant fragments, and the model would still reason over them.

Thus, citation validation could succeed while the diagnosis remained poorly scoped. Real records are not automatically relevant records.

The updated `recall()` adds a token budget and result limit:

```python
def recall(query, machine=None, n=8):
    q = (
        f"{machine} {query}"
        if machine and machine not in query.upper()
        else query
    )

    r, e = call_with_retry(
        lambda: hs().recall(
            bank_id=BANK_ID,
            query=q,
            max_tokens=1500,
            budget="low"
        )
    )

    if e:
        return [], friendly_error(e)

    return [
        x.text
        for x in (getattr(r, "results", None) or [])[:n]
    ], None
```

Recall is limited to `max_tokens=1500` and eight results by default.

Grounding means validating references and controlling noisy context.

## Making confidence mean something

Confidence was the next issue.

Early versions displayed a fixed percentage for every diagnosis. It measured nothing and could make unsupported results look authoritative.

I replaced it with a `confidence()` function based on retrieval signals.

```python
def confidence(query, texts):
    ids = set(MCH_RE.findall(query.upper()))

    if ids and not (ids & KNOWN):
        return {
            "score": 0,
            "label": "0% match",
            "type": "Unknown machine: no fleet history"
        }

    if not texts:
        return {
            "score": 0,
            "label": "0% match",
            "type": "No matching memory"
        }

    blob = " ".join(texts)
    rec_m = set(MCH_RE.findall(blob))
    recs = {
        i for i in ID_RE.findall(blob)
        if i in BY_ID
    }

    score = 25

    if ids and ids & rec_m:
        score += 40
        kind = "Same-machine recurrence"
    elif ids:
        score += 5
        kind = "Weak: machine not in recalled memory"
    else:
        score += 20
        kind = "Symptom / cross-fleet correlation"

    score += min(20, len(recs) * 7)

    dates = sorted(r["date"] for r in HISTORY)

    if any(BY_ID[i]["date"] >= dates[-4] for i in recs):
        score += 10

    score = min(95, score)

    return {
        "score": score,
        "label": f"{score}% match",
        "type": kind
    }
```

The function checks whether the machine is known, whether it appears in memory, how many records support the answer, and how recent they are.

Confidence cannot be borrowed from an unrelated asset. `MCH-999` receives a 0% match and is identified as having no fleet history.

## Where recall alone wasn't enough

The comparison mode runs the same question with and without memory using identical prompts, showing what the retained evidence contributes.

Without memory, the symptom produces broad suggestions such as checking alignment, imbalance, or bearing wear. Those may be reasonable, but they do not describe this machine's history.

With memory, the query surfaces the earlier drive-end bearing seizure under `MAINT-2025-001`, the unsuccessful regreasing attempt, and the root cause under `MAINT-2025-002`: the lubrication interval changed from monthly to quarterly after a maintenance software update.

This separates generic symptom matching from evidence about the actual asset.

## Results, in concrete terms

Two cases show the difference:

- `MCH-017` suffered the same underlying bearing problem twice six months apart. During the second incident, the agent surfaces the earlier failed repair before it is repeated.

- `MCH-999` has no history in the memory bank. The system reports a 0% match, assigns a novel triage level, and recommends the OEM procedure and outcome logging instead of inventing a diagnosis.

Neither case required a smarter model. The surrounding system verified the response against the evidence actually stored.

## Lessons

**A grounding bug can hide behind a valid citation.** A citation can point to a real record while retrieval still supplies poor context. "Real" and "relevant" are different properties.

**Specific memory is more useful than a generic label.** Knowing that regreasing failed tells the next technician what not to repeat. A documented lubrication-schedule change is also more actionable than a broad "bearing wear" description.

**"I don't know" is a feature, not a failure state.** When a machine has no history, 0% confidence is preferable to treating unrelated memory as evidence.

**Recall and reasoning are different jobs.** One problem came from the evidence recall supplied; another came from how the model used it. Separating those stages made both easier to test.

These changes do not make the model smarter. They make the system clearer about what the model generated and what memory supports. For maintenance work, that traceability is what makes the agent useful when the same problem appears again.
