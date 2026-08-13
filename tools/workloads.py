#!/usr/bin/env python3
"""
workloads.py — named workload profiles with different prefix-sharing shapes.

WHY THIS EXISTS
---------------
The finding so far is: llama-server routes by longest-common-prefix similarity,
and for a multi-tenant *agent* the shared preamble makes any two tenants ~0.72
similar, so the 0.10 default lets every slot match every tenant.

That argument has a falsifiable consequence which one workload cannot test.
Inter-tenant similarity is approximately

    shared_preamble_tokens / total_prompt_tokens

so it is a property of PROMPT SHAPE, not of llama.cpp. Change the shape and the
threshold that separates tenants must move with it. Specifically:

  agent   large shared preamble, small per-tenant text  -> similarity HIGH
          the default fails badly, and a high threshold is required

  rag     small shared preamble, large per-tenant doc   -> similarity LOW
          tenants are already distinguishable, so the default may be adequate
          and a high threshold buys little

If that holds, "set it to 0.9" is the wrong lesson. The right lesson is that the
correct threshold is a function of the workload, and it can be measured.

If it does NOT hold — if the cliff sits in the same place for both shapes — the
similarity model is wrong and this repository should say so.

Each profile supplies:
    system(...)              the shared preamble, byte-identical per tenant
    turn(tenant, turn_idx)   that tenant's text for a turn
    seed(tenant)             per-tenant content injected before turn 1

Consumed by bench_agent.run(..., workload=...).
"""

from __future__ import annotations

import json

# --------------------------------------------------------------------------- #
# A · multi-tenant agent  (the original workload; flagship result)
# --------------------------------------------------------------------------- #

AGENT_TOOLS = [
    {"name": "search_flights",
     "description": "Search available flights between two airports on a date.",
     "parameters": {"origin": "IATA code", "destination": "IATA code",
                    "date": "YYYY-MM-DD", "passengers": "integer",
                    "cabin": "economy|premium|business|first",
                    "max_stops": "integer", "refundable": "boolean"}},
    {"name": "book_hotel",
     "description": "Reserve a hotel room in a city for a date range.",
     "parameters": {"city": "string", "checkin": "YYYY-MM-DD",
                    "checkout": "YYYY-MM-DD", "guests": "integer",
                    "rooms": "integer", "star_rating": "integer",
                    "breakfast": "boolean", "cancellable": "boolean"}},
    {"name": "get_weather",
     "description": "Weather forecast for a city on a date.",
     "parameters": {"city": "string", "date": "YYYY-MM-DD", "units": "c|f"}},
    {"name": "convert_currency",
     "description": "Convert an amount between two currencies.",
     "parameters": {"amount": "number", "from": "ISO code", "to": "ISO code"}},
    {"name": "create_calendar_event",
     "description": "Create an event in the user's calendar.",
     "parameters": {"title": "string", "start": "ISO8601", "end": "ISO8601",
                    "location": "string", "attendees": "list of emails",
                    "reminder_minutes": "integer"}},
]

AGENT_SYSTEM = (
    "You are a meticulous travel planning agent operating in a tool-calling "
    "loop. On each turn you inspect the conversation so far, decide whether a "
    "tool call is required, and if so emit exactly one JSON object and nothing "
    "else. Never invent tool results. Never call a tool you have already called "
    "with identical arguments. If every necessary fact is known, produce a final "
    "answer instead of a tool call. Always prefer refundable options when the "
    "user has not stated a preference. Treat all dates as ISO-8601. "
    "Available tools:\n" + json.dumps(AGENT_TOOLS, indent=2) + "\n"
    "Respond with a single JSON object of the form "
    '{"tool": "<name>", "arguments": {...}} or {"final": "<answer>"}.'
)

AGENT_TRIPS = [
    ("Delhi", "Singapore", "2026-09-14", "two", "DEL", "SIN"),
    ("London", "Tokyo", "2026-10-02", "one", "LHR", "HND"),
    ("Mumbai", "Dubai", "2026-08-30", "four", "BOM", "DXB"),
    ("San Francisco", "Seoul", "2026-11-11", "three", "SFO", "ICN"),
    ("Berlin", "Reykjavik", "2026-12-01", "one", "BER", "KEF"),
    ("Sydney", "Auckland", "2026-09-05", "two", "SYD", "AKL"),
    ("Toronto", "Lisbon", "2026-10-19", "five", "YYZ", "LIS"),
    ("Nairobi", "Amsterdam", "2026-11-27", "two", "NBO", "AMS"),
]

AGENT_FLAVOR = [
    "I am travelling for a wedding and have a strict budget.",
    "I need wheelchair accessible options throughout.",
    "This is a business trip; prioritise schedule over cost.",
    "I am travelling with a toddler and need family facilities.",
    "I am a vegetarian and need meal options confirmed.",
    "I have a tight connection and cannot risk delays.",
    "I am afraid of flying and prefer daytime departures.",
    "I am a frequent flyer and want lounge access noted.",
]


def _agent_turn(tenant: int, turn: int) -> str:
    o, d, date, pax, oi, di = AGENT_TRIPS[tenant % len(AGENT_TRIPS)]
    t = [
        f"Plan a trip: {o} to {d} on {date} for {pax} passengers, economy.",
        f"Now find a hotel in {d} near {di} for two nights from {date}.",
        f"What will the weather in {d} be like when we land on {date}?",
        f"Convert the running total for this {o}-{d} trip into rupees.",
        f"Add the {oi} to {di} departure to my calendar with a 3 hour reminder.",
        f"Summarise the full {o} to {d} itinerary for {pax} travellers.",
    ]
    return t[turn % len(t)]


# --------------------------------------------------------------------------- #
# B · RAG  (the contrast: the shared part is SMALL, the private part is LARGE)
# --------------------------------------------------------------------------- #
#
# The system prompt is deliberately short — a retrieval assistant needs
# instructions, not tool schemas. The bulk of each tenant's prompt is its own
# retrieved document, which no other tenant shares. That inverts the ratio the
# agent workload has.

RAG_SYSTEM = (
    "You are a retrieval-grounded assistant. Answer only from the provided "
    "document. If the document does not contain the answer, say so plainly "
    "rather than guessing. Quote the relevant passage before answering. Keep "
    "answers under four sentences."
)

# Each tenant gets a distinct ~600-token "retrieved document". Content is
# synthetic but structurally realistic: a title, sections, and dense prose with
# tenant-specific nouns so no two tenants share a long prefix.
_RAG_SUBJECTS = [
    ("Coastal Erosion in the Baltic", "sediment transport", "Gdansk",
     "shoreline retreat", "1974", "groyne fields"),
    ("Perovskite Tandem Photovoltaics", "carrier recombination", "Lausanne",
     "bandgap tuning", "2019", "encapsulation stacks"),
    ("Antarctic Subglacial Hydrology", "basal meltwater", "Thwaites",
     "ice-stream velocity", "1998", "borehole arrays"),
    ("Post-Quantum Lattice Signatures", "module learning-with-errors", "Bochum",
     "signature compression", "2022", "rejection sampling"),
    ("Mycorrhizal Nutrient Exchange", "phosphorus transfer", "Uppsala",
     "hyphal networks", "1987", "isotope tracers"),
    ("High-Entropy Alloy Creep", "dislocation pinning", "Sendai",
     "grain-boundary sliding", "2015", "nanoindentation"),
    ("Urban Heat Island Mitigation", "albedo modification", "Phoenix",
     "canopy coverage", "2003", "pyranometer grids"),
    ("Cephalopod Chromatophore Control", "radial muscle actuation", "Naples",
     "pattern latency", "1991", "electrophysiology"),
]


def _rag_document(tenant: int) -> str:
    title, mech, place, metric, year, method = _RAG_SUBJECTS[tenant % len(_RAG_SUBJECTS)]
    paras = []
    paras.append(
        f"## {title}\n\n"
        f"Field measurements collected near {place} since {year} indicate that "
        f"{mech} is the dominant control on {metric}. Earlier surveys attributed "
        f"the variation to seasonal forcing, but repeated campaigns using "
        f"{method} showed that the seasonal component accounts for less than a "
        f"fifth of the observed variance."
    )
    paras.append(
        f"### Method\n\n"
        f"Instrumentation was deployed in three transects around {place}. Each "
        f"transect carried redundant sensors so that a single failure would not "
        f"invalidate a season of data. Calibration against the {year} reference "
        f"series was repeated at the start and end of every campaign, and drift "
        f"exceeding two percent triggered re-deployment rather than correction "
        f"in post-processing."
    )
    paras.append(
        f"### Findings\n\n"
        f"Across eleven campaigns the relationship between {mech} and {metric} "
        f"was monotonic but distinctly non-linear, with a threshold above which "
        f"{metric} responded sharply. Below that threshold the response was "
        f"indistinguishable from measurement noise. Attempts to fit a single "
        f"linear model across the full range produced residuals structured by "
        f"season, which is the signature of a missing term rather than of "
        f"random error."
    )
    paras.append(
        f"### Limitations\n\n"
        f"All measurements come from a single region, and {place} is not "
        f"representative of settings where {method} cannot be deployed. The "
        f"{year} reference series has known gaps. No causal claim is made about "
        f"{mech} outside the observed range, and the threshold location is "
        f"reported with wide uncertainty because campaigns were not designed to "
        f"resolve it."
    )
    return "\n\n".join(paras)


_RAG_QUESTIONS = [
    "What does the document identify as the dominant control, and on what evidence?",
    "Summarise the calibration procedure and why drift triggered re-deployment.",
    "Is the relationship described as linear? Quote the passage that says so.",
    "What limitations does the document state about geographic generality?",
    "Which year is used as the reference series, and what is said about its gaps?",
    "Does the document make a causal claim outside the observed range?",
]


def _rag_turn(tenant: int, turn: int) -> str:
    return _RAG_QUESTIONS[turn % len(_RAG_QUESTIONS)]


def _rag_seed(tenant: int) -> str:
    return ("Here is the retrieved document for this session.\n\n"
            + _rag_document(tenant))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

class Workload:
    """One prompt shape. `expect` records the prediction BEFORE measuring."""

    def __init__(self, name, system, turn_fn, seed_fn, flavor, expect):
        self.name = name
        self.system = system
        self.turn = turn_fn
        self.seed = seed_fn
        self.flavor = flavor
        self.expect = expect


WORKLOADS = {
    "agent": Workload(
        "agent", AGENT_SYSTEM, _agent_turn,
        lambda t: AGENT_FLAVOR[t % len(AGENT_FLAVOR)],
        AGENT_FLAVOR,
        "Large shared preamble, small private text. Inter-tenant similarity "
        "should be HIGH (~0.7), so the 0.10 default should fail badly and the "
        "usable threshold should sit near 0.8.",
    ),
    "rag": Workload(
        "rag", RAG_SYSTEM, _rag_turn, _rag_seed,
        ["Session start."] * 8,
        "Small shared preamble, large private document. Inter-tenant "
        "similarity should be LOW, so tenants are already distinguishable and "
        "the 0.10 default may be adequate. If the cliff lands in the same "
        "place as the agent workload, the similarity model is wrong.",
    ),
}


def get(name: str) -> Workload:
    if name not in WORKLOADS:
        raise SystemExit(f"unknown workload '{name}'. "
                         f"choose one of: {', '.join(WORKLOADS)}")
    return WORKLOADS[name]


if __name__ == "__main__":
    import sys
    # Every entry point in this project reconfigures stdio: the default Windows
    # console is cp1252 and will crash on the first non-ASCII character.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    # Character-level shape check. Token counts come from the server at run
    # time; this is only a sanity check that the two shapes really differ.
    print()
    print("  workload  system(ch)  seed(ch)  turn(ch)   shared/total")
    print("  " + "-" * 58)
    for n, w in WORKLOADS.items():
        sysn = len(w.system)
        seed = len(w.seed(0))
        turn = len(w.turn(0, 0))
        ratio = sysn / (sysn + seed + turn)
        print(f"  {n:<9} {sysn:>10} {seed:>9} {turn:>9}   {ratio:>10.2f}")
    print()
    print("  'shared/total' approximates inter-tenant similarity: the fraction")
    print("  of one tenant's prompt that every OTHER tenant also sends.")
    print("  A high ratio is what makes the 0.10 default fail.")
    print()
    for n, w in WORKLOADS.items():
        print(f"  {n}: {w.expect}")
        print()
