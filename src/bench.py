"""The indicator bench: a candidate pool to argue about, not a model input.

Nothing here is scored. `config/bench.yml` holds one row per indicator, live or
proposed, and this module turns it into the counts and diffs a discussion needs:
how many indicators a driver would carry, how far ahead the selection looks, and
how much of it needs a vendor we do not have yet.

The live rows carry the same `id` as `config/indicators.yml`, so a test can
check the bench has not drifted from the set it claims to describe.
"""

from __future__ import annotations

import yaml

HORIZON_ORDER = ("short", "medium", "long")
HORIZON_SHORT = {"short": "ST", "medium": "MT", "long": "LT"}
HORIZON_LABEL = {"short": "Short", "medium": "Medium", "long": "Long"}

# The second axis. Horizon says how far ahead an indicator looks; nature says
# what kind of thing it measures, and the two come apart exactly where the set
# is weakest. A valuation or a risk premium can look a decade ahead and still
# mean-revert inside a cycle, so "long-horizon" flatters a set that carries
# almost nothing structural. Keeping both makes that visible instead of
# arguable.
# The third axis: who produced the number. A driver built from one kind of
# source inherits that source's blind spot whatever its horizon — five BLS
# releases disagree about very little. Inflation expectations works because a
# market price, a model and two surveys can disagree, and the disagreement is
# the signal.
KIND_ORDER = ("market", "model", "survey", "official", "company")
KIND_SHORT = {"market": "MKT", "model": "MODEL", "survey": "SURVEY",
              "official": "OFFICIAL", "company": "COMPANY"}
KIND_LABEL = {"market": "Market price", "model": "Model estimate",
              "survey": "Survey", "official": "Official statistic",
              "company": "Company financials"}
KIND_HELP = {
    "market": "A traded price. Never revised, available daily, and it is someone's money.",
    "model": "An estimator's output. Revises when the model is re-run.",
    "survey": "Someone was asked. Captures beliefs, including wrong ones.",
    "official": "A statistical agency's count or census of what happened.",
    "company": "Read off corporate financial statements.",
}

NATURE_ORDER = ("news", "cyclical", "structural")
NATURE_SHORT = {"news": "NEWS", "cyclical": "CYC", "structural": "STR"}
NATURE_LABEL = {"news": "News", "cyclical": "Cyclical", "structural": "Structural"}
NATURE_HELP = {
    "news": "What just happened. Turns in weeks and mean-reverts fast.",
    "cyclical": "Where output sits relative to capacity. Oscillates over one to three years.",
    "structural": "What capacity is, and where it is drifting. Does not mean-revert "
                  "inside a cycle — this is where the Solow terms live.",
}

# Where a row can come from, in order of how much work it is to get.
SOURCE_LABEL = {
    "macrobond": "Macrobond",
    "macrobond_check": "Macrobond, code unconfirmed",
    "external": "New source",
}
SOURCE_SHORT = {"macrobond": "MB", "macrobond_check": "MB?", "external": "EXT"}

# A driver a reader can hold in their head. Same bounds the config test enforces.
MIN_PER_DRIVER, MAX_PER_DRIVER = 3, 6


def load(path: str = "config/bench.yml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def drivers(bench: dict) -> dict:
    return bench.get("drivers") or {}


def rows(bench: dict) -> list[dict]:
    """Every row, flattened, each carrying the driver it belongs to."""
    out = []
    for key, block in drivers(bench).items():
        for item in block.get("items") or []:
            out.append({**item, "driver": key, "driver_label": block.get("label", key)})
    return out


def in_set_ids(bench: dict) -> list[str]:
    return [r["id"] for r in rows(bench) if r.get("status") == "in_set"]


def by_driver(bench: dict, driver: str) -> list[dict]:
    block = drivers(bench).get(driver) or {}
    return list(block.get("items") or [])


def horizon_mix(items: list[dict]) -> dict:
    """Share of a selection by horizon, counted per indicator.

    By count rather than by weight: candidates have no weight, and the question
    in the room is how many of each kind, not how the weights would land.
    """
    total = len(items) or 1
    mix = {h: 0.0 for h in HORIZON_ORDER}
    for it in items:
        mix[it.get("horizon", "medium")] += 1 / total
    return mix


def nature_mix(items: list[dict]) -> dict:
    """Share of a selection by what it measures, counted per indicator."""
    total = len(items) or 1
    mix = {n: 0.0 for n in NATURE_ORDER}
    for it in items:
        mix[it.get("nature", "cyclical")] += 1 / total
    return mix


def kinds_present(items: list[dict]) -> list[str]:
    """Which source types a selection draws on, in a stable order."""
    have = {i.get("kind", "official") for i in items}
    return [k for k in KIND_ORDER if k in have]


def flattered(items: list[dict]) -> list[dict]:
    """Rows tagged long-horizon that are cyclical in nature.

    These are the ones that made the set look longer-dated than it is: the
    horizon tag was widened without the content changing.
    """
    return [i for i in items
            if i.get("horizon") == "long" and i.get("nature") != "structural"]


def source_mix(items: list[dict]) -> dict:
    mix = {k: 0 for k in SOURCE_LABEL}
    for it in items:
        mix[it.get("where", "macrobond_check")] = mix.get(it.get("where", "macrobond_check"), 0) + 1
    return mix


def tally(bench: dict, selected: set[str]) -> list[dict]:
    """Per driver: what the selection would look like, and whether it is legal."""
    out = []
    for key, block in drivers(bench).items():
        items = [i for i in (block.get("items") or []) if i["id"] in selected]
        live = [i for i in (block.get("items") or []) if i.get("status") == "in_set"]
        n = len(items)
        out.append({
            "driver": key,
            "label": block.get("label", key),
            "n": n,
            "was": len(live),
            "ok": MIN_PER_DRIVER <= n <= MAX_PER_DRIVER,
            "horizons": horizon_mix(items),
            "natures": nature_mix(items),
            "structural": sum(1 for i in items if i.get("nature") == "structural"),
            "was_structural": sum(1 for i in live if i.get("nature") == "structural"),
            "kinds": kinds_present(items),
            "was_kinds": kinds_present(live),
            "sources": source_mix(items),
            "new_sources": sum(1 for i in items if i.get("where") == "external"),
        })
    return out


def diff(bench: dict, selected: set[str]) -> dict:
    """What the selection changes against the set that is actually scored."""
    added, dropped = [], []
    for r in rows(bench):
        live = r.get("status") == "in_set"
        picked = r["id"] in selected
        if picked and not live:
            added.append(r)
        elif live and not picked:
            dropped.append(r)
    return {"added": added, "dropped": dropped}


def source_note(row: dict) -> str:
    """'Macrobond · PCEC96' or 'New source · CIQ'."""
    where = row.get("where", "macrobond_check")
    label = SOURCE_LABEL.get(where, where)
    detail = row.get("code") if where != "external" else row.get("vendor")
    return f"{label} · {detail}" if detail else label


def summary(bench: dict, selected: set[str]) -> str:
    """A plain-text record of what was decided, to paste into notes.

    A meeting that leaves no artefact has to be held again, so the bench ends
    with something copyable rather than a screenshot.
    """
    lines = ["INDICATOR BENCH — proposed set", ""]
    for block in tally(bench, selected):
        flag = "" if block["ok"] else f"  <-- outside {MIN_PER_DRIVER}-{MAX_PER_DRIVER}"
        lines.append(f"{block['label']}  ({block['n']}, was {block['was']}){flag}")
        lines.append(f"  structural {block['structural']} of {block['n']}"
                     f"  (was {block['was_structural']} of {block['was']})")
        lines.append(f"  sources: {', '.join(block['kinds']) or 'none'}")
        for item in by_driver(bench, block["driver"]):
            if item["id"] not in selected:
                continue
            mark = " " if item.get("status") == "in_set" else "+"
            tag = HORIZON_SHORT.get(item.get("horizon", "medium"), "MT")
            nat = NATURE_SHORT.get(item.get("nature", "cyclical"), "CYC")
            knd = item.get("kind", "official")
            lines.append(f"  {mark} [{tag}/{nat:<4}/{knd:<8}] {item['name']}"
                         f"  ({source_note(item)})")
        lines.append("")

    picked = [r for r in rows(bench) if r["id"] in selected]
    d = diff(bench, selected)
    lines.append(f"CHANGES: {len(d['added'])} added, {len(d['dropped'])} dropped")
    for r in d["added"]:
        lines.append(f"  + {r['driver_label']}: {r['name']}  ({source_note(r)})")
    for r in d["dropped"]:
        lines.append(f"  - {r['driver_label']}: {r['name']}")

    mix = nature_mix(picked)
    lines += ["", f"WHOLE SET: {len(picked)} indicators — "
                  f"news {mix['news']:.0%}, cyclical {mix['cyclical']:.0%}, "
                  f"structural {mix['structural']:.0%}"]
    empty = [b["label"] for b in tally(bench, selected) if not b["structural"]]
    if empty:
        lines.append(f"  no structural content at all in: {', '.join(empty)}")
    single = [f"{b['label']} ({b['kinds'][0]} only)"
              for b in tally(bench, selected) if len(b["kinds"]) == 1]
    if single:
        lines.append(f"  built from one kind of source: {'; '.join(single)}")
    flat = flattered(picked)
    if flat:
        # Semicolons, because many of these names carry a comma of their own.
        lines.append(f"  tagged long-horizon but cyclical in nature: "
                     f"{'; '.join(r['name'] for r in flat)}")
    new_sources = sum(1 for r in picked if r.get("where") == "external")
    lines.append(f"  needs a source we do not have today: {new_sources}")
    return "\n".join(lines)
