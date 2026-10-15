"""The country registry: which countries exist and where their config lives.

One place decides what the country menu offers and which files a country is
scored from, so adding Japan is an entry in config/countries.yml plus its
config files, with no code change here or in the pages.

A country is "live" when its three config files exist. Anything else is
"planned": named on the page, not selectable, so the menu can never lead to an
empty dashboard.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REGISTRY = "config/countries.yml"
REQUIRED_PATHS = ("indicators", "regimes", "sources")


def load_registry(path: str | Path = REGISTRY) -> dict:
    with open(path) as f:
        reg = yaml.safe_load(f)
    if not reg.get("countries"):
        raise ValueError(f"{path} lists no countries")
    return reg


def _spec(reg: dict, code: str) -> dict:
    try:
        return reg["countries"][code]
    except KeyError:
        raise KeyError(f"unknown country {code!r}; "
                       f"known: {', '.join(reg['countries'])}") from None


def is_live(reg: dict, code: str, root: str | Path = ".") -> bool:
    """Live means declared live and every config file is actually present."""
    spec = _spec(reg, code)
    if str(spec.get("status", "planned")) != "live":
        return False
    return all(spec.get(k) and (Path(root) / spec[k]).exists() for k in REQUIRED_PATHS)


def live_codes(reg: dict, root: str | Path = ".") -> list[str]:
    return [c for c in reg["countries"] if is_live(reg, c, root)]


def planned(reg: dict, root: str | Path = ".") -> list[dict]:
    """Countries named in the registry that cannot be scored yet."""
    return [{"code": c, **reg["countries"][c]}
            for c in reg["countries"] if not is_live(reg, c, root)]


def default_code(reg: dict, root: str | Path = ".") -> str:
    """The country a reader lands on: the declared default if it is live."""
    live = live_codes(reg, root)
    if not live:
        raise ValueError("no country in the registry has all its config files")
    wanted = str(reg.get("default", "")) or live[0]
    return wanted if wanted in live else live[0]


def frameworks(reg: dict, code: str, root: str | Path = ".") -> dict[str, dict]:
    """The frameworks a country can be scored under, default first: each a set
    of signposts and the scenarios placed on them. Only those whose files exist
    are offered. A country without a `frameworks` entry has one, from its own
    indicators and regimes paths."""
    spec = _spec(reg, code)
    listed = spec.get("frameworks") or {
        "default": {"label": "", "indicators": spec.get("indicators"), "regimes": spec.get("regimes")}}
    return {k: v for k, v in listed.items()
            if v.get("indicators") and v.get("regimes")
            and (Path(root) / v["indicators"]).exists() and (Path(root) / v["regimes"]).exists()}


def resolve(reg: dict, code: str | None = None, root: str | Path = ".",
            strict: bool = False, framework: str | None = None) -> dict:
    """Everything a page needs for one country: label and the three paths.

    strict=False falls back to the default country, which is what a page wants
    when a stale session or bookmark names a country that is no longer live.
    strict=True raises instead, which is what a command line wants: asking for
    Japan and silently getting the United States is worse than an error.

    `framework` picks which of the country's frameworks supplies the
    indicators and regimes; an unknown or missing one falls back to the
    default, the first listed. `all_indicators` lists every framework's
    indicator file, for ingest to fetch the series all of them need.
    """
    code = code or default_code(reg, root)
    if code not in live_codes(reg, root):
        if strict:
            status = _spec(reg, code).get("status", "planned")
            raise ValueError(
                f"country {code!r} is {status}, not live: its config files are not in place. "
                f"Live: {', '.join(live_codes(reg, root))}")
        code = default_code(reg, root)
    spec = _spec(reg, code)
    fws = frameworks(reg, code, root)
    default_fw = next(iter(fws))
    fw = framework if framework in fws else default_fw
    paths = {k: spec[k] for k in REQUIRED_PATHS}
    paths.update({k: fws[fw][k] for k in ("indicators", "regimes")})
    # Forward slashes even on Windows: these paths are shown on the page and
    # quoted in the methodology, where a backslash would be wrong for everyone
    # reading the repository.
    return {"code": code,
            "label": spec.get("label", code.upper()),
            "short": spec.get("short", code.upper()),
            "framework": fw, "default_framework": default_fw,
            "frameworks": {k: v.get("label", k) for k, v in fws.items()},
            "all_indicators": list(dict.fromkeys(
                (Path(root) / v["indicators"]).as_posix() for v in fws.values())),
            **{k: (Path(root) / v).as_posix() for k, v in paths.items()}}


def label(reg: dict, code: str) -> str:
    return _spec(reg, code).get("label", code.upper())
