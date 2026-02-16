from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import xarray as xr

_SYNONYMS = {
    "ocean": "sea",
    "sea": "ocean",
    "temp": "temperature",
}

_DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "ocean": {"sea", "ocean", "salinity", "marine", "sea_water"},
    "atmosphere": {"air", "atmosphere", "aerosol", "cloud", "wind"},
    "land": {"soil", "land", "terrestrial", "vegetation", "canopy"},
    "cryosphere": {"ice", "snow", "glacier", "sea_ice"},
    "biogeochemistry": {"ph", "alkalinity", "nitrate", "oxygen", "chlorophyll"},
}


@dataclass(frozen=True)
class StandardNameEntry:
    name: str
    canonical_units: str


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _tokenize_name(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", _normalize_name(text).replace("_", " ")))
    expanded: set[str] = set(tokens)
    for token in list(tokens):
        synonym = _SYNONYMS.get(token)
        if synonym:
            expanded.add(synonym)
    return expanded


@lru_cache(maxsize=8)
def _load_standard_name_entries(xml_path: str) -> tuple[StandardNameEntry, ...]:
    parsed = urlparse(xml_path)
    if parsed.scheme in {"http", "https", "file"}:
        with urlopen(xml_path) as response:
            payload = response.read()
        root = ET.fromstring(payload)
    else:
        tree = ET.parse(xml_path)
        root = tree.getroot()

    entries: dict[str, str] = {}
    for entry in root.findall(".//entry"):
        name = entry.attrib.get("id")
        canonical_units = entry.findtext("canonical_units", default="").strip()
        if name:
            entries[name.strip()] = canonical_units

    for alias in root.findall(".//alias"):
        alias_name = alias.attrib.get("id")
        entry_id = alias.findtext("entry_id", default="").strip()
        if alias_name and entry_id in entries:
            entries[alias_name.strip()] = entries[entry_id]

    return tuple(
        StandardNameEntry(name=key, canonical_units=value)
        for key, value in sorted(entries.items())
    )


def _units_compatible(actual: str | None, expected: str | None) -> bool:
    if actual is None or expected is None:
        return False

    a = _normalize_name(actual)
    e = _normalize_name(expected)
    if a == e:
        return True

    synonyms = {
        "k": {"kelvin"},
        "kelvin": {"k"},
        "degrees_celsius": {"celsius", "degc", "degree_celsius"},
        "celsius": {"degrees_celsius", "degc", "degree_celsius"},
        "1": {"1.0", "dimensionless"},
    }
    if e in synonyms and a in synonyms[e]:
        return True
    if a in synonyms and e in synonyms[a]:
        return True
    return False


def _best_standard_name_candidates(
    var_name: str,
    long_name: str | None,
    entries: tuple[StandardNameEntry, ...],
    top_n: int = 3,
    domain: str | None = None,
) -> list[StandardNameEntry]:
    query = _tokenize_name(var_name)
    if long_name:
        query |= _tokenize_name(long_name)
    if not query:
        return []

    domain_tokens = _DOMAIN_KEYWORDS.get(_normalize_name(domain or ""), set())

    scored: list[tuple[float, StandardNameEntry]] = []
    for entry in entries:
        cand_tokens = _tokenize_name(entry.name)
        if not cand_tokens:
            continue
        overlap = len(query & cand_tokens)
        if overlap == 0:
            continue
        score = overlap / max(len(cand_tokens), 1)
        if entry.name in var_name:
            score += 0.2
        if domain_tokens and (cand_tokens & domain_tokens):
            score += 0.3
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for score, entry in scored[:top_n] if score >= 0.4]


def augment_issues_with_standard_name_suggestions(
    ds: xr.Dataset,
    issues: dict[str, Any],
    standard_name_xml: str | None,
    domain: str | None = None,
) -> None:
    if not standard_name_xml:
        return

    try:
        entries = _load_standard_name_entries(standard_name_xml)
    except Exception as exc:
        issues["notes"].append(
            f"Could not read CF standard-name table '{standard_name_xml}': {type(exc).__name__}: {exc}"
        )
        return

    suggestions = issues.setdefault("suggestions", {})
    var_suggestions = suggestions.setdefault("variables", {})
    table_lookup = {entry.name: entry for entry in entries}

    for var_name, da in ds.data_vars.items():
        var_name = str(var_name)
        standard_name = da.attrs.get("standard_name")
        units = da.attrs.get("units")
        long_name = da.attrs.get("long_name")

        if standard_name is None:
            candidates = _best_standard_name_candidates(
                var_name,
                long_name,
                entries,
                domain=domain,
            )
            if candidates:
                var_suggestions[var_name] = {
                    "recommended_standard_names": [entry.name for entry in candidates],
                    "recommended_units": candidates[0].canonical_units or None,
                    "domain": domain,
                }
            continue

        entry = table_lookup.get(str(standard_name))
        if entry is None:
            var_suggestions[var_name] = {
                "unknown_standard_name": str(standard_name),
            }
            continue

        if units is None:
            var_suggestions[var_name] = {
                "units_check": {
                    "status": "missing",
                    "expected_units": entry.canonical_units or None,
                }
            }
            continue

        if entry.canonical_units and not _units_compatible(str(units), entry.canonical_units):
            var_suggestions[var_name] = {
                "units_check": {
                    "status": "mismatch",
                    "current_units": str(units),
                    "expected_units": entry.canonical_units,
                }
            }
