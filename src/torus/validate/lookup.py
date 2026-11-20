"""Lookup validators (P7.2): PIN exists, PIN agrees with state/city, closed state list.

All checks are exact comparisons after presentation-only normalisation (case,
whitespace, "&" vs "and"). Nothing is fuzzy-matched or corrected: a mismatch
fails and the field goes to review. If the PIN master is not available the
validators fail closed (``lookup_unavailable``) so a PIN is never confirmed
unchecked. Reasons are codes only and never echo the value.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from torus.core.errors import ConfigError
from torus.core.models import ValidationContext
from torus.validate.base import RuleValidator, strip_separators

PIN_VALIDATOR_NAME = "indian_pin"
STATE_VALIDATOR_NAME = "indian_state"
PIN_LOCATION_VALIDATOR_NAME = "pin_matches_location"

PIN_MASTER_FILE = "india_post_pincodes.csv"
STATES_FILE = "india_states_uts.txt"
ALIASES_FILE = "india_state_aliases.csv"

_PIN_RE = re.compile(r"[1-9][0-9]{5}")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[.,]")


def lookups_dir() -> Path:
    """Repo ``resources/lookups`` (src/torus/validate/lookup.py -> parents[3])."""
    return Path(__file__).resolve().parents[3] / "resources" / "lookups"


def normalise_place(text: str) -> str:
    """Casefold, drop . and ,, collapse whitespace, unify ampersand with 'and'."""
    t = _PUNCT.sub(" ", text.casefold().replace("&", " and "))
    return _WS.sub(" ", t).strip()


@dataclass(frozen=True)
class PinEntry:
    """Everything the master lists for one PIN (several post offices possible)."""

    states: frozenset[str]
    districts: frozenset[str]
    offices: frozenset[str]


@dataclass(frozen=True)
class PinMaster:
    """PIN -> entry, with normalised place names."""

    entries: dict[str, PinEntry]

    def get(self, pin: str) -> PinEntry | None:
        return self.entries.get(pin)


@dataclass(frozen=True)
class StateList:
    """Closed state/UT list plus former-name aliases (all normalised)."""

    official: frozenset[str]
    aliases: dict[str, str]

    def canonical(self, name: str) -> str | None:
        """Normalised official name for *name*, or None if outside the list."""
        n = normalise_place(name)
        n = self.aliases.get(n, n)
        return n if n in self.official else None


def load_state_list(directory: Path | None = None) -> StateList:
    d = directory or lookups_dir()
    official: set[str] = set()
    for raw in (d / STATES_FILE).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            official.add(normalise_place(line))
    if not official:
        raise ConfigError("state list is empty")
    aliases: dict[str, str] = {}
    alias_path = d / ALIASES_FILE
    if alias_path.exists():
        with alias_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                target = normalise_place(row["official"])
                if target not in official:
                    raise ConfigError("state alias points outside the closed list")
                aliases[normalise_place(row["alias"])] = target
    return StateList(frozenset(official), aliases)


def load_pin_master(path: Path, states: StateList) -> PinMaster:
    """Load the India Post master (``pincode,office,district,state``).

    A malformed PIN or a state outside the closed list raises: a master we
    cannot trust must not silently validate values.
    """
    acc: dict[str, tuple[set[str], set[str], set[str]]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        needed = {"pincode", "office", "district", "state"}
        missing = needed - set(reader.fieldnames or ())
        if missing:
            raise ConfigError(f"PIN master missing columns: {sorted(missing)}")
        for n, row in enumerate(reader, start=2):
            pin = (row["pincode"] or "").strip()
            if not _PIN_RE.fullmatch(pin):
                raise ConfigError(f"PIN master line {n}: malformed PIN")
            state = states.canonical(row["state"] or "")
            if state is None:
                raise ConfigError(f"PIN master line {n}: state not in closed list")
            s, d, o = acc.setdefault(pin, (set(), set(), set()))
            s.add(state)
            d.add(normalise_place(row["district"] or ""))
            o.add(normalise_place(row["office"] or ""))
    return PinMaster(
        {
            pin: PinEntry(frozenset(s), frozenset(d - {""}), frozenset(o - {""}))
            for pin, (s, d, o) in acc.items()
        }
    )


@dataclass
class LazyLookups:
    """Loads the state list and repo PIN master on first use, then caches.

    ``master`` is None while the public master file has not been supplied.
    """

    directory: Path | None = None
    _states: StateList | None = field(default=None, init=False, repr=False)
    _master: PinMaster | None = field(default=None, init=False, repr=False)
    _master_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def states(self) -> StateList:
        if self._states is None:
            self._states = load_state_list(self.directory)
        return self._states

    @property
    def master(self) -> PinMaster | None:
        if not self._master_loaded:
            path = (self.directory or lookups_dir()) / PIN_MASTER_FILE
            self._master = load_pin_master(path, self.states) if path.exists() else None
            self._master_loaded = True
        return self._master


def _clean_pin(value: str) -> str | None:
    pin = strip_separators(value)
    return pin if _PIN_RE.fullmatch(pin) else None


def make_pin_exists_validator(lookups: LazyLookups) -> RuleValidator:
    def rule(value: str, ctx: ValidationContext) -> str | None:
        pin = _clean_pin(value)
        if pin is None:
            return "pin_format"
        master = lookups.master
        if master is None:
            return "lookup_unavailable"
        return None if master.get(pin) else "pin_not_found"

    return RuleValidator(PIN_VALIDATOR_NAME, rule)


def make_state_validator(lookups: LazyLookups) -> RuleValidator:
    def rule(value: str, ctx: ValidationContext) -> str | None:
        return None if lookups.states.canonical(value) else "state_not_in_closed_list"

    return RuleValidator(STATE_VALIDATOR_NAME, rule)


def make_pin_location_validator(
    lookups: LazyLookups,
    *,
    state_key: str | None,
    city_key: str | None,
) -> RuleValidator:
    """PIN must agree with the row's state and city (column keys from config).

    City agrees if it equals the master's district or a post-office name for
    the PIN. At least one of state/city must be present on the row, otherwise
    agreement cannot be shown and the check fails.
    """
    if state_key is None and city_key is None:
        raise ConfigError("pin_matches_location needs a state or city column key")

    def rule(value: str, ctx: ValidationContext) -> str | None:
        pin = _clean_pin(value)
        if pin is None:
            return "pin_format"
        master = lookups.master
        if master is None:
            return "lookup_unavailable"
        entry = master.get(pin)
        if entry is None:
            return "pin_not_found"
        state = _present(ctx, state_key)
        city = _present(ctx, city_key)
        if state is None and city is None:
            return "location_missing"
        if state is not None:
            canon = lookups.states.canonical(state)
            if canon is None:
                return "state_not_in_closed_list"
            if canon not in entry.states:
                return "pin_state_mismatch"
        if city is not None:
            c = normalise_place(city)
            if c not in entry.districts and c not in entry.offices:
                return "pin_city_mismatch"
        return None

    return RuleValidator(PIN_LOCATION_VALIDATOR_NAME, rule)


def _present(ctx: ValidationContext, key: str | None) -> str | None:
    if key is None:
        return None
    v = ctx.row_values.get(key)
    return v if v is not None and v.strip() else None


# Keyless validators, registered by name at import (see registry.py). The
# location validator needs column keys from the template, so the pipeline
# registers it with ``make_pin_location_validator`` at start-up.
DEFAULT_LOOKUPS = LazyLookups()
LOOKUP_VALIDATORS = (
    make_pin_exists_validator(DEFAULT_LOOKUPS),
    make_state_validator(DEFAULT_LOOKUPS),
)
