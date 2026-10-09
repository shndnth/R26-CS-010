"""Does the calibration stage spend the privacy budget it claims?"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

RELATIVE_TOLERANCE = 1e-3


@dataclass(frozen=True, slots=True)
class QueryCost:
    query: str
    sensitivity: float
    recorded_value: float
    laplace_scale: float
    epsilon: float

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "sensitivity": self.sensitivity,
            "recorded_epsilon_used": self.recorded_value,
            "laplace_scale": round(self.laplace_scale, 6),
            "epsilon_spent": round(self.epsilon, 6),
        }


@dataclass
class CompositionResult:
    claimed_total: float
    field_meaning: str
    queries: list[QueryCost] = field(default_factory=list)

    @property
    def composed_total(self) -> float:
        return sum(q.epsilon for q in self.queries)

    @property
    def consistent(self) -> bool:
        return self.composed_total <= self.claimed_total * (1 + RELATIVE_TOLERANCE)

    @property
    def verdict(self) -> str:
        return "PASS" if self.consistent else "FAIL"

    @property
    def per_query_budget_for_claim(self) -> float:
        return self.claimed_total / len(self.queries) if self.queries else math.nan

    def interpretation(self) -> str:
        n = len(self.queries)
        if self.consistent:
            return (
                f"The {n} queries compose to epsilon {self.composed_total:.4f}, within the "
                f"claimed {self.claimed_total:.4f}."
            )
        return (
            f"The {n} queries compose to epsilon {self.composed_total:.4f} under sequential "
            f"composition, not the claimed {self.claimed_total:.4f} (read from the "
            f"{self.field_meaning}). To meet the claim, each query would need epsilon "
            f"{self.per_query_budget_for_claim:.4f} (the total divided by {n}), or the claimed "
            f"total must be restated as {self.composed_total:.4f}."
        )

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "claimed_total_epsilon": self.claimed_total,
            "composed_total_epsilon": round(self.composed_total, 6),
            "composition_rule": "sequential (basic): the queries read the same records",
            "epsilon_used_field_holds": self.field_meaning,
            "queries": [q.as_dict() for q in self.queries],
            "per_query_epsilon_to_meet_claim": round(self.per_query_budget_for_claim, 6),
            "interpretation": self.interpretation(),
        }


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-9)


def analyse(audit_log: dict, published: dict | None = None) -> CompositionResult:
    """Recover each query's spend from the audit log and compose them."""
    queries = audit_log.get("queries") or []
    if not queries:
        raise ValueError("the audit log records no queries")

    stated = float(audit_log.get("epsilon_per_query", audit_log["epsilon_calibration"]))
    claimed = float((published or {}).get(
        "calibration_epsilon_spent", audit_log.get("epsilon_total", stated)
    ))

    explicit = all("epsilon_spent" in q or "laplace_scale" in q for q in queries)
    holds_scale = not explicit and all(
        _close(float(q["epsilon_used"]), float(q["sensitivity"]) / stated) for q in queries
    )
    if explicit:
        meaning = "explicit epsilon_spent and laplace_scale fields"
    elif holds_scale:
        meaning = "Laplace scale (sensitivity / epsilon)"
    else:
        meaning = "per-query epsilon"

    result = CompositionResult(claimed_total=claimed, field_meaning=meaning)
    for q in queries:
        sensitivity = float(q["sensitivity"])
        recorded = float(q.get("epsilon_used", q.get("epsilon_spent", 0.0)))
        if "epsilon_spent" in q:
            epsilon = float(q["epsilon_spent"])
        elif "laplace_scale" in q:
            epsilon = sensitivity / float(q["laplace_scale"])
        elif holds_scale:
            epsilon = sensitivity / recorded
        else:
            epsilon = recorded
        if epsilon <= 0:
            raise ValueError(f"query {q.get('query')} records a non-positive budget")
        cost = QueryCost(str(q["query"]), sensitivity, recorded, sensitivity / epsilon, epsilon)
        result.queries.append(cost)
    return result
