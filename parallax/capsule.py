from parallax.models import EvidenceAssessment


def evidence_value(item: dict, assessment: EvidenceAssessment) -> int:
    hypothesis_coverage = len(set(assessment.supports_hypotheses))
    return assessment.priority * 10 + hypothesis_coverage * 8


def build_capsule(
    available_evidence: list[dict],
    assessments: list[EvidenceAssessment],
    budget: int,
) -> tuple[list[dict], list[dict]]:
    assessment_map = {a.evidence_id: a for a in assessments}

    # Items Gemma didn't assess get a default low-priority stub
    _default = EvidenceAssessment(
        evidence_id="",
        priority=1,
        reason="Not assessed by model.",
        supports_hypotheses=[],
        consequence_if_lost="Unknown.",
    )
    n = len(available_evidence)

    # 0/1 knapsack: maximize total preservation value within byte budget
    # dp[i][w] = max value using first i items with budget w bytes
    # budget can be large so we work in 100-byte units to keep table small
    unit = 100
    W = budget // unit
    values = [evidence_value(e, assessment_map.get(e["id"], _default)) for e in available_evidence]
    weights = [max(1, e["size_bytes"] // unit) for e in available_evidence]

    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w_i = weights[i - 1]
        v_i = values[i - 1]
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= w_i and dp[i - 1][w - w_i] + v_i > dp[i][w]:
                dp[i][w] = dp[i - 1][w - w_i] + v_i

    # Backtrack to find selected items
    selected_ids = set()
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_ids.add(available_evidence[i - 1]["id"])
            w -= weights[i - 1]

    # Sort selected by priority descending for display
    selected = sorted(
        [e for e in available_evidence if e["id"] in selected_ids],
        key=lambda e: assessment_map.get(e["id"], _default).priority,
        reverse=True,
    )
    rejected = [e for e in available_evidence if e["id"] not in selected_ids]

    return selected, rejected


def hypotheses_covered(selected: list[dict], assessments: list[EvidenceAssessment]) -> set[str]:
    assessment_map = {a.evidence_id: a for a in assessments}
    covered = set()
    for item in selected:
        assessment = assessment_map.get(item["id"])
        if assessment:
            covered.update(assessment.supports_hypotheses)
    return covered
