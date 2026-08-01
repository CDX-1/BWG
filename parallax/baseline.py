def naive_capsule(items: list[dict], budget: int) -> tuple[list[dict], list[dict]]:
    """Select smallest files first."""
    ranked = sorted(items, key=lambda item: item["size_bytes"])

    selected = []
    rejected = []
    used = 0

    for item in ranked:
        if used + item["size_bytes"] <= budget:
            selected.append(item)
            used += item["size_bytes"]
        else:
            rejected.append(item)

    return selected, rejected
