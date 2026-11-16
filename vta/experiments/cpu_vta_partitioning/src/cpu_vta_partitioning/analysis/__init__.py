"""Boot-level paired inference; no frame-level pseudo-replication."""
import math
import statistics


def paired(a, b, margin=.05):
    """Positive log(b/a) means a has lower II. 95% paired t CI; 90% TOST CI."""
    from scipy.stats import t
    if set(a) != set(b) or len(a) < 2:
        raise ValueError("paired independent boot IDs required")
    if any(v <= 0 for v in [*a.values(), *b.values()]):
        raise ValueError("nonpositive II")
    values = [math.log(b[k] / a[k]) for k in sorted(a)]
    n, mean = len(values), statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    p = float(2 * t.sf(abs(mean / se), n - 1)) if se else (1. if mean == 0 else 0.)
    radius = float(t.ppf(.975, n - 1)) * se
    eq_radius = float(t.ppf(.95, n - 1)) * se
    def gain(x):
        return 1 - math.exp(-x)
    return {"n_boots": n, "gain": gain(mean), "gain_ci95": [gain(mean - radius), gain(mean + radius)],
            "p_two_sided": p, "equivalent_5pct": mean - eq_radius > -math.log(1 + margin)
                and mean + eq_radius < -math.log(1 - margin),
            "log_ratio_ci90": [mean - eq_radius, mean + eq_radius]}


def holm(p_values):
    result, previous = {}, 0.
    for i, (key, value) in enumerate(sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))):
        previous = max(previous, min(1., value * (len(p_values) - i)))
        result[key] = previous
    return result


def report(selection, measured, *, mechanism_supported=False, min_boots=6):
    """measured[candidate][boot] is a mean over valid rounds in that boot."""
    for group in selection.values():
        for key in group.values():
            if len(measured[key]) != min_boots:
                raise ValueError("confirmation requires the preregistered independent boot count")
    comparisons = {}
    for budget in ("1", "5"):
        for other in ("H", "G", "D"):
            ours, theirs = selection["O"][budget], selection[other][budget]
            comparisons[f"O_vs_{other}_budget{budget}"] = {**paired(measured[ours], measured[theirs]),
                                                            "same_configuration": ours == theirs}
    adjusted = holm({k: v["p_two_sided"] for k,v in comparisons.items()})
    for key, row in comparisons.items():
        row["p_holm"] = adjusted[key]
        row["passes_gain_gate"] = row["gain"] >= .05 and adjusted[key] < .05 and row["gain_ci95"][0] > 0
    secondary = {}
    for budget in ("1","5"):
        for with_host, without_host in (("H","C"),("O","A")):
            a,b = selection[with_host][budget],selection[without_host][budget]
            secondary[f"{with_host}_vs_{without_host}_budget{budget}"] = {
                **paired(measured[a],measured[b]),"same_configuration":a==b}
    secondary_adjusted = holm({k:v["p_two_sided"] for k,v in secondary.items()})
    for key,row in secondary.items():
        row["p_holm_secondary"] = secondary_adjusted[key]
        row["passes_gain_gate"] = row["gain"] >= .05 and secondary_adjusted[key] < .05 and row["gain_ci95"][0] > 0
    # H already accounts for host work: O vs H isolates the local CPU constraint.
    # Host-work selection effects are separate secondary ablations, not a
    # substitute for beating all three primary comparators in the SAME budget.
    budget_gates = {}
    for budget in ("1", "5"):
        gates = {name: comparisons[f"O_vs_{other}_budget{budget}"]["passes_gain_gate"]
                 for name, other in (("local_cpu_constraint", "H"), ("generic_capacity", "G"),
                                     ("core_isolation", "D"))}
        gates["all_primary"] = all(gates.values())
        gates["host_work_ablation"] = any(secondary[f"{a}_vs_{b}_budget{budget}"]["passes_gain_gate"]
                                          for a, b in (("H", "C"), ("O", "A")))
        budget_gates[budget] = gates
    eligible_budgets = [b for b, gates in budget_gates.items() if gates["all_primary"]]
    performance = bool(eligible_budgets)
    host_work = any(g["host_work_ablation"] for g in budget_gates.values())
    local_constraint = any(g["local_cpu_constraint"] for g in budget_gates.values())
    proceed = performance and mechanism_supported
    direction = ("continue_method_research" if proceed else
                 "performance_gain_but_mechanism_unconfirmed" if performance else
                 "effect_but_generic_or_isolation_explanation_unresolved" if host_work or local_constraint else
                 "insufficient_evidence_for_method_claim")
    return {"comparisons": comparisons, "host_work_ablation_secondary":secondary, "performance_gate": performance,
            "host_work_gate": host_work, "local_cpu_constraint_gate": local_constraint,
            "budget_gates": budget_gates, "eligible_budgets": eligible_budgets,
            "mechanism_supported": mechanism_supported,
            "continue_sensitivity": proceed, "direction": direction,
            "publication_claim": "These experiments do not establish conference acceptance or novelty."}
