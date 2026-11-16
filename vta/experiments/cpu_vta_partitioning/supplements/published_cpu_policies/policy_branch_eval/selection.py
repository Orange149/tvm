"""Selection functions receive configurations/scores only, never FPS."""
from cpu_policies.open_resources.domain import compatible_groups


def fixed_family(cfg):
    masks = [tuple(s['mask']) for s in cfg['stages'] if s['device'] == 'cpu']
    return (cfg['execution_mode'] == 'resource_group' and len(set(masks)) >= 2
            and compatible_groups(masks))


def mixed(original, fixed, k=20):
    """Alternate branches, deduplicate and retain provenance for each actual pick."""
    if k < 1 or k % 2:
        raise ValueError('mixed list requires a positive even budget')
    branches = [('original', original), ('fixed', fixed)]
    cursors = {n: 0 for n, _ in branches}
    counts = {n: 0 for n, _ in branches}
    result, seen, reasons = [], set(), []

    def take(name, rows):
        while cursors[name] < len(rows):
            index = cursors[name]; cursors[name] += 1
            row = rows[index]
            if row['id'] in seen:
                continue
            seen.add(row['id']); result.append(row); counts[name] += 1
            reasons.append(dict(id=row['id'], branch=name, branch_rank=index+1))
            return True
        return False

    while len(result) < k:
        added = False
        for name, rows in branches:
            if counts[name] < k//2 and len(result) < k:
                added |= take(name, rows)
        if not added:
            break
    while len(result) < k:
        added = False
        for name, rows in branches:
            if len(result) < k:
                added |= take(name, rows)
        if not added:
            break
    return result, reasons
