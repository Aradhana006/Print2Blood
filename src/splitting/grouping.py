from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd


def _stable_group_id(members: list[str]) -> str:
    payload = "|".join(sorted(members)).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()[:10]
    return f"PROXY_{digest}"


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def build_proxy_groups(
    index_df: pd.DataFrame,
    similarity_pairs: pd.DataFrame,
    *,
    phash_max_hamming: int,
    confirm_ssim_min: float,
    confirm_with_ssim: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Build leakage-control proxy groups.

    Priority:
    1. Exact MD5 duplicates always share a group.
    2. Same-label highly similar pairs may merge groups if they pass thresholds.
    3. Cross-label pairs never merge.
    """
    uf = UnionFind(index_df["image_path"].tolist())

    exact_groups = []
    for md5, group in index_df.groupby("md5"):
        paths = group["image_path"].tolist()
        if len(paths) < 2:
            continue
        root = paths[0]
        for path in paths[1:]:
            uf.union(root, path)
        for path in paths:
            exact_groups.append(
                {
                    "group_id": f"EXACT_{md5[:8]}",
                    "image_path": path,
                    "hash": md5,
                    "label": group.loc[group["image_path"] == path, "label"].iloc[0],
                }
            )

    merge_count = 0
    if similarity_pairs is not None and len(similarity_pairs):
        candidates = similarity_pairs.copy()
        if "same_label" in candidates.columns:
            candidates = candidates[candidates["same_label"]]
        if "method" in candidates.columns:
            # Prefer confirmed phash+ssim edges if present.
            if "phash_ssim" in set(candidates["method"]):
                candidates = candidates[candidates["method"] == "phash_ssim"]
            elif "phash" in set(candidates["method"]):
                candidates = candidates[candidates["method"] == "phash"]

        for row in candidates.itertuples(index=False):
            score = getattr(row, "similarity_score", None)
            method = getattr(row, "method", "")
            if method == "phash":
                # similarity_score for hashes is stored as 1 - hamming/max_bits later;
                # also accept explicit hamming if present.
                hamming = getattr(row, "hamming", None)
                if hamming is not None and hamming > phash_max_hamming:
                    continue
            if confirm_with_ssim and method == "phash_ssim":
                if score is None or score < confirm_ssim_min:
                    continue
            if uf.union(row.image_a, row.image_b):
                merge_count += 1

    roots = {path: uf.find(path) for path in index_df["image_path"]}
    root_to_members = defaultdict(list)
    for path, root in roots.items():
        root_to_members[root].append(path)

    # Deterministic group IDs: EXACT_ for pure exact-duplicate groups when possible,
    # PROXY_ otherwise.
    md5_map = dict(zip(index_df["image_path"], index_df["md5"]))
    label_map = dict(zip(index_df["image_path"], index_df["label"]))

    group_rows = []
    group_id_by_root = {}
    exact_only = 0
    proxy_merged = 0
    for root, members in sorted(root_to_members.items(), key=lambda kv: min(kv[1])):
        members = sorted(members)
        md5s = {md5_map[m] for m in members}
        if len(members) > 1 and len(md5s) == 1:
            gid = f"EXACT_{next(iter(md5s))[:8]}"
            exact_only += 1
        elif len(members) > 1:
            gid = _stable_group_id(members)
            proxy_merged += 1
        else:
            gid = f"SINGLE_{md5_map[members[0]][:10]}"
        group_id_by_root[root] = gid
        for member in members:
            group_rows.append(
                {
                    "image_path": member,
                    "label": label_map[member],
                    "group_id": gid,
                    "group_size": len(members),
                    "md5": md5_map[member],
                }
            )

    groups_df = pd.DataFrame(group_rows).sort_values(["group_id", "image_path"]).reset_index(drop=True)
    meta = {
        "total_images": int(len(index_df)),
        "total_groups": int(groups_df["group_id"].nunique()),
        "exact_duplicate_groups": int(exact_only),
        "similarity_merged_groups": int(proxy_merged),
        "singleton_groups": int((groups_df.drop_duplicates("group_id")["group_size"] == 1).sum()),
        "similarity_merges_applied": int(merge_count),
    }
    exact_df = pd.DataFrame(exact_groups)
    return groups_df, exact_df, meta
