#!/usr/bin/env python3
"""Quarantine non-flat-ground and interaction-dependent BUMI motions.

The classifier only uses the descriptive motion name before ``__A...``.
By default it is a dry run.  ``--apply`` moves candidates from score1 to
score3 with ``os.replace`` while preserving their relative paths.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ROOT = Path("/data2/zcx/datasets/any4hdmi-bumi-v2/motions")

TERRAIN_TOKENS = {
    "stair", "stairs", "staircase", "ladder", "obstacle", "obstacles",
    "hurdle", "hurdles", "ramp", "slope", "incline", "decline",
    "terrain", "uneven", "curb", "platform", "stepdown", "jumpdown",
    "edge", "crossroad",
}

# Physical props, furniture, tools, vehicles, wearables, instruments, food,
# and fixed interfaces.  Ambiguous dance/exercise terms are handled before
# this set is consulted.
OBJECT_TOKENS = {
    "apple", "axe", "ball", "bar", "beer", "bin", "binoculars", "book",
    "bottle", "brick", "broom", "burger", "bus", "button", "cane", "car",
    "carcrash", "cards", "cart", "cellphone", "cellpone",
    "chainsaw", "chair", "cigarette", "coin", "crate", "crank", "cup",
    "cupboard", "desk", "dog", "door", "drawer", "fish", "flute",
    "flyers", "fridge", "glass", "guitar", "hammer", "handbag", "handle",
    "hat", "horse", "icecream", "item", "lasso", "lever", "lighter",
    "luggage", "meat", "mic", "microwave", "mop", "mug", "newspaper",
    "object", "orange", "paper", "pepper", "percussion", "phone",
    "pickaxe", "picklock", "pills", "plants", "purse", "recorder", "rifle",
    "rope", "shelf", "shoes", "shovel", "sledgehammering", "sofa", "stick",
    "stool", "suitcase", "switch", "table", "taxi", "tennis", "ticket",
    "tool", "trash", "tree", "trolley", "umbrella", "valve", "vegetables",
    "vehicle", "violin", "wall", "wine", "wood", "zippo",
    "crutch", "crutches", "bench", "bed",
}

# Verbs whose captured motion assumes a manipulated object or fixed fixture.
OBJECT_ACTION_TOKENS = {
    "catch", "clean", "cutting", "digging", "drive", "drinking",
    "drop", "eat", "eating", "fanning", "feeding", "fishing", "fishin",
    "fixing", "grab", "grating", "grinding", "hold", "lift", "lock",
    "looting", "nailing", "open", "operating", "overturning", "painting",
    "pass", "passing", "peeling", "pick", "picking", "place", "play",
    "playing", "pounding", "press", "pull", "pulled", "pulling", "put",
    "read", "shut", "slam", "smoke", "smoking", "take",
    "throw", "typing", "unlock", "wash", "watering",
}

# Motions that require another person or animal to determine/support the pose.
INTERACTION_TOKENS = {
    "adult", "child", "crowd", "handshake", "hifive", "highfive", "hug",
    "hugging", "partner", "people", "person", "petting", "slap", "someone",
    "stroke", "talking",
}

OBJECT_PREFIXES = {"small", "medium", "big", "outside", "inside", "item"}

# Object-like words that still mean a real prop/person when they occur in a
# dance name.  Other such words (for example orange_justice, handbag_walk,
# horse_step, body_percussion, fishin and lock_step) are dance vocabulary.
DANCE_EXPLICIT_CONTEXT_TOKENS = {
    "adult", "baby", "bottle", "chair", "child", "dog", "guitar", "lasso",
    "mic", "object", "partner", "person", "rope", "table", "violin",
}


def normalized_name(path: Path) -> str:
    name = path.stem.split("__", 1)[0].lower().replace("-", "_")
    return re.sub(r"_+", "_", name).strip("_")


def classify(name: str) -> tuple[str, str] | None:
    tokens = name.split("_")
    token_set = set(tokens)
    first = tokens[0] if tokens else ""

    # Known false friends: these are names of flat-ground dance/exercise moves.
    ambiguous_keep = set()
    if "ball_change" in name:
        ambiguous_keep.add("ball")
    if "dance_vouge_cat_walk" in name:
        ambiguous_keep.add("cat")
    if name.startswith("ab_bicycle"):
        ambiguous_keep.add("bicycle")
    if "dance_raise_the_roof" in name:
        ambiguous_keep.add("roof")
    if "box_step" in name:
        ambiguous_keep.add("box")
    is_dance = bool(token_set & {"dance", "dancing", "dancecard", "dancecards1"})
    if is_dance:
        ambiguous_keep.update(
            (OBJECT_TOKENS | OBJECT_ACTION_TOKENS | INTERACTION_TOKENS)
            - DANCE_EXPLICIT_CONTEXT_TOKENS
        )
    if name.startswith("self_hifive"):
        ambiguous_keep.add("hifive")
    effective_tokens = token_set - ambiguous_keep

    terrain = sorted(effective_tokens & TERRAIN_TOKENS)
    if terrain:
        return "terrain_or_support", ",".join(terrain)
    if (
        "50cm_box" in name
        or "jump_on_box" in name
        or name.startswith(("box_jump", "box_dips"))
    ):
        return "terrain_or_support", "elevated_box"
    if "bump_into" in name or "avoid_bump" in name or "avoid_obstacle" in name:
        return "terrain_or_support", "obstacle_context"
    if "step_in_shit" in name:
        return "terrain_or_support", "ground_obstacle"

    # These prefixes are the corpus' explicit size/weight object-manipulation
    # families, not descriptions of body size or movement amplitude.
    if first in OBJECT_PREFIXES:
        return "object_or_fixture", f"object_family:{first}"

    objects = sorted(effective_tokens & OBJECT_TOKENS)
    if objects:
        return "object_or_fixture", ",".join(objects)

    # "push_up" is a floor exercise and deliberately retained.
    object_actions = effective_tokens & OBJECT_ACTION_TOKENS
    if name.startswith("push_up"):
        object_actions.discard("push")
    if "_in_place_" in f"_{name}_":
        object_actions.discard("place")
    if "grab_head" in name:
        object_actions.discard("grab")
    if object_actions:
        return "object_or_fixture", ",".join(sorted(object_actions))
    if "checking_time" in name or "pocket_search" in name or "empty_pockets" in name:
        return "object_or_fixture", "wearable_or_pocket"
    if first in {"grab", "relax"} and "grab" in token_set:
        return "object_or_fixture", "carried_object"

    interactions = sorted(effective_tokens & INTERACTION_TOKENS)
    if interactions:
        return "person_or_animal", ",".join(interactions)
    if "dandle_baby" in name or "toss_baby" in name:
        return "person_or_animal", "baby"
    if "get_pulled" in name or "pull_shoudler" in name:
        return "person_or_animal", "two_person_pull"
    if "fanning_someone" in name:
        return "person_or_animal", "someone"
    if "dont_touch_reaction" in name:
        return "person_or_animal", "interaction_reaction"

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", default="score1")
    parser.add_argument("--destination", default="score3")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.root / args.source
    dst = args.root / args.destination
    paths = sorted(src.rglob("*.npz"))
    candidates: list[tuple[Path, str, str, str]] = []
    for path in paths:
        name = normalized_name(path)
        result = classify(name)
        if result is not None:
            category, reason = result
            candidates.append((path.relative_to(src), category, reason, name))

    category_counts = Counter(row[1] for row in candidates)
    reason_counts = Counter((row[1], row[2]) for row in candidates)
    print(f"source={src}")
    print(f"destination={dst}")
    print(f"source_files={len(paths)}")
    print(f"candidates={len(candidates)}")
    print(f"remaining={len(paths) - len(candidates)}")
    print("category_counts=" + repr(dict(sorted(category_counts.items()))))
    print("top_reasons:")
    for (category, reason), count in reason_counts.most_common(40):
        print(f"  {count:6d}  {category:20s}  {reason}")

    rng = random.Random(args.seed)
    by_category: dict[str, list[tuple[Path, str, str, str]]] = defaultdict(list)
    for row in candidates:
        by_category[row[1]].append(row)
    for category in sorted(by_category):
        sample = rng.sample(by_category[category], min(args.examples, len(by_category[category])))
        print(f"examples[{category}]:")
        for rel, _, reason, _ in sample:
            print(f"  {reason:28s} {rel}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.report.with_name(args.report.name + f".tmp.{os.getpid()}")
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["relative_path", "category", "reason", "normalized_name"])
            writer.writerows(candidates)
        os.replace(tmp, args.report)
        print(f"report={args.report}")

    if not args.apply:
        print("mode=dry-run")
        return

    missing_sources = [rel for rel, _, _, _ in candidates if not (src / rel).is_file()]
    collisions = [rel for rel, _, _, _ in candidates if (dst / rel).exists()]
    if missing_sources:
        raise FileNotFoundError(f"Missing source files before move: {missing_sources[:10]}")
    if collisions:
        raise FileExistsError(f"Destination collisions before move: {collisions[:10]}")

    moved = 0
    for rel, _, _, _ in candidates:
        source_path = src / rel
        destination_path = dst / rel
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_path, destination_path)
        moved += 1
    print(f"mode=apply moved={moved}")


if __name__ == "__main__":
    main()
