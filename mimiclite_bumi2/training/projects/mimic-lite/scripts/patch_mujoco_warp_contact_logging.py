#!/usr/bin/env python3
"""Patch MuJoCo Warp contact-overflow logs with world and sensor IDs.

Run this with the Python interpreter from the mjlab environment after installing
or updating its dependencies. The operation is idempotent and fails rather than
changing an unrecognized MuJoCo Warp source version.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ORIGINAL = '''      wp.printf("contact match overflow: please increase Option.contact_sensor_maxmatch to %u\\n", contactmatchid)'''
PATCHED = '''      wp.printf(
        "contact match overflow: worldid=%u contactsensorid=%u "
        "contactmatchid=%u maxmatch=%u\\n",
        worldid,
        contactsensorid,
        contactmatchid,
        opt_contact_sensor_maxmatch,
      )'''


def sensor_source() -> Path:
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco_warp is not installed in this Python environment")
    return Path(spec.origin).resolve().parent / "_src" / "sensor.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check whether the enhanced log is present without changing files",
    )
    args = parser.parse_args()

    path = sensor_source()
    source = path.read_text(encoding="utf-8")

    if PATCHED in source:
        print(f"already patched: {path}")
        return 0
    if ORIGINAL not in source:
        raise RuntimeError(
            f"unsupported mujoco_warp sensor source: expected overflow log not found in {path}"
        )
    if args.check:
        print(f"patch required: {path}")
        return 1

    path.write_text(source.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    print(f"patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
