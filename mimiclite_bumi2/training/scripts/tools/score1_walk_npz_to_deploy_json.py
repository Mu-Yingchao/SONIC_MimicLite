#!/usr/bin/env python3
"""Compatibility entry point for the generic Bumi NPZ deploy converter.

Use ``npz_to_deploy_json.py`` directly for new workflows. This filename is
kept so the command used for the first score1 walking conversion still works.
"""

from npz_to_deploy_json import main


if __name__ == "__main__":
    main()
