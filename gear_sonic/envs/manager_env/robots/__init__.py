# Re-export robot configs for backward compatibility.
# Import from gear_sonic.envs.manager_env.robots.g1, .h2 or .bumi2 directly for new code.
# BUMI3 已于 2026-09-24 移除：本项目目标机型是 BUMI2，BUMI3 只在早期阶段用过。
from gear_sonic.envs.manager_env.robots.bumi2 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.g1 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.h2 import *  # noqa: F401,F403
