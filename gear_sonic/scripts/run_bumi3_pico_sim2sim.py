# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run BUMI3 MuJoCo sim2sim from the live PICO SMPL ZMQ stream."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import tyro

from gear_sonic.utils.mujoco_sim.bumi3_realtime_smpl import (
    Bumi3RealtimeSmplSim2Sim,
    ZmqSmplWindowReceiver,
)
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    OnnxRobotPolicy,
)


@dataclass
class Args:
    policy: Path
    """BUMI3 SMPL joint policy, with input [1,1470] and output [1,21]."""

    zmq_url: str = "tcp://127.0.0.1:5556"
    """PICO pose publisher endpoint."""

    topic: str = "pose"
    """Packed-message topic emitted by pico_manager_thread_server.py."""

    startup_timeout: float = 60.0
    """Seconds to wait for the first complete ten-frame window."""

    stream_timeout: float = 0.5
    """Stop control when no fresh valid pose window arrives for this many seconds."""

    config: Path = DEFAULT_BUMI3_SIM2SIM_CONFIG
    provider: str = "cpu"
    duration: float | None = None
    headless: bool = False
    real_time: bool = True
    autoplay: bool = False
    align_reference_heading: bool = True


def main(args: Args) -> None:
    contract = Bumi3Contract.from_yaml(args.config)
    policy = OnnxRobotPolicy(args.policy, contract, provider=args.provider, encoder="smpl")
    receiver = ZmqSmplWindowReceiver(
        args.zmq_url, topic=args.topic, max_age=args.stream_timeout
    )
    try:
        print(
            f"BUMI3_PICO_WAITING endpoint={args.zmq_url} topic={args.topic!r} "
            "required_frames=10",
            flush=True,
        )
        initial = receiver.receive(timeout=args.startup_timeout)
        print(
            "BUMI3_PICO_CONNECTED="
            + json.dumps(
                {
                    "first_frame": int(initial.frame_index[0]),
                    "last_frame": int(initial.frame_index[-1]),
                    "buffer_latency_ms": 180,
                    "policy_input_dim": policy.input_dim,
                    "onnx_intra_op_threads": policy.intra_op_num_threads,
                    "onnx_inter_op_threads": policy.inter_op_num_threads,
                }
            ),
            flush=True,
        )
        runner = Bumi3RealtimeSmplSim2Sim(
            contract,
            policy,
            receiver,
            initial,
            start_paused=not (args.headless or args.autoplay),
            align_reference_heading=args.align_reference_heading,
        )
        if args.duration is None:
            control_steps = None if not args.headless else round(10.0 / contract.control_dt)
        else:
            if args.duration <= 0.0:
                raise ValueError("duration must be positive")
            control_steps = max(1, round(args.duration / contract.control_dt))
        if not args.headless and not args.autoplay:
            print("PICO reference is frozen. Focus the MuJoCo window and press T to arm live control.", flush=True)
        stats = runner.run(
            control_steps,
            headless=args.headless,
            real_time=args.real_time,
            show_reference=False,
        )
        print("BUMI3_PICO_SIM2SIM_STATS=" + json.dumps(stats, indent=2), flush=True)
    finally:
        receiver.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
