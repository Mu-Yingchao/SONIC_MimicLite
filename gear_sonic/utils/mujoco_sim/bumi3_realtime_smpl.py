# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive PICO SMPL windows and feed the BUMI3 1470-D sim2sim policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import time

import numpy as np
import zmq

from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    Bumi3SonicSim2Sim,
    Policy,
    make_static_reference_motion,
    quaternion_conjugate,
    quaternion_heading,
    quaternion_multiply,
)
from gear_sonic.utils.mujoco_sim.bumi3_smpl_reference import (
    SMPL_FUTURE_FRAMES,
    SmplReference,
    build_smpl_tokenizer,
)


HEADER_SIZE = 1280
_DTYPES = {
    "f32": np.dtype("<f4"),
    "f64": np.dtype("<f8"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "bool": np.dtype("?"),
}


def unpack_pose_message(message: bytes, *, topic: str = "pose") -> dict[str, np.ndarray]:
    """Decode the packed single-part message emitted by pico_manager_thread_server."""

    prefix = topic.encode("utf-8")
    if not message.startswith(prefix):
        raise ValueError(f"ZMQ message does not start with topic {topic!r}")
    header_start = len(prefix)
    payload_start = header_start + HEADER_SIZE
    if len(message) < payload_start:
        raise ValueError("ZMQ pose message is shorter than its fixed header")
    raw_header = message[header_start:payload_start].split(b"\0", 1)[0]
    header = json.loads(raw_header.decode("utf-8"))
    result: dict[str, np.ndarray] = {}
    offset = payload_start
    for field in header.get("fields", []):
        name = field["name"]
        dtype_name = field["dtype"]
        if dtype_name not in _DTYPES:
            raise ValueError(f"unsupported ZMQ field dtype: {dtype_name!r}")
        dtype = _DTYPES[dtype_name]
        shape = tuple(int(value) for value in field["shape"])
        count = int(np.prod(shape, dtype=np.int64))
        size = count * dtype.itemsize
        if offset + size > len(message):
            raise ValueError(f"truncated ZMQ field: {name}")
        result[name] = np.frombuffer(message, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        offset += size
    if offset != len(message):
        raise ValueError(f"ZMQ pose message has {len(message) - offset} trailing bytes")
    return result


@dataclass(frozen=True)
class RealtimeSmplWindow:
    """Ten chronological 50 Hz samples; index 0 is the delayed control reference."""

    local_joints: np.ndarray
    root_quat_wxyz: np.ndarray
    frame_index: np.ndarray
    received_monotonic: float

    @classmethod
    def from_fields(cls, fields: dict[str, np.ndarray]) -> "RealtimeSmplWindow":
        missing = {"smpl_joints", "body_quat_w", "frame_index"} - fields.keys()
        if missing:
            raise ValueError(f"PICO pose message missing fields: {sorted(missing)}")
        joints = np.asarray(fields["smpl_joints"], dtype=np.float32)
        quaternions = np.asarray(fields["body_quat_w"], dtype=np.float64)
        indices = np.asarray(fields["frame_index"], dtype=np.int64).reshape(-1)
        if joints.ndim != 3 or joints.shape[1:] != (24, 3):
            raise ValueError(f"smpl_joints must be [N,24,3], got {joints.shape}")
        if quaternions.shape != (joints.shape[0], 4):
            raise ValueError(f"body_quat_w must be [N,4], got {quaternions.shape}")
        if indices.shape != (joints.shape[0],):
            raise ValueError(f"frame_index must be [N], got {indices.shape}")
        if joints.shape[0] < SMPL_FUTURE_FRAMES:
            raise ValueError(
                f"PICO must send at least {SMPL_FUTURE_FRAMES} frames; got {joints.shape[0]}. "
                "Start pico_manager_thread_server.py with --num_frames_to_send 10."
            )
        joints = joints[-SMPL_FUTURE_FRAMES:].copy()
        quaternions = quaternions[-SMPL_FUTURE_FRAMES:].copy()
        indices = indices[-SMPL_FUTURE_FRAMES:].copy()
        if not np.isfinite(joints).all() or not np.isfinite(quaternions).all():
            raise ValueError("PICO SMPL window contains NaN/Inf")
        norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("PICO body_quat_w contains a zero quaternion")
        quaternions /= norms
        if np.any(np.diff(indices) <= 0):
            raise ValueError("PICO frame_index must be strictly increasing")
        return cls(joints, quaternions, indices, time.monotonic())

    def as_reference(self) -> SmplReference:
        return SmplReference(
            local_joints=self.local_joints,
            root_quat_wxyz=self.root_quat_wxyz,
            fps=50.0,
            name="pico_live",
        )


class ZmqSmplWindowReceiver:
    """CONFLATE subscriber retaining only the newest validated PICO pose window."""

    def __init__(self, endpoint: str, *, topic: str = "pose", max_age: float = 0.5):
        if max_age <= 0.0:
            raise ValueError("max_age must be positive")
        self.endpoint = endpoint
        self.topic = topic
        self.max_age = max_age
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self.socket.connect(endpoint)
        self.latest: RealtimeSmplWindow | None = None

    def receive(self, *, timeout: float) -> RealtimeSmplWindow:
        if timeout < 0.0:
            raise ValueError("timeout cannot be negative")
        if not self.socket.poll(round(timeout * 1000), zmq.POLLIN):
            raise TimeoutError(f"no PICO pose received from {self.endpoint} within {timeout:.2f}s")
        self.latest = RealtimeSmplWindow.from_fields(
            unpack_pose_message(self.socket.recv(), topic=self.topic)
        )
        return self.latest

    def newest(self) -> RealtimeSmplWindow:
        while self.socket.poll(0, zmq.POLLIN):
            self.latest = RealtimeSmplWindow.from_fields(
                unpack_pose_message(self.socket.recv(), topic=self.topic)
            )
        if self.latest is None:
            raise RuntimeError("PICO stream has not produced a valid window")
        age = time.monotonic() - self.latest.received_monotonic
        if age > self.max_age:
            raise RuntimeError(
                f"PICO stream stale for {age:.3f}s (limit {self.max_age:.3f}s); stopping control"
            )
        return self.latest

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class Bumi3RealtimeSmplSim2Sim(Bumi3SonicSim2Sim):
    """BUMI3 runner whose SMPL tokenizer comes from a live PICO window."""

    def __init__(
        self,
        contract: Bumi3Contract,
        policy: Policy,
        receiver: ZmqSmplWindowReceiver,
        initial_window: RealtimeSmplWindow,
        *,
        start_paused: bool = True,
        align_reference_heading: bool = True,
    ):
        static = make_static_reference_motion(contract, num_frames=64)
        # The parent runner validates that its placeholder Robot and SMPL
        # references have identical lengths.  Live observations below never
        # consume this placeholder; repeat the calibration sample solely to
        # satisfy reset/diagnostic invariants without inventing robot motion.
        placeholder_smpl = SmplReference(
            local_joints=np.repeat(initial_window.local_joints[:1], static.num_frames, axis=0),
            root_quat_wxyz=np.repeat(
                initial_window.root_quat_wxyz[:1], static.num_frames, axis=0
            ),
            fps=50.0,
            name="pico_live",
        )
        static = replace(
            static,
            smpl_reference=placeholder_smpl,
            has_robot_reference=False,
            name="pico_live",
        )
        self.receiver = receiver
        self.frozen_window = initial_window
        super().__init__(
            contract,
            static,
            policy,
            loop_motion=True,
            start_paused=start_paused,
            align_reference_heading=False,
            encoder="smpl",
        )
        if align_reference_heading:
            robot_heading = quaternion_heading(self.data.xquat[self.anchor_body_id])
            reference_heading = quaternion_heading(initial_window.root_quat_wxyz[0])
            self.stream_heading_delta_wxyz = quaternion_multiply(
                robot_heading, quaternion_conjugate(reference_heading)
            )
        else:
            self.stream_heading_delta_wxyz = np.asarray([1.0, 0.0, 0.0, 0.0])

    def build_observation(self) -> np.ndarray:
        self._append_current_state()
        window = self.receiver.newest() if self.playing else self.frozen_window
        tokenizer = build_smpl_tokenizer(
            window.as_reference(),
            np.arange(SMPL_FUTURE_FRAMES, dtype=np.int64),
            self.data.xquat[self.anchor_body_id],
            self.stream_heading_delta_wxyz,
        )
        value = np.concatenate((tokenizer, self._build_proprioception())).astype(np.float32)
        if value.size != self.policy_input_dim or not np.isfinite(value).all():
            raise ValueError(f"live SMPL policy observation is invalid: {value.shape}")
        self.last_observation = value
        return value
