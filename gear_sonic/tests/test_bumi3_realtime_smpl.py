import numpy as np
import pytest

from gear_sonic.utils.mujoco_sim.bumi3_realtime_smpl import (
    Bumi3RealtimeSmplSim2Sim,
    RealtimeSmplWindow,
    unpack_pose_message,
)
from gear_sonic.utils.mujoco_sim.bumi3_sim2sim import (
    Bumi3Contract,
    DEFAULT_BUMI3_SIM2SIM_CONFIG,
    ZeroPolicy,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


def _fields(count=10):
    quaternions = np.zeros((count, 4), dtype=np.float32)
    quaternions[:, 0] = 1.0
    return {
        "smpl_joints": np.arange(count * 24 * 3, dtype=np.float32).reshape(count, 24, 3),
        "body_quat_w": quaternions,
        "frame_index": np.arange(20, 20 + count, dtype=np.int64),
    }


def test_pico_packed_message_round_trip_and_last_ten_frames():
    fields = _fields(12)
    decoded = unpack_pose_message(pack_pose_message(fields, topic="pose"))
    window = RealtimeSmplWindow.from_fields(decoded)
    np.testing.assert_array_equal(window.local_joints, fields["smpl_joints"][-10:])
    np.testing.assert_array_equal(window.frame_index, np.arange(22, 32))
    np.testing.assert_allclose(window.root_quat_wxyz[:, 0], 1.0)


def test_pico_window_rejects_too_few_frames_and_bad_index():
    with pytest.raises(ValueError, match="num_frames_to_send 10"):
        RealtimeSmplWindow.from_fields(_fields(5))
    fields = _fields()
    fields["frame_index"][5] = fields["frame_index"][4]
    with pytest.raises(ValueError, match="strictly increasing"):
        RealtimeSmplWindow.from_fields(fields)


def test_live_window_builds_full_smpl_policy_observation():
    window = RealtimeSmplWindow.from_fields(_fields())

    class Receiver:
        def newest(self):
            return window

    contract = Bumi3Contract.from_yaml(DEFAULT_BUMI3_SIM2SIM_CONFIG)
    runner = Bumi3RealtimeSmplSim2Sim(
        contract,
        ZeroPolicy(contract, encoder="smpl"),
        Receiver(),
        window,
        start_paused=False,
    )
    observation = runner.build_observation()
    assert observation.shape == (1470,)
    assert np.isfinite(observation).all()
