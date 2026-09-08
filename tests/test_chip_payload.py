import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

spec = importlib.util.spec_from_file_location(
    "chip_payload", Path(__file__).parents[1] / "mimic_lite/assets/chip_payload.py")
payload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payload)


class PayloadTests(unittest.TestCase):
    def test_parallel_axis_and_original_preserved(self):
        source = mujoco.MjSpec.from_string('''<mujoco><worldbody>
        <body name="right_wrist_yaw_link"><freejoint/>
        <inertial mass=".25" pos=".07 0 .002" diaginertia=".001 .002 .0025"/>
        </body></worldbody></mujoco>''')
        original = source.compile()
        modified = payload.add_wrist_payload(source.copy()).compile()
        i = 1
        self.assertAlmostEqual(modified.body_mass[i] - original.body_mass[i], .347)
        expected_com = (.25 * original.body_ipos[i] - payload.HAND_MASS * payload.HAND_COM +
                        (payload.PAYLOAD_MASSES[:, None] * payload.PAYLOAD_POSITIONS).sum(0)) / .597
        np.testing.assert_allclose(modified.body_ipos[i], expected_com)

        def about_origin(model):
            r = model.body_ipos[i]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, model.body_iquat[i])
            rotation = rotation.reshape(3, 3)
            return rotation @ np.diag(model.body_inertia[i]) @ rotation.T + model.body_mass[i] * (
                np.dot(r, r) * np.eye(3) - np.outer(r, r))

        expected_delta = sum(m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
                             for m, r in zip(payload.PAYLOAD_MASSES, payload.PAYLOAD_POSITIONS))
        r = payload.HAND_COM
        expected_delta -= payload.HAND_INERTIA + payload.HAND_MASS * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        np.testing.assert_allclose(about_origin(modified) - about_origin(original), expected_delta, atol=1e-12)
        self.assertAlmostEqual(source.compile().body_mass[i], .25)
        self.assertEqual(original.nq, modified.nq)


if __name__ == "__main__":
    unittest.main()
