"""CHIP wrist payload, represented as two rigidly attached point masses.

The stock URDF right_hand_palm_joint defines the mounting reference at
[.0415, -.003, 0] in right_wrist_yaw_link. Distances extend along local +X.
The 170 g rubber-hand inertial contribution is removed; visual/collision
geometry is unchanged. Hand inertial constants come from the stock mode-15 URDF.
"""
import numpy as np
import mujoco

PAYLOAD_MASSES = np.array([.237, .280])
PAYLOAD_POSITIONS = np.array([[.044, -.003, 0.], [.059, -.003, 0.]])
HAND_MASS = .170
HAND_COM = np.array([.0415, -.003, 0.]) + np.array([.05361310808, .00295905240, .00215413091])
HAND_INERTIA = np.array([
    [.00010099485234748, -.00003618590790516, -.00000074301518642],
    [-.00003618590790516, .00028135871571621, -.00000330189743286],
    [-.00000074301518642, -.00000330189743286, .00021894770413514],
])


def add_wrist_payload(spec):
    model = spec.compile()
    idx = model.body("right_wrist_yaw_link").id
    mass = model.body_mass[idx]
    com = model.body_ipos[idx].copy()
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[idx])
    rotation = rotation.reshape(3, 3)
    inertia = rotation @ np.diag(model.body_inertia[idx]) @ rotation.T
    total = mass - HAND_MASS + PAYLOAD_MASSES.sum()
    if mass <= HAND_MASS:
        raise ValueError("Expected stock wrist body with fused rubber-hand mass")
    combined_com = (mass * com - HAND_MASS * HAND_COM +
                    (PAYLOAD_MASSES[:, None] * PAYLOAD_POSITIONS).sum(0)) / total

    def parallel_axis(m, r):
        return m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

    inertia += parallel_axis(mass, com - combined_com)
    inertia -= HAND_INERTIA + parallel_axis(HAND_MASS, HAND_COM - combined_com)
    for m, pos in zip(PAYLOAD_MASSES, PAYLOAD_POSITIONS):
        inertia += parallel_axis(m, pos - combined_com)
    eigenvalues, axes = np.linalg.eigh(inertia)
    if eigenvalues.min() <= 0 or eigenvalues[-1] > eigenvalues[:2].sum() + 1e-12:
        raise ValueError("Invalid composite inertia after removing rubber hand")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, axes.flatten())
    body = spec.body("right_wrist_yaw_link")
    body.mass = total
    body.ipos = combined_com
    body.iquat = quat
    body.inertia = eigenvalues
    return spec
