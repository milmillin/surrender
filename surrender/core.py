import numpy as np
import numpy.typing as npt
import pyoctree.pyoctree as ot

from typing import Optional, Any


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def _generate_rays(
    eye: npt.ArrayLike,
    width: int,
    height: int,
    target: npt.ArrayLike = np.zeros((3,)),
    up: npt.ArrayLike = np.array([0, 1, 0]),
    fov: float = 45.0,
    ray_length: float = 1000,
) -> np.ndarray:
    """
    Constructs a viewing matrix where a camera is located at the eye position and looking at
    (or rotating to) the target point. The eye position and target are defined in world space.

    Args:
        eye: (3,)
        width: int
        height: int
        target: (3,)
        up: (3,)
        fov: float (degrees)
    """
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    assert eye.shape == (3,), f"Invalid shape for eye. Got {eye.shape}"
    assert target.shape == (3,), f"Invalid shape for target. Got {target.shape}"
    assert up.shape == (3,), f"Invalid shape for up. Got {up.shape}"

    up = up / np.linalg.norm(up)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    half_fov_rad: float = np.radians(fov) / 2
    ox: float = -np.tan(half_fov_rad)
    oy = ox / width * height
    pixel_size = -ox * 2 / width

    xs = [ox + pixel_size * (i + 0.5) for i in range(width)]
    ys = reversed([oy + pixel_size * (j + 0.5) for j in range(height)])

    rays = []
    for y in ys:
        for x in xs:
            rays.append([eye, eye + (forward + x * right + y * up) * ray_length])
    return np.array(rays, dtype=np.float32).reshape(height, width, 2, 3)


def _look_at(
    eye: npt.ArrayLike, target: npt.ArrayLike, up: npt.ArrayLike = np.array([0, -1, 0])
) -> np.ndarray:
    """
    Constructs a viewing matrix where a camera is located at the eye position and looking at
    (or rotating to) the target point. The eye position and target are defined in world space.

    Args:
        eye: (3,)
        target: (3,)
        up: (3,)
    """
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    assert eye.shape == (3,), f"Invalid shape for eye. Got {eye.shape}"
    assert target.shape == (3,), f"Invalid shape for target. Got {target.shape}"
    assert up.shape == (3,), f"Invalid shape for up. Got {up.shape}"

    up = up / np.linalg.norm(up)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, forward)

    # Create the view matrix
    view_matrix = np.array(
        [
            [right[0], new_up[0], -forward[0], 0.0],
            [right[1], new_up[1], -forward[1], 0.0],
            [right[2], new_up[2], -forward[2], 0.0],
            [-np.dot(right, eye), -np.dot(new_up, eye), np.dot(forward, eye), 1.0],
        ]  # type: ignore
    )
    return view_matrix


def _perspective(
    fov_rad: float = np.radians(45.0),
    ar: float = 1.0,
    near: float = 1.0,
    far: float = 50.0,
):
    """
    Builds a perspective projection matrix.

    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Args:
        fov_x : float
            Horizontal field of view (in radians).
        ar : float
            Aspect ratio (w/h).
        near : float
            Depth of the near plane relative to the camera.
        far : float
            Depth of the far plane relative to the camera.
    """
    tanhalffov = np.tan((fov_rad / 2))
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)

    return proj_mat


def _translate(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]).astype(
        np.float32
    )


def _scale(x: float, y: float, z: float) -> np.ndarray:
    return np.array([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]]).astype(
        np.float32
    )


def _transform_V(V: np.ndarray, mtx: np.ndarray) -> np.ndarray:
    """
    V: (n_V, 3)
    mtx: (4, 4)
    """
    assert V.shape[-1] == 3
    assert mtx.shape == (4, 4)
    return (mtx @ np.concatenate((V, np.ones((len(V), 1))), axis=-1))[:, :3]  # (n_V, 3)


def render(
    V: np.ndarray,
    F: np.ndarray,
    height: int,
    width: int,
    N: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
    cam_pos: npt.ArrayLike = [3, 0, 0],
    fov: float = 45.0,
    spp: int = 1,
) -> np.ndarray:
    """
    Renders a mesh.

    V: (n_V, 3): vertices
    F: (n_F, 3): faces
    height, width: output image dimension
    N: (n_F, 3): face normals
    C: (n_F, 3): [0, 1): face colors
    cam_pos: camera position
    fov: field of view in radians
    spp: sqrt(samples per pixel)
    """

    V = V.astype(np.float32)
    F = F.astype(np.int32)

    width *= spp
    height *= spp

    cam_pos = np.array(cam_pos)
    tree = ot.PyOctree(V, F)
    rays = _generate_rays(cam_pos, width, height, fov=fov)

    if N is None:
        V0 = V[F[:, 0]]
        N = _normalize(np.cross(V[F[:, 1]] - V0, V[F[:, 2]] - V0))
    C = C or np.ones((len(F), 3))

    canvas = np.ones((height, width, 3), dtype=np.float32)

    for iy in range(height):
        for ix in range(width):
            intersections = tree.rayIntersection(rays[iy, ix])
            if len(intersections) == 0:
                continue
            intersections = [(i.triLabel, i.s, i.p) for i in intersections]
            intersections.sort(key=lambda x: x[1])
            fid, dist, p = intersections[0]
            v = _normalize(cam_pos - p)
            l = v  # TODO

            canvas[iy, ix] = C[fid] * (np.dot(l, N[fid])) ** 0.45

    if spp != 1:
        canvas = (
            canvas.reshape(height // spp, spp, width // spp, spp, 3)
            .mean(axis=1)
            .mean(axis=2)
        )

    return canvas
