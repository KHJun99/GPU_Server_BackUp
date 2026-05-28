"""Isaac Lab scene for the Phase 1 cube-lift task.

Builds a single-env ``InteractiveSceneCfg`` bundling a ground plane, a dome
light, a static table, the SO-ARM101 + PincOpen articulation, and the cube
rigid object. Phase 1 stays single-env (``num_envs=1``) on purpose — the
vectorized path moves in alongside the ManagerBasedRLEnv migration in Phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from khj_rl.envs.cube_lift.cfg import CubeLiftEnvCfg

if TYPE_CHECKING:  # avoid pulling carb at module import time
    from isaaclab.scene import InteractiveSceneCfg


# Table geometry — sits just below the spawned cube so cube.z = table_top_z + size/2.
_TABLE_SIZE_XY = (0.6, 0.6)
_TABLE_THICKNESS = 0.04


def cube_lift_scene_cfg(
    env_cfg: CubeLiftEnvCfg,
    include_viewer_camera: bool = False,
) -> "InteractiveSceneCfg":
    """Build a single-env ``InteractiveSceneCfg`` from a ``CubeLiftEnvCfg``.

    Must be called inside an AppLauncher-booted process; isaaclab imports are
    lazy so the path constants above remain importable without booting kit.

    ``include_viewer_camera`` adds a fixed third-person ``Camera`` sensor for
    the WebSocket viewer (``scripts/launch_ws_viewer.py``); leave it False for
    training/eval runs so we don't pay the render cost when nobody is looking.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    from khj_rl.envs.cube_lift.articulation import soarm101_pincopen_cfg

    robot_cfg = soarm101_pincopen_cfg(robot=env_cfg.robot)
    table_top_z = env_cfg.cube.table_top_z_m

    # Third-person viewer camera — declared in the class body below so
    # @configclass picks it up as a dataclass field. We can't add it via
    # post-hoc setattr because Isaac Lab's InteractiveScene iterates
    # ``dataclasses.fields(cfg)`` to discover assets/sensors, and a plain
    # class attribute set after @configclass wouldn't appear in that list.
    _viewer_cam_cfg = CameraCfg(
        prim_path="/World/ViewerCamera",
        update_period=0.05,  # 20 Hz, matches policy rate.
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=0.6,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.7, 0.5, 0.5),
            # quat (w,x,y,z) for eye=(0.7,0.5,0.5) looking at (0.2,0,0.05) in
            # Isaac Lab "world" convention (cam local: +X forward, +Y left,
            # +Z up). Derived from a forward = (target-eye)/|.| basis via
            # scipy Rotation.from_matrix(...).as_quat(); the earlier hand-
            # picked yaw/pitch quat had the camera staring at the sky and
            # the rendered frame came back ~95% dome light with the scene
            # in the bottom 16 rows only.
            rot=(-0.367, -0.258, -0.107, 0.887),
            convention="world",
        ),
    )

    @configclass
    class _CubeLiftSceneCfgBase(InteractiveSceneCfg):
        # World-level: ground + dome light live outside the per-env namespace.
        ground = AssetBaseCfg(
            prim_path="/World/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0)),
        )
        # Per-env: static table, the SO-ARM101 + PincOpen articulation, the cube.
        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(_TABLE_SIZE_XY[0], _TABLE_SIZE_XY[1], _TABLE_THICKNESS),
                # kinematic_enabled keeps the table static under the cube's contact.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.2, 0.0, table_top_z - 0.5 * _TABLE_THICKNESS),
            ),
        )
        robot = robot_cfg
        # Goal visualizer — a thin red disc sitting on the table at the
        # goal xy with radius = SuccessCfg.place_radius_xy. Visual only:
        # collision disabled and rigid_body disabled (AssetBaseCfg) so it
        # never participates in physics. Lives at z = table_top + 1 mm to
        # avoid z-fighting with the table top in the viewer / mp4.
        # Goal disc — first pass (opacity 0.6, height 2 mm, z+1 mm) was
        # effectively invisible under the dome light at intensity 600 in
        # the rendered video. Bump to fully opaque + 5 mm tall + 3 mm above
        # the table top so the marker reads cleanly in the third-person
        # camera. No physics change — still AssetBaseCfg with collisions
        # and rigid_props disabled.
        goal_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GoalMarker",
            spawn=sim_utils.CylinderCfg(
                radius=env_cfg.goal.radius_xy_m,
                height=0.005,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                    opacity=1.0,
                ),
                collision_props=None,
                rigid_props=None,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(
                    env_cfg.goal.pos_xyz_m[0],
                    env_cfg.goal.pos_xyz_m[1],
                    table_top_z + 0.003,
                ),
            ),
        )
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(env_cfg.cube.size_m,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=env_cfg.cube.mass_kg),
                # Tight contact offsets so the cube only feels finger
                # contact when they visually touch it. Defaults (~2 cm
                # contact aura, 0.4 cm rest separation) made the cube
                # vibrate while fingers were still visibly several
                # millimetres away — phantom contact at the edge of
                # the offset zone applied force without making mesh
                # contact, and the close motion just shoved the cube
                # without ever clamping.
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                # High-friction matte cube — default PhysX rigid body
                # material has static/dynamic friction ≈ 0.5 which lets
                # the cube squirt out of the PincOpen jaws on close.
                # 1.5/1.2 mimics a real matte rubber finish.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5,
                    dynamic_friction=1.2,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=tuple(env_cfg.cube.color_rgba[:3]),
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(
                    env_cfg.goal.pos_xyz_m[0],
                    env_cfg.goal.pos_xyz_m[1],
                    env_cfg.cube.spawn_z_m,
                ),
            ),
        )

    if include_viewer_camera:
        # Build a subclass adding the viewer_camera field, then let
        # @configclass re-process so the dataclass field list picks it up.
        @configclass
        class CubeLiftSceneCfg(_CubeLiftSceneCfgBase):
            viewer_camera: CameraCfg = _viewer_cam_cfg
    else:
        CubeLiftSceneCfg = _CubeLiftSceneCfgBase

    return CubeLiftSceneCfg(num_envs=1, env_spacing=2.5)
