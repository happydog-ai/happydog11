#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import heapq
import math
import os
import zlib
import struct
import binascii
import numpy as np
from datetime import datetime
def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径
    GLOBAL_REDUCED_SIZE = 16

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.last_battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)
        self.last_pos = (0, 0)               # 上一步位置
        self.no_move_count = 0               # 连续未移动步数

        self.dirt_delta = 0.0
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        self.clean_streak = 0                # 连续清扫步数
        self.cleaned_region_walk_streak = 0   # 连续走在已清扫区域的步数
        self.dirt_prefer_weight = 0.35
        self.npc_avoid_radius = 2.5
        self.goal_center_tolerance = 1.0

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        # Charger info
        self.charger_positions = []
        self.charger_cells = set()
        self.nearest_charger_dist = 999.0
        self.last_nearest_charger_dist = 999.0

        # Obstacle distance
        self.nearest_obstacle_dist = 999.0
        self.last_nearest_obstacle_dist = 999.0

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # NPC info
        self.npc_positions = []
        self.last_npc_positions = []
        self.nearest_npc_dist = 999.0
        self.last_nearest_npc_dist = 999.0

        # Trajectory / exploration / dirt memory
        self.visited_counts = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.explored_mask = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.global_dirt_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.unique_visited = 0

        self.collision_type = "none"   # none / npc / charger / obstacle
        self.terminated = False
        self.reward_info = {}

        # Low battery guide
        self.low_battery_threshold = 0.30
        self.very_low_battery_threshold = 0.15

        self.rule_action = -1          # 当前规则动作，-1 表示无
        self.use_rule_guide = 0.0      # 当前状态是否建议使用规则引导
        self.on_charger = 0.0          # 当前是否在充电桩区域

        # Optional heading support for map stitching
        self.heading = 0.0
        self.has_heading = False

        # A* debug save
        self.astar_save_count = 0
        self.debug_save_dir = "/data/projects/robot_vacuum/images"

    def _extract_heading_from_hero(self, hero):
        """尽量从 hero 中提取朝向角（弧度）。没有则返回 0。"""
        candidates = ["yaw", "heading", "theta", "rotation", "dir"]
        for key in candidates:
            if key in hero:
                try:
                    val = float(hero[key])
                    if abs(val) > 2.0 * math.pi + 1e-6:
                        val = math.radians(val)
                    return val, True
                except Exception:
                    pass

        for key in candidates:
            sub = hero.get(key, None)
            if isinstance(sub, dict):
                for sub_key in ["value", "yaw", "theta", "angle"]:
                    if sub_key in sub:
                        try:
                            val = float(sub[sub_key])
                            if abs(val) > 2.0 * math.pi + 1e-6:
                                val = math.radians(val)
                            return val, True
                        except Exception:
                            pass

        return 0.0, False

    def _local_cell_to_global(self, hx, hz, ri, ci, vsize):
        """把局部图索引映射到全局格点。默认 row->z, col->x；若有 heading 则自动旋转。"""
        half = vsize // 2

        local_dx = ci - half   # col -> x
        local_dz = ri - half   # row -> z

        if self.has_heading:
            cos_yaw = math.cos(self.heading)
            sin_yaw = math.sin(self.heading)
            world_dx = int(round(cos_yaw * local_dx - sin_yaw * local_dz))
            world_dz = int(round(sin_yaw * local_dx + cos_yaw * local_dz))
        else:
            world_dx = local_dx
            world_dz = local_dz

        gx = hx + world_dx
        gz = hz + world_dz
        return gx, gz

    def _global_delta_to_local_index(self, dx, dz):
        """把全局相对位移映射到局部图索引。"""
        if self.has_heading:
            cos_yaw = math.cos(self.heading)
            sin_yaw = math.sin(self.heading)
            local_dx = cos_yaw * dx + sin_yaw * dz
            local_dz = -sin_yaw * dx + cos_yaw * dz
            local_dx = int(round(local_dx))
            local_dz = int(round(local_dz))
        else:
            local_dx = dx
            local_dz = dz

        r = self.VIEW_HALF + local_dz   # row <- z
        c = self.VIEW_HALF + local_dx   # col <- x
        return r, c

    def _write_png(self, image_array, save_path):
        """使用 Python 标准库把 RGB 图像写成 PNG，不依赖 PIL。"""
        image_array = np.ascontiguousarray(image_array, dtype=np.uint8)
        height, width, channels = image_array.shape
        if channels != 3:
            raise ValueError("PNG writer only supports RGB images with 3 channels.")

        def _png_chunk(chunk_type, data):
            chunk = chunk_type + data
            crc = binascii.crc32(chunk) & 0xffffffff
            return struct.pack("!I", len(data)) + chunk + struct.pack("!I", crc)

        raw = b"".join(b"\x00" + image_array[row].tobytes() for row in range(height))
        compressed = zlib.compress(raw, level=9)

        png = bytearray()
        png.extend(b"\x89PNG\r\n\x1a\n")
        png.extend(_png_chunk(b'IHDR', struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        png.extend(_png_chunk(b'IDAT', compressed))
        png.extend(_png_chunk(b'IEND', b''))

        with open(save_path, "wb") as f:
            f.write(png)

    def _save_astar_debug_grid(self, path):
        """保存当前全局地图和 A* 路径为 PNG 栅格图到当前目录。"""
        try:
            h, w = self.GRID_SIZE, self.GRID_SIZE
            debug_grid = np.full((h, w, 3), 255, dtype=np.uint8)

            explored = self.explored_mask == 1
            obstacles = explored & (self.passable_map == 0)
            unknown = self.explored_mask == 0

            debug_grid[unknown] = np.array([180, 180, 180], dtype=np.uint8)
            debug_grid[obstacles] = np.array([0, 0, 0], dtype=np.uint8)

            dirt_mask = self.global_dirt_map == 1
            debug_grid[dirt_mask] = np.array([210, 170, 80], dtype=np.uint8)

            for x, z in self.charger_cells:
                if self._in_bounds(x, z):
                    debug_grid[x, z] = np.array([0, 200, 0], dtype=np.uint8)

            for x, z in self.npc_positions:
                if self._in_bounds(x, z):
                    debug_grid[x, z] = np.array([220, 30, 30], dtype=np.uint8)

            for x, z in path:
                if self._in_bounds(x, z):
                    debug_grid[x, z] = np.array([30, 100, 255], dtype=np.uint8)

            sx, sz = self.cur_pos
            if self._in_bounds(sx, sz):
                debug_grid[sx, sz] = np.array([255, 220, 0], dtype=np.uint8)

            goal = self._get_nearest_charger_center()
            if goal is not None:
                gx, gz = goal
                if self._in_bounds(gx, gz):
                    debug_grid[gx, gz] = np.array([180, 0, 255], dtype=np.uint8)

            vis_grid = debug_grid.transpose(1, 0, 2)
            scale = 4
            vis_grid = np.repeat(np.repeat(vis_grid, scale, axis=0), scale, axis=1)

            self.astar_save_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_name = f"astar_debug_{self.astar_save_count:04d}_{timestamp}.png"
            os.makedirs(self.debug_save_dir, exist_ok=True)
            save_path = os.path.join(self.debug_save_dir, save_name)
            self._write_png(vis_grid, save_path)
        except Exception as e:
            print(f"[A* DEBUG SAVE ERROR] {e}")

    def _get_npc_positions(self, npcs):
        """Extract npc positions from frame_state['npcs'].""" 
        npc_positions = []
        for npc in npcs:
            pos = npc.get("pos", {})
            npc_positions.append((int(pos["x"]), int(pos["z"])))
        return npc_positions

    def _calc_nearest_npc_dist(self):
        """Compute Euclidean distance to the nearest official robot."""
        if not self.npc_positions:
            return 999.0

        hx, hz = self.cur_pos
        min_dist = 999.0
        for nx, nz in self.npc_positions:
            dist = np.sqrt((nx - hx) ** 2 + (nz - hz) ** 2)
            min_dist = min(min_dist, dist)
        return float(min_dist)

    def _calc_nearest_charger_dist(self):
        """计算当前位置到最近充电桩中心的欧氏距离。"""
        if not self.charger_positions:
            return 999.0

        hx, hz = self.cur_pos
        min_dist = 999.0
        for cx, cz in self.charger_positions:
            dist = ((cx - hx) ** 2 + (cz - hz) ** 2) ** 0.5
            min_dist = min(min_dist, dist)
        return float(min_dist)

    def _calc_nearest_obstacle_dist(self):
        """Find nearest obstacle distance from local view.

        从局部视野中找最近障碍物的欧氏距离。
        地图协议：0=障碍物/边界，1=已清扫，2=污渍
        """
        view = self._view_map
        if view is None:
            return 999.0

        obs_coords = np.argwhere(view == 0.0)
        if len(obs_coords) == 0:
            return 999.0

        center = self.VIEW_HALF
        dists = np.sqrt((obs_coords[:, 0] - center) ** 2 + (obs_coords[:, 1] - center) ** 2)

        dists = dists[dists > 0]
        if len(dists) == 0:
            return 999.0

        return float(np.min(dists))

    def _get_action_dirs(self):
        """动作编号与环境协议严格一致。"""
        return [
            (1, 0),    # 0: 右
            (1, -1),   # 1: 右上
            (0, -1),   # 2: 上
            (-1, -1),  # 3: 左上
            (-1, 0),   # 4: 左
            (-1, 1),   # 5: 左下
            (0, 1),    # 6: 下
            (1, 1),    # 7: 右下
        ]

    def _is_charger_cell_global(self, x, z):
        """判断全局坐标 (x, z) 是否属于充电桩区域。"""
        return (int(x), int(z)) in self.charger_cells

    def _build_collision_free_legal_action(self):
        """根据局部视野图构造无碰撞 legal_action。

        规则：
        - map_info == 0 表示障碍/边界，不可通行
        - 若目标格属于充电桩区域，则仍视为可通行
        - 若一步后会过于接近 NPC，则该动作视为不可行
        """
        if self._view_map is None:
            return [1] * 8

        center = self.VIEW_HALF
        action_dirs = self._get_action_dirs()
        legal = [1] * 8
        hx, hz = self.cur_pos

        for act, (dx, dz) in enumerate(action_dirs):
            r, c = self._global_delta_to_local_index(dx, dz)
            gx = hx + dx
            gz = hz + dz

            if not (0 <= r < self._view_map.shape[0] and 0 <= c < self._view_map.shape[1]):
                legal[act] = 0
                continue

            cell_value = int(self._view_map[r, c])
            if cell_value == 0.0 and not self._is_charger_cell_global(gx, gz):
                legal[act] = 0
                continue

            # 额外过滤：一步后如果离 NPC 太近，也禁掉
            for nx, nz in self.npc_positions:
                dist = math.sqrt((gx - nx) ** 2 + (gz - nz) ** 2)
                if dist < 1.5:
                    legal[act] = 0
                    break

        return [int(x) for x in legal]

    def pb2struct(self, env_obs, last_action):

        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        # Reset rule-guide state every step / 每步先重置规则引导状态
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])

        # Reset rule-guide state every step / 每步先重置规则引导状态
        self.rule_action = -1
        self.use_rule_guide = 0.0

        # 先保存上一时刻位置，再更新当前位置
        self.last_pos = self.cur_pos
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        # Optional heading / 朝向（如果环境提供）
        self.heading, self.has_heading = self._extract_heading_from_hero(hero)

        # Battery / 电量
        self.last_battery = self.battery
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Chargers
        organs = frame_state.get("organs", [])
        self.charger_positions = self._get_charger_positions(organs)
        self.charger_cells = self._get_charger_cells(organs)

        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist = self._calc_nearest_charger_dist()

        # NPCs
        npcs = frame_state.get("npcs", [])
        self.last_npc_positions = list(self.npc_positions)
        self.npc_positions = self._get_npc_positions(npcs)

        self.last_nearest_npc_dist = self.nearest_npc_dist
        self.nearest_npc_dist = self._calc_nearest_npc_dist()

        # Local view map (21×21)
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)
            self._update_exploration_and_dirt_map(hx, hz)
            self.passable_map

        # Update nearest obstacle distance after _view_map updated
        self.last_nearest_obstacle_dist = self.nearest_obstacle_dist
        self.nearest_obstacle_dist = self._calc_nearest_obstacle_dist()

        # Legal actions: recompute by map_info + charger cells
        self._legal_act = self._build_collision_free_legal_action()

        self._update_trajectory()
        self.terminated = bool(env_obs.get("terminated", False))
        self.collision_type = self._get_collision_type()

    def _get_charger_positions(self, organs):
        """提取所有充电桩中心位置。"""
        charger_positions = []
        for organ in organs:
            if int(organ.get("sub_type", -1)) == 1:
                pos = organ.get("pos", {})
                charger_positions.append((int(pos["x"]), int(pos["z"])))
        return charger_positions

    def _get_charger_cells(self, organs):
        """获取充电桩占据的所有格子坐标。

        默认按中心点 + w/h 展开。
        """
        charger_cells = set()
        for organ in organs:
            if int(organ.get("sub_type", -1)) != 1:
                continue

            pos = organ.get("pos", {})
            cx = int(pos["x"])
            cz = int(pos["z"])
            w = int(organ.get("w", 1))
            h = int(organ.get("h", 1))

            half_w = w // 2
            half_h = h // 2

            for x in range(cx - half_w, cx + half_w + 1):
                for z in range(cz - half_h, cz + half_h + 1):
                    charger_cells.add((x, z))

        return charger_cells

    def _get_collision_type(self):
        """Classify collision type when episode terminates."""
        if not self.terminated:
            return "none"

        hx, hz = self.cur_pos

        if self.nearest_npc_dist <= 0.5:
            return "npc"

        if (hx, hz) in self.charger_cells:
            return "charger"

        return "obstacle"

    def _update_passable(self, hx, hz):
        """Write local view into global passable map."""
        view = self._view_map
        vsize = view.shape[0]

        for ri in range(vsize):
            for ci in range(vsize):
                gx, gz = self._local_cell_to_global(hx, hz, ri, ci, vsize)

                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    cell = float(view[ri, ci])
                    if cell != 0.0:
                        self.passable_map[gx, gz] = 1
                    else:
                        self.passable_map[gx, gz] = 0
        self.passable_map

    def _update_exploration_and_dirt_map(self, hx, hz):
        """Update explored cells and global dirt memory from local view."""
        view = self._view_map
        vsize = view.shape[0]
        for ri in range(vsize):
            for ci in range(vsize):
                gx, gz = self._local_cell_to_global(hx, hz, ri, ci, vsize)
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    cell = int(view[ri, ci])
                    self.explored_mask[gx, gz] = 1
                    self.global_dirt_map[gx, gz] = 1 if cell == 2 else 0

    def _update_trajectory(self):
        """Update visited count and unique visited counter for current position."""
        x, z = self.cur_pos
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            if self.visited_counts[x, z] == 0:
                self.unique_visited += 1
            self.visited_counts[x, z] += 1

    def _get_battery_dead_penalty(self):
        """电量耗尽惩罚。"""
        if self.battery <= 0:
            return -2.0
        return 0.0

    def _get_local_view_feature(self):
        """Local view feature (441D): use full 21×21 field of view."""
        return (self._view_map / 2.0).flatten()

    def _downsample_grid(self, grid):
        """Average-pool a 128×128 map into a 16×16 coarse map."""
        block = self.GRID_SIZE // self.GLOBAL_REDUCED_SIZE
        coarse = grid.reshape(
            self.GLOBAL_REDUCED_SIZE,
            block,
            self.GLOBAL_REDUCED_SIZE,
            block,
        ).mean(axis=(1, 3))
        return coarse.astype(np.float32)

    def _get_global_map_feature(self):
        """Global map memory feature (1024D)."""
        explored_map = self.explored_mask.astype(np.float32)
        dirt_map = self.global_dirt_map.astype(np.float32)
        obstacle_map = explored_map * (1.0 - self.passable_map.astype(np.float32))
        visited_map = np.clip(self.visited_counts.astype(np.float32), 0.0, 5.0) / 5.0

        coarse_maps = [
            self._downsample_grid(explored_map),
            self._downsample_grid(dirt_map),
            self._downsample_grid(obstacle_map),
            self._downsample_grid(visited_map),
        ]
        return np.concatenate([m.flatten() for m in coarse_maps], axis=0)
    def _get_clean_streak_reward(self):
        """连续清扫奖励：辅助主任务，不喧宾夺主。"""
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)

        if cleaned_this_step > 0:
            self.clean_streak += 1
        else:
            self.clean_streak = 0

        if self.clean_streak <= 1:
            return 0.0

        reward = 0.05 * min(self.clean_streak - 1, 4)

        # 低电量时弱化，但不关闭
        if self._is_low_battery_mode():
            reward *= 0.5

        return float(reward)
    def _is_low_battery_mode(self):
        """是否进入低电量模式。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        return battery_ratio < self.low_battery_threshold

    def _get_no_move_penalty(self):
        """位置不变惩罚。

        规则：
        - 如果当前位置和上一时刻相同，则认为本步没有有效移动
        - 连续不动时惩罚逐步增大
        - 如果当前在充电桩上且电量正在恢复，则不惩罚，避免把“正常充电停留”也打掉
        """
        if self.step_no <= 0:
            return 0.0

        on_charger = (self.cur_pos in self.charger_cells) or (self.collision_type == "charger")
        battery_gain = self.battery - self.last_battery

        # 在充电桩上且电量正在增加，允许停留
        if on_charger and battery_gain > 0:
            self.no_move_count = 0
            return 0.0

        if self.cur_pos == self.last_pos:
            self.no_move_count += 1

            # 单步不动惩罚 + 连续不动增强惩罚
            penalty = -0.1 - 0.05 * min(self.no_move_count - 1, 4)
            return float(penalty)

        self.no_move_count = 0
        return 0.0

    def _get_global_state_feature(self):
        """Global state feature (13D)."""
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        battery_current_norm = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30

        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break

                if self._view_map is not None:
                    dx = x - hx
                    dz = z - hz
                    local_r, local_c = self._global_delta_to_local_index(dx, dz)

                    if 0 <= local_r < 21 and 0 <= local_c < 21:
                        cell = int(self._view_map[local_r, local_c])
                    else:
                        cell = 0

                    if cell == 2:
                        found = step
                        break

            ray_dirt.append(_norm(found, max_ray))

        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)
        self.dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        return np.array(
            [
                step_norm,
                battery_ratio,
                battery_current_norm,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                self.dirt_delta,
            ],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view."""
        view = self._view_map

        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2.0)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list)."""
        return list(self._legal_act)

    def _get_npc_velocity_feature(self):
        """npc速度特征: [vx_norm, vz_norm, speed_norm]."""
        if not self.npc_positions or not self.last_npc_positions:
            return np.zeros(3, dtype=np.float32)

        cx, cz = self.cur_pos
        cur_idx = int(np.argmin([((x - cx) ** 2 + (z - cz) ** 2) for x, z in self.npc_positions]))
        px, pz = self.npc_positions[cur_idx]

        last_idx = int(np.argmin([((x - px) ** 2 + (z - pz) ** 2) for x, z in self.last_npc_positions]))
        lx, lz = self.last_npc_positions[last_idx]

        vx = float(px - lx)
        vz = float(pz - lz)
        speed = float(np.sqrt(vx * vx + vz * vz))
        vmax = 5.0
        return np.array(
            [_norm(vx, vmax, -vmax), _norm(vz, vmax, -vmax), _norm(speed, vmax)],
            dtype=np.float32
        )

    def _get_npc_approach_direction_feature(self):
        """npc接近方向特征: [dir_x, dir_z, approaching]."""
        if not self.npc_positions:
            return np.zeros(3, dtype=np.float32)

        hx, hz = self.cur_pos
        nearest_idx = int(np.argmin([((x - hx) ** 2 + (z - hz) ** 2) for x, z in self.npc_positions]))
        nx, nz = self.npc_positions[nearest_idx]
        dx = float(nx - hx)
        dz = float(nz - hz)
        dist = float(np.sqrt(dx * dx + dz * dz))
        if dist > 1e-6:
            dir_x = dx / dist
            dir_z = dz / dist
        else:
            dir_x, dir_z = 0.0, 0.0

        approaching = 1.0 if self.nearest_npc_dist < self.last_nearest_npc_dist else 0.0
        return np.array([dir_x, dir_z, approaching], dtype=np.float32)

    def _get_trajectory_feature(self):
        """轨迹特征: [visit_norm, revisit_flag, unique_visit_ratio]."""
        x, z = self.cur_pos
        visit = 0
        if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
            visit = int(self.visited_counts[x, z])

        visit_norm = _norm(min(visit, 10), 10)
        revisit_flag = 1.0 if visit > 1 else 0.0
        total_cells = float(self.GRID_SIZE * self.GRID_SIZE)
        unique_visit_ratio = float(self.unique_visited) / total_cells
        return np.array([visit_norm, revisit_flag, unique_visit_ratio], dtype=np.float32)

    def _get_global_dirt_distribution_feature(self):
        """全局污渍统计特征: [known_dirt_ratio, nearby_dirt_density, nearest_dirt_norm]."""
        explored = float(np.sum(self.explored_mask))
        known_dirt = float(np.sum(self.global_dirt_map))
        known_dirt_ratio = known_dirt / max(explored, 1.0)

        hx, hz = self.cur_pos
        r = 5
        x0, x1 = max(hx - r, 0), min(hx + r + 1, self.GRID_SIZE)
        z0, z1 = max(hz - r, 0), min(hz + r + 1, self.GRID_SIZE)
        near_patch = self.global_dirt_map[x0:x1, z0:z1]
        nearby_dirt_density = float(np.mean(near_patch)) if near_patch.size > 0 else 0.0

        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 200.0)
        return np.array([known_dirt_ratio, nearby_dirt_density, nearest_dirt_norm], dtype=np.float32)

    def _get_exploration_uncleaned_feature(self):
        """探索/未清扫特征: [explore_ratio, unknown_ratio, uncleaned_ratio]."""
        total_cells = float(self.GRID_SIZE * self.GRID_SIZE)
        explored = float(np.sum(self.explored_mask))
        explore_ratio = explored / total_cells
        unknown_ratio = 1.0 - explore_ratio

        known_dirt = float(np.sum(self.global_dirt_map))
        uncleaned_ratio = known_dirt / max(explored, 1.0)
        return np.array([explore_ratio, unknown_ratio, uncleaned_ratio], dtype=np.float32)

    def _get_low_battery_feature(self):
        """低电量回充特征 (8D)."""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        is_low_battery = 1.0 if battery_ratio < self.low_battery_threshold else 0.0
        is_very_low_battery = 1.0 if battery_ratio < self.very_low_battery_threshold else 0.0

        nearest_charger_dist_norm = _norm(self.nearest_charger_dist, 180.0)

        charger_dir_x = 0.0
        charger_dir_z = 0.0
        if self.charger_positions:
            hx, hz = self.cur_pos
            nearest_idx = int(np.argmin([
                (cx - hx) ** 2 + (cz - hz) ** 2 for cx, cz in self.charger_positions
            ]))
            cx, cz = self.charger_positions[nearest_idx]
            dx = float(cx - hx)
            dz = float(cz - hz)
            dist = float(np.sqrt(dx * dx + dz * dz))
            if dist > 1e-6:
                charger_dir_x = dx / dist
                charger_dir_z = dz / dist

        charger_approaching = 1.0 if self.nearest_charger_dist < self.last_nearest_charger_dist else 0.0
        on_charger = 1.0 if (self.cur_pos in self.charger_cells) or (self.collision_type == "charger") else 0.0

        self.on_charger = on_charger

        return np.array(
            [
                battery_ratio,
                is_low_battery,
                is_very_low_battery,
                nearest_charger_dist_norm,
                charger_dir_x,
                charger_dir_z,
                charger_approaching,
                on_charger,
            ],
            dtype=np.float32,
        )

    def feature_process(self, env_obs, last_action):
        """Generate feature vector, legal action mask, and scalar reward."""
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()          # 441D
        global_map = self._get_global_map_feature()          # 1024D
        global_state = self._get_global_state_feature()      # 13D
        legal_action = self.get_legal_action()               # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)
        npc_velocity = self._get_npc_velocity_feature()      # 3D
        npc_approach = self._get_npc_approach_direction_feature()  # 3D
        trajectory = self._get_trajectory_feature()          # 3D
        global_dirt = self._get_global_dirt_distribution_feature() # 3D
        exploration = self._get_exploration_uncleaned_feature()    # 3D
        low_battery_feat = self._get_low_battery_feature()   # 8D

        feature = np.concatenate(
            [
                local_view,
                global_map,
                global_state,
                legal_arr,
                npc_velocity,
                npc_approach,
                trajectory,
                global_dirt,
                exploration,
                low_battery_feat,
            ]
        )  # 1509D

        reward, reward_info = self.reward_process()
        self.reward_info = reward_info
        return feature, legal_action, reward

    def _get_cleaning_reward(self):
        """清扫奖励：主任务始终是清扫。"""
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        if cleaned_this_step <= 0:
            return 0.0

        # 主奖励提高，让模型明确知道“扫到污渍”才是最重要的事
        reward = 0.8 * cleaned_this_step

        # 低电量时仍保留清扫奖励，但适度减弱，避免和回充目标完全冲突
        if self._is_low_battery_mode():
            reward *= 0.7

        return float(reward)
    def _get_dirt_approach_reward(self):
        """接近污渍奖励：引导模型朝污渍移动，但权重小于真实清扫。"""
        if self.step_no > 0 and self.dirt_delta > 0:
            reward = 0.04
            if self._is_low_battery_mode():
                reward *= 0.5
            return float(reward)
        return 0.0
    def _get_npc_avoid_reward(self):
        """官方机器人避碰惩罚：更强、更前瞻。"""
        d = self.nearest_npc_dist

        if d <= 0.5:
            return -2.0
        elif d <= 1.0:
            return -1.0
        elif d <= 1.5:
            return -0.4
        elif d <= 2.5:
            return -0.1
        return 0.0

    def _get_low_battery_charger_approach_reward(self):

        """低电量时接近充电桩奖励。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        if self.step_no <= 0 or battery_ratio >= self.low_battery_threshold:
            return 0.0

        delta = float(self.last_nearest_charger_dist - self.nearest_charger_dist)
        reward = 0.12 * np.clip(delta, -2.0, 2.0)
        return float(reward)

    def _get_low_battery_charging_reward(self):
        """低电量时到达充电桩并实际充电的奖励。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        on_charger = (self.cur_pos in self.charger_cells) or (self.collision_type == "charger")

        # 只有低电量时，充电才有正奖励
        if battery_ratio >= self.low_battery_threshold or not on_charger:
            return 0.0

        reward = 0.8
        battery_gain = self.battery - self.last_battery

        # 真正开始回血，再给额外奖励
        if battery_gain > 0:
            reward += 0.25 + 0.03 * min(battery_gain, 10)

        return float(reward)
    def _get_low_battery_leave_charger_penalty(self):
        """低电量时远离充电桩惩罚。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        if self.step_no <= 0 or battery_ratio >= self.low_battery_threshold:
            return 0.0

        delta = float(self.nearest_charger_dist - self.last_nearest_charger_dist)
        if delta > 0:
            return -0.05 * min(delta, 2.0)
        return 0.0

    def _is_current_on_cleaned_region(self):
        """判断当前位置是否处于已清扫区域。"""
        x, z = self.cur_pos

        if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
            return False

        # 充电桩不算“回头路惩罚区域”
        if (x, z) in self.charger_cells:
            return False

        # 当前格如果在全局污渍图里仍是污渍，则不是已清扫区域
        if self.global_dirt_map[x, z] == 1:
            return False

        # 已探索、非障碍、非污渍，可视为已清扫区域
        explored = self.explored_mask[x, z] == 1
        passable = self.passable_map[x, z] == 1

        return bool(explored and passable)

    def _get_cleaned_region_revisit_penalty(self):
        """连续走在已清扫区域时的递增惩罚。"""
        if self.step_no <= 0:
            return 0.0

        # 没移动，不在这里罚，交给 no_move_penalty 处理
        if self.cur_pos == self.last_pos:
            self.cleaned_region_walk_streak = 0
            return 0.0

        if self._is_current_on_cleaned_region():
            self.cleaned_region_walk_streak += 1
            penalty = -min(0.02 * self.cleaned_region_walk_streak, 0.12)
            return float(penalty)

        self.cleaned_region_walk_streak = 0
        return 0.0

    def _get_high_battery_charger_penalty(self):
        """高电量时赖在充电桩附近/上面的惩罚。"""
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
        on_charger = (self.cur_pos in self.charger_cells) or (self.collision_type == "charger")

        if not on_charger:
            return 0.0

        # 低电量允许充电，不惩罚
        if battery_ratio < self.low_battery_threshold:
            return 0.0

        # 电量越高，越不应该赖在充电桩
        if battery_ratio >= 0.8:
            return -0.5
        elif battery_ratio >= 0.6:
            return -0.4
        else:
            return -0.3

    def _get_step_penalty(self):
        """时间步惩罚。"""
        return -0.001

    def get_rule_action(self):
        """返回低电量时基于 A* 的规则动作。

        目标：
        - 接近最近充电桩中心
        - 而不是仅仅进入 charger 区域

        注意：
        - 本函数只在外部“规则启用”时调用
        - 不在 feature_process() 中主动调用
        """
        battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)

        if battery_ratio >= self.low_battery_threshold:
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        if not self.charger_positions:
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        goal = self._get_nearest_charger_center()
        if goal is None:
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        gx, gz = goal
        hx, hz = self.cur_pos

        # 如果已经足够接近充电桩中心，就不再强制规则动作
        if math.sqrt((hx - gx) ** 2 + (hz - gz) ** 2) <= float(self.goal_center_tolerance):
            self.rule_action = -1
            self.use_rule_guide = 0.0
            return -1

        path = self._astar_to_nearest_charger()
        # self._save_astar_debug_grid(path)
        # self._save_global_dirt_map()

        if len(path) >= 2:
            self.cur_pos
            cur = path[0]
            nxt = path[1]
            dx = int(nxt[0] - cur[0])
            dz = int(nxt[1] - cur[1])

            act = self._dir_to_action(dx, dz)

            if act >= 0 and self._legal_act[act] == 1:
                self.rule_action = act
                self.use_rule_guide = 1.0
                return act

        # A* 失败时回退到局部贪心
        fallback_act = self._get_greedy_charger_action_fallback()
        if fallback_act >= 0:
            self.rule_action = fallback_act
            self.use_rule_guide = 1.0
            return fallback_act

        self.rule_action = -1
        self.use_rule_guide = 0.0
        return -1

    def _dir_to_action(self, dx, dz):
        """将一步位移映射为动作编号。"""
        action_dirs = self._get_action_dirs()
        for act, (adx, adz) in enumerate(action_dirs):
            if dx == adx and dz == adz:
                return act
        return -1

    def _in_bounds(self, x, z):
        """判断是否在地图范围内。"""
        return 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE

    def _is_passable_for_planning(self, x, z):
        """A* 规划时判断格子是否可通行。"""
        if not self._in_bounds(x, z):
            return False

        if (int(x), int(z)) in self.charger_cells:
            return True

        return bool(self.passable_map[x, z] == 1)

    def _get_nearest_charger_center(self):
        """获取离当前位置最近的充电桩中心。"""
        if not self.charger_positions:
            return None

        hx, hz = self.cur_pos
        nearest_idx = int(np.argmin([
            (cx - hx) ** 2 + (cz - hz) ** 2 for cx, cz in self.charger_positions
        ]))
        cx, cz = self.charger_positions[nearest_idx]
        return (int(cx), int(cz))

    def _astar_heuristic_to_center(self, x, z, goal):
        """A* 启发函数：到充电桩中心的 octile distance。"""
        gx, gz = goal
        dx = abs(gx - x)
        dz = abs(gz - z)
        return float((dx + dz) + (math.sqrt(2.0) - 2.0) * min(dx, dz))

    def _reconstruct_path(self, came_from, current):
        """从 came_from 回溯路径。"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _get_npc_penalty_for_cell(self, x, z):
        """计算某个格子的 NPC 风险代价。"""
        if not self.npc_positions:
            return 0.0

        penalty = 0.0

        # 当前 NPC 位置风险
        for nx, nz in self.npc_positions:
            dist = math.sqrt((x - nx) ** 2 + (z - nz) ** 2)

            if dist < 1.0:
                penalty += 100.0
            elif dist < 1.5:
                penalty += 8.0
            elif dist < 2.5:
                penalty += 2.0
            elif dist < 3.5:
                penalty += 0.5

        # 短时未来风险（基于上一帧位置粗略预测）
        if self.last_npc_positions:
            used = set()
            for cx, cz in self.npc_positions:
                best_j = -1
                best_d = float("inf")
                for j, (lx, lz) in enumerate(self.last_npc_positions):
                    if j in used:
                        continue
                    d = (cx - lx) ** 2 + (cz - lz) ** 2
                    if d < best_d:
                        best_d = d
                        best_j = j

                if best_j >= 0:
                    used.add(best_j)
                    lx, lz = self.last_npc_positions[best_j]
                    vx = cx - lx
                    vz = cz - lz

                    for t, w in [(1.0, 0.8), (2.0, 0.5)]:
                        px = cx + t * vx
                        pz = cz + t * vz
                        dist = math.sqrt((x - px) ** 2 + (z - pz) ** 2)

                        if dist < 1.0:
                            penalty += 20.0 * w
                        elif dist < 2.0:
                            penalty += 5.0 * w
                        elif dist < 3.0:
                            penalty += 1.0 * w

        return float(penalty)

    def _astar_to_nearest_charger(self):
        """使用 A* 朝最近充电桩中心规划路径。

        特点：
        - 目标是 charger center，不是 charger_cells
        - 终止条件是“足够接近中心”
        - 支持 8 邻域
        - 直走/斜走代价不同
        - 防止对角穿墙
        - 低电量时尽量顺路经过污渍
        - 同时主动避让 NPC
        """
        start = (int(self.cur_pos[0]), int(self.cur_pos[1]))

        goal = self._get_nearest_charger_center()
        if goal is None:
            return []

        gx, gz = goal
        goal_radius = float(self.goal_center_tolerance)

        # 已经足够接近中心
        if math.sqrt((start[0] - gx) ** 2 + (start[1] - gz) ** 2) <= goal_radius:
            return [start]

        if not self._is_passable_for_planning(start[0], start[1]):
            return []

        open_heap = []
        heapq.heappush(
            open_heap,
            (self._astar_heuristic_to_center(start[0], start[1], goal), 0.0, start)
        )

        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        action_dirs = self._get_action_dirs()

        while open_heap:
            _, cur_g, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            closed.add(current)

            cx, cz = current

            # 终止条件：接近充电桩中心
            if math.sqrt((cx - gx) ** 2 + (cz - gz) ** 2) <= goal_radius:
                return self._reconstruct_path(came_from, current)

            for dx, dz in action_dirs:
                nx = cx + dx
                nz = cz + dz
                nxt = (nx, nz)

                if not self._is_passable_for_planning(nx, nz):
                    continue

                # 防止对角穿墙
                if dx != 0 and dz != 0:
                    side1 = (cx + dx, cz)
                    side2 = (cx, cz + dz)

                    side1_ok = self._is_passable_for_planning(side1[0], side1[1])
                    side2_ok = self._is_passable_for_planning(side2[0], side2[1])

                    if not (side1_ok and side2_ok):
                        continue

                # 基础步长代价
                base_cost = math.sqrt(2.0) if (dx != 0 and dz != 0) else 1.0

                # 污渍偏好：低电量但还没极低时，允许顺路踩点污渍
                battery_ratio = float(self.battery) / max(float(self.battery_max), 1.0)
                dirt_weight = 0.0 if battery_ratio < self.very_low_battery_threshold else self.dirt_prefer_weight

                dirt_bonus = 0.0
                if self.global_dirt_map[nx, nz] == 1:
                    dirt_bonus += dirt_weight

                # 轻微减少重复走
                revisit_penalty = 0.0
                visit_cnt = int(self.visited_counts[nx, nz])
                if visit_cnt > 0:
                    revisit_penalty = 0.03 * min(visit_cnt, 5)

                # NPC 风险惩罚
                npc_penalty = self._get_npc_penalty_for_cell(nx, nz)

                # 最终代价
                step_cost = base_cost - dirt_bonus + revisit_penalty + npc_penalty
                step_cost = max(0.05, step_cost)

                tentative_g = cur_g + step_cost

                if tentative_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = tentative_g
                    came_from[nxt] = current
                    f = tentative_g + self._astar_heuristic_to_center(nx, nz, goal)
                    heapq.heappush(open_heap, (f, tentative_g, nxt))

        return []

    def _get_greedy_charger_action_fallback(self):
        """A* 失败时的回退策略：保留你原来的局部贪心规则。"""
        if not self.charger_positions:
            return -1

        hx, hz = self.cur_pos

        nearest_idx = int(np.argmin([
            (cx - hx) ** 2 + (cz - hz) ** 2 for cx, cz in self.charger_positions
        ]))
        cx, cz = self.charger_positions[nearest_idx]

        dx = float(cx - hx)
        dz = float(cz - hz)

        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return -1

        dist = np.sqrt(dx * dx + dz * dz)
        target_dir = (dx / dist, dz / dist)

        action_dirs = self._get_action_dirs()

        best_act = -1
        best_score = -1e9

        for act, (ax, az) in enumerate(action_dirs):
            if self._legal_act[act] != 1:
                continue

            move_len = np.sqrt(ax * ax + az * az)
            move_dir = (ax / move_len, az / move_len)

            direction_score = move_dir[0] * target_dir[0] + move_dir[1] * target_dir[1]

            nx = hx + ax
            nz = hz + az
            new_dist = np.sqrt((cx - nx) ** 2 + (cz - nz) ** 2)
            approach_bonus = 1.0 if new_dist < dist else -1.0

            score = direction_score + 0.5 * approach_bonus

            if score > best_score:
                best_score = score
                best_act = act

        return best_act

    def _save_global_dirt_map(self):
        """保存全局污渍图为 PNG，方便检查污渍累计是否合理。"""
        try:
            h, w = self.GRID_SIZE, self.GRID_SIZE
            dirt_grid = np.full((h, w, 3), 255, dtype=np.uint8)

            # 未探索区域：灰色
            unknown = self.explored_mask == 0
            dirt_grid[unknown] = np.array([180, 180, 180], dtype=np.uint8)

            # 已探索且无污渍：白色
            explored_clean = (self.explored_mask == 1) & (self.global_dirt_map == 0)
            dirt_grid[explored_clean] = np.array([255, 255, 255], dtype=np.uint8)

            # 已探索且有污渍：棕黄色
            dirt_mask = self.global_dirt_map == 1
            dirt_grid[dirt_mask] = np.array([210, 170, 80], dtype=np.uint8)

            # 障碍：黑色
            obstacle_mask = (self.explored_mask == 1) & (self.passable_map == 0)
            dirt_grid[obstacle_mask] = np.array([0, 0, 0], dtype=np.uint8)

            # 充电桩：绿色
            for x, z in self.charger_cells:
                if self._in_bounds(x, z):
                    dirt_grid[x, z] = np.array([0, 200, 0], dtype=np.uint8)

            # 机器人当前位置：蓝色
            sx, sz = self.cur_pos
            if self._in_bounds(sx, sz):
                dirt_grid[sx, sz] = np.array([30, 100, 255], dtype=np.uint8)

            vis_grid = dirt_grid.transpose(1, 0, 2)
            scale = 4
            vis_grid = np.repeat(np.repeat(vis_grid, scale, axis=0), scale, axis=1)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_name = f"dirt_debug_{self.astar_save_count:04d}_{timestamp}.png"
            os.makedirs(self.debug_save_dir, exist_ok=True)
            save_path = os.path.join(self.debug_save_dir, save_name)
            self._write_png(vis_grid, save_path)
        except Exception as e:
            print(f"[DIRT DEBUG SAVE ERROR] {e}")

    def reward_process(self):
        """Compute reward with cleaning-first design.

        设计目标：
        - 主任务始终是清扫
        - 低电量时允许并鼓励合理回充
        - 高电量赖在充电桩上要惩罚
        - 连续走在已清扫区域时给递增惩罚
        """
        low_battery_mode = self._is_low_battery_mode()

        # ===== 主任务相关 =====
        cleaning_reward = self._get_cleaning_reward()
        clean_streak_reward = self._get_clean_streak_reward()
        dirt_approach_reward = self._get_dirt_approach_reward()

        # ===== 安全 / 行为约束 =====
        npc_avoid_reward = self._get_npc_avoid_reward()
        cleaned_region_revisit_penalty = self._get_cleaned_region_revisit_penalty()
        step_penalty = self._get_step_penalty()
        no_move_penalty = 0;
        # ===== 充电相关 =====
        charger_approach_reward = self._get_low_battery_charger_approach_reward()
        leave_charger_penalty = self._get_low_battery_leave_charger_penalty()
        charging_reward = self._get_low_battery_charging_reward()
        high_battery_charger_penalty = self._get_high_battery_charger_penalty()
        battery_dead_penalty = self._get_battery_dead_penalty()

        obstacle_reward = npc_avoid_reward

        total_reward = (
            cleaning_reward
            + clean_streak_reward
            + dirt_approach_reward
            + obstacle_reward
            + cleaned_region_revisit_penalty
            + step_penalty
            + battery_dead_penalty
            +no_move_penalty
        )

        if low_battery_mode:
            total_reward += (
                charger_approach_reward
                + leave_charger_penalty
                + charging_reward
            )
        else:
            total_reward += high_battery_charger_penalty

        reward_info = {
            "reward": float(total_reward),
            "cleaning_reward": float(cleaning_reward),
            "clean_streak_reward": float(clean_streak_reward),
            "dirt_approach_reward": float(dirt_approach_reward),
            "obstacle_reward": float(obstacle_reward),
            "npc_avoid_reward": float(npc_avoid_reward),
            "charger_approach_reward": float(charger_approach_reward),
            "leave_charger_penalty": float(leave_charger_penalty),
            "charging_reward": float(charging_reward),
            "high_battery_charger_penalty": float(high_battery_charger_penalty),
            "battery_dead_penalty": float(battery_dead_penalty),
            "no_move_penalty": float(no_move_penalty),
            "cleaned_region_revisit_penalty": float(cleaned_region_revisit_penalty),
            "cleaned_region_walk_streak": int(self.cleaned_region_walk_streak),
            "step_penalty": float(step_penalty),
            "clean_streak": int(self.clean_streak),
            "no_move_count": int(self.no_move_count),
            "rule_action": int(self.rule_action),
            "use_rule_guide": float(self.use_rule_guide),
            "low_battery_mode": float(low_battery_mode),
            "battery_ratio": float(self.battery / max(self.battery_max, 1)),
            "nearest_charger_dist": float(self.nearest_charger_dist),
        }

        return total_reward, reward_info
