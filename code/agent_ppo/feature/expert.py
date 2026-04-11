"""Expert policy: minimal safety filter + A* charger navigation.

Only intervene when RL clearly fails:
  - Layer 1: NPC collision safety (block stepping ON NPC only)
  - Layer 2: Battery emergency → A* to nearest charger
"""
from __future__ import annotations

import heapq
import numpy as np
from collections import deque


class ExpertPolicy:
    GRID = 128
    DELTAS = (
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    )

    CHARGE_DIST_MULT = 2.0
    CHARGE_SAFETY_MARGIN = 40

    def __init__(self):
        self._charger_list = []

    def update_chargers(self, prep):
        self._charger_list = []
        for organ in prep._organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue
            pos = organ.get("pos") or {}
            cx, cz = int(pos.get("x", 0)), int(pos.get("z", 0))
            w = max(int(organ.get("w", 3)) // 2, 1)
            h = max(int(organ.get("h", 3)) // 2, 1)
            self._charger_list.append((cx, cz, w, h))

    def filter_actions(self, prep, legal_action):
        """Layer 1: Only block stepping directly onto NPC."""
        legal = list(legal_action)
        hx, hz = prep.cur_pos

        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            for idx, (dx, dz) in enumerate(self.DELTAS):
                if not legal[idx]:
                    continue
                nx2, nz2 = hx + dx, hz + dz
                if nx2 == nx and nz2 == nz:
                    legal[idx] = 0

        if sum(legal) == 0:
            return list(legal_action)
        return legal

    def get_override(self, prep, legal_action):
        """Layer 2: Only override for battery emergency."""
        self.update_chargers(prep)

        threshold = prep.nearest_charger_dist * self.CHARGE_DIST_MULT + self.CHARGE_SAFETY_MARGIN
        if prep.battery <= threshold:
            act = self._astar_to_charger(prep)
            if act is not None and legal_action[act]:
                return True, act

        return False, None

    # ------------------------------------------------------------------
    # A* pathfinding
    # ------------------------------------------------------------------

    def _astar(self, prep, is_goal, h_func):
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return None
        G = self.GRID
        counter = 0
        open_set = []
        g_score = np.full((G, G), 1e9, dtype=np.float32)
        g_score[sx, sz] = 0
        closed = np.zeros((G, G), dtype=np.bool_)
        heapq.heappush(open_set, (h_func(sx, sz), counter, sx, sz, -1))

        while open_set:
            f, _, x, z, first = heapq.heappop(open_set)
            if closed[x, z]:
                continue
            closed[x, z] = True
            if is_goal(x, z):
                return first if first >= 0 else 0
            cur_g = g_score[x, z]
            for idx, (dx, dz) in enumerate(self.DELTAS):
                nx, nz = x + dx, z + dz
                if not (0 <= nx < G and 0 <= nz < G) or closed[nx, nz]:
                    continue
                if not self._can_move(prep, x, z, nx, nz):
                    continue
                new_g = cur_g + 1.0
                if new_g < g_score[nx, nz]:
                    g_score[nx, nz] = new_g
                    counter += 1
                    fa = first if first >= 0 else idx
                    heapq.heappush(open_set, (new_g + h_func(nx, nz), counter, nx, nz, fa))
        return None

    def _bfs(self, prep, is_goal):
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return None
        G = self.GRID
        visited = np.zeros((G, G), dtype=np.bool_)
        visited[sx, sz] = True
        queue = deque()
        for idx, (dx, dz) in enumerate(self.DELTAS):
            nx, nz = sx + dx, sz + dz
            if 0 <= nx < G and 0 <= nz < G and not visited[nx, nz] and self._can_move(prep, sx, sz, nx, nz):
                visited[nx, nz] = True
                if is_goal(nx, nz): return idx
                queue.append((nx, nz, idx))
        while queue:
            x, z, first = queue.popleft()
            for dx, dz in self.DELTAS:
                nx, nz = x + dx, z + dz
                if 0 <= nx < G and 0 <= nz < G and not visited[nx, nz] and self._can_move(prep, x, z, nx, nz):
                    visited[nx, nz] = True
                    if is_goal(nx, nz): return first
                    queue.append((nx, nz, first))
        return None

    def _astar_to_charger(self, prep):
        chargers = self._charger_list
        if not chargers:
            return None
        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in chargers)
        def h_func(x, z):
            return min(max(abs(x - cx) - w, abs(z - cz) - h, 0) for cx, cz, w, h in chargers)
        act = self._astar(prep, is_goal, h_func)
        if act is None:
            act = self._bfs(prep, is_goal)
        return act

    def _can_move(self, prep, fx, fz, nx, nz):
        G = self.GRID
        if not (0 <= nx < G and 0 <= nz < G):
            return False
        if prep.explored_map[nx, nz] > 0.5 and prep.passable_map[nx, nz] < 0.5:
            return False
        dx, dz = nx - fx, nz - fz
        if dx != 0 and dz != 0:
            for cx, cz in ((fx + dx, fz), (fx, fz + dz)):
                if 0 <= cx < G and 0 <= cz < G:
                    if prep.explored_map[cx, cz] > 0.5 and prep.passable_map[cx, cz] < 0.5:
                        return False
        return True
