"""Expert policy: NPC safety filter + weighted Dijkstra charger navigation.

Two layers:
  - Layer 1: NPC collision safety (always active, blocks dangerous NPC directions)
  - Layer 2: Battery emergency → Dijkstra with NPC danger cost map to charger

Dynamic replanning: cost map rebuilt each step with updated vision + NPC positions.
Unexplored cells are impassable; NPCs add exponential danger cost to nearby cells.
"""
from __future__ import annotations

import heapq
import numpy as np


class ExpertPolicy:
    GRID = 128
    DELTAS = (
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    )

    CHARGE_DIST_MULT = 2.0
    CHARGE_SAFETY_MARGIN = 40

    # Cost-map parameters
    _INF_COST = 1e6
    _NPC_DANGER_MAX = 15.0       # peak danger cost at NPC position
    _NPC_DANGER_DECAY = 2.0      # exponential decay rate
    _NPC_DANGER_RADIUS = 8       # Chebyshev radius of danger zone

    def __init__(self):
        self._charger_list = []

    # ------------------------------------------------------------------
    # Charger tracking
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Layer 1: NPC safety filter
    # ------------------------------------------------------------------

    def filter_actions(self, prep, legal_action):
        """Block stepping onto NPC and moving directly toward nearby NPCs."""
        legal = list(legal_action)
        hx, hz = prep.cur_pos

        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            ndx, ndz = nx - hx, nz - hz
            npc_dist = max(abs(ndx), abs(ndz))
            for idx, (dx, dz) in enumerate(self.DELTAS):
                if not legal[idx]:
                    continue
                nx2, nz2 = hx + dx, hz + dz
                # Block stepping directly onto NPC
                if nx2 == nx and nz2 == nz:
                    legal[idx] = 0
                    continue
                # Block moving directly toward NPC within distance 3
                if npc_dist <= 3:
                    sx = (1 if ndx > 0 else -1) if ndx != 0 else 0
                    sz = (1 if ndz > 0 else -1) if ndz != 0 else 0
                    if dx == sx and dz == sz:
                        legal[idx] = 0

        if sum(legal) == 0:
            return list(legal_action)
        return legal

    # ------------------------------------------------------------------
    # Layer 2: Battery emergency override
    # ------------------------------------------------------------------

    def get_override(self, prep, legal_action):
        """Override for battery emergency with NPC-aware Dijkstra planning."""
        self.update_chargers(prep)

        threshold = prep.nearest_charger_dist * self.CHARGE_DIST_MULT + self.CHARGE_SAFETY_MARGIN
        if prep.battery <= threshold:
            act = self._plan_to_charger(prep)
            if act is not None and legal_action[act]:
                return True, act

        return False, None

    # ------------------------------------------------------------------
    # Cost-map construction
    # ------------------------------------------------------------------

    def _build_cost_map(self, prep, npc_weight=1.0):
        """Build 128x128 weighted cost map.

        Cell costs:
          - 1.0            : explored + passable
          - _INF_COST      : unexplored or impassable (wall)
          - 1.0 + danger   : passable cell near NPC (danger decays exponentially)

        npc_weight=0 produces a binary cost map (no NPC avoidance).
        """
        G = self.GRID
        cost = np.ones((G, G), dtype=np.float32)

        impassable = (prep.explored_map < 0.5) | (prep.passable_map < 0.5)
        cost[impassable] = self._INF_COST

        if npc_weight > 0:
            for npc in prep._npcs:
                pos = npc.get("pos") or {}
                nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
                R = self._NPC_DANGER_RADIUS
                x0, x1 = max(nx - R, 0), min(nx + R + 1, G)
                z0, z1 = max(nz - R, 0), min(nz + R + 1, G)

                lx = np.arange(x0, x1, dtype=np.float32)
                lz = np.arange(z0, z1, dtype=np.float32)
                xx, zz = np.meshgrid(lx, lz, indexing='ij')
                dist = np.maximum(np.abs(xx - nx), np.abs(zz - nz))
                danger = npc_weight * self._NPC_DANGER_MAX * np.exp(-dist / self._NPC_DANGER_DECAY)
                danger[dist > R] = 0.0

                region = cost[x0:x1, z0:z1]
                safe = region < self._INF_COST
                region[safe] += danger[safe]

        return cost

    # ------------------------------------------------------------------
    # Weighted A* search (Dijkstra with admissible heuristic)
    # ------------------------------------------------------------------

    def _weighted_astar(self, prep, cost_map, is_goal, h_func):
        """A* on weighted cost map.  Returns the first-action index to reach goal."""
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return None

        G = self.GRID
        INF = 1e9
        dist_arr = np.full((G, G), INF, dtype=np.float32)
        dist_arr[sx, sz] = 0.0
        first_act = np.full((G, G), -1, dtype=np.int8)
        closed = np.zeros((G, G), dtype=np.bool_)

        counter = 0
        heap = [(h_func(sx, sz), counter, sx, sz)]

        while heap:
            f, _, x, z = heapq.heappop(heap)
            if closed[x, z]:
                continue
            closed[x, z] = True
            if is_goal(x, z):
                fa = int(first_act[x, z])
                return fa if fa >= 0 else 0

            cur_d = dist_arr[x, z]
            for idx, (dx, dz) in enumerate(self.DELTAS):
                nx, nz = x + dx, z + dz
                if not (0 <= nx < G and 0 <= nz < G) or closed[nx, nz]:
                    continue
                step_cost = cost_map[nx, nz]
                if step_cost >= self._INF_COST:
                    continue
                # Diagonal: at least one adjacent cardinal cell must be passable
                if dx != 0 and dz != 0:
                    c1 = (0 <= x + dx < G) and cost_map[x + dx, z] < self._INF_COST
                    c2 = (0 <= z + dz < G) and cost_map[x, z + dz] < self._INF_COST
                    if not c1 and not c2:
                        continue
                move = 1.414 if (dx != 0 and dz != 0) else 1.0
                new_d = cur_d + step_cost * move
                if new_d < dist_arr[nx, nz]:
                    dist_arr[nx, nz] = new_d
                    first_act[nx, nz] = first_act[x, z] if first_act[x, z] >= 0 else idx
                    counter += 1
                    heapq.heappush(heap, (new_d + h_func(nx, nz), counter, nx, nz))

        return None

    # ------------------------------------------------------------------
    # Charger path planning
    # ------------------------------------------------------------------

    def _charger_heuristic(self, x, z):
        """Chebyshev distance to nearest charger boundary (admissible)."""
        return min(
            max(abs(x - cx) - w, abs(z - cz) - h, 0)
            for cx, cz, w, h in self._charger_list
        )

    def _plan_to_charger(self, prep):
        """Plan path to nearest charger with graduated fallback.

        1. Full NPC danger avoidance
        2. Reduced NPC avoidance (half danger weight)
        3. No NPC avoidance (binary cost map)
        """
        chargers = self._charger_list
        if not chargers:
            return None

        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h
                       for cx, cz, w, h in chargers)

        h_func = self._charger_heuristic

        # Primary: full NPC danger avoidance
        cost_map = self._build_cost_map(prep, npc_weight=1.0)
        act = self._weighted_astar(prep, cost_map, is_goal, h_func)
        if act is not None:
            return act

        # Fallback 1: reduced NPC danger
        cost_map = self._build_cost_map(prep, npc_weight=0.3)
        act = self._weighted_astar(prep, cost_map, is_goal, h_func)
        if act is not None:
            return act

        # Fallback 2: no NPC avoidance at all
        cost_map = self._build_cost_map(prep, npc_weight=0.0)
        return self._weighted_astar(prep, cost_map, is_goal, h_func)
