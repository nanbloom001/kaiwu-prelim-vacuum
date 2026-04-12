"""Expert policy: NPC safety filter + weighted A* charger navigation with hysteresis.

Three layers:
  - Layer 1: NPC collision safety (always active, blocks dangerous NPC directions)
  - Layer 2: Battery return with hysteresis state machine + A* actual path distance
  - Layer 3: Blocked cell memory + path caching for efficient replanning

Charging state machine:
  - return_mode flag persists once triggered until battery >= 95% AND on charger
  - Uses A* actual path distance (not Chebyshev) for accurate threshold
  - Dynamic margin based on path complexity
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

    # Charging state machine parameters
    EXIT_RETURN_RATIO = 0.95       # Leave return_mode when battery >= 95% and on charger
    LOW_BATTERY_RATIO = 0.26       # Force return_mode when battery < 26%
    BASE_RETURN_MARGIN = 14.0      # Base safety margin for return threshold

    # Blocked cell memory
    BLOCKED_TTL = 8                # Steps before blocked cell expires

    # Cost-map parameters
    _INF_COST = 1e6
    _NPC_DANGER_MAX = 15.0         # peak danger cost at NPC position
    _NPC_DANGER_DECAY = 2.0        # exponential decay rate
    _NPC_DANGER_RADIUS = 8         # Chebyshev radius of danger zone
    _UNEXPLORED_COST = 3.0         # moderate cost for unexplored (passable but uncertain)

    def __init__(self):
        self._charger_list = []
        # Charging state machine
        self.return_mode = False
        # Path caching
        self._cached_path = []          # List[(x, z)]
        self._cached_distance = float('inf')
        self._cached_target = None
        # Blocked cell memory
        self.blocked_cells = {}         # Dict[(x, z), int] TTL countdown
        self._prev_pos = None

    def reset(self):
        """Reset per-episode state."""
        self.return_mode = False
        self._cached_path = []
        self._cached_distance = float('inf')
        self._cached_target = None
        self.blocked_cells = {}
        self._prev_pos = None

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

    def _is_on_charger(self, x, z):
        """Check if position is within any charger region."""
        return any(abs(x - cx) <= w and abs(z - cz) <= h
                   for cx, cz, w, h in self._charger_list)

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
    # Blocked cell memory
    # ------------------------------------------------------------------

    def update_blocked(self, prep, last_action):
        """Track cells that blocked movement (action executed but position unchanged)."""
        cur = prep.cur_pos

        # Decay TTL
        expired = [k for k, v in self.blocked_cells.items() if v <= 1]
        for k in expired:
            del self.blocked_cells[k]
        for k in list(self.blocked_cells):
            self.blocked_cells[k] -= 1

        # Detect blocked movement
        if self._prev_pos is not None and cur == self._prev_pos and last_action is not None and last_action >= 0:
            dx, dz = self.DELTAS[last_action]
            tx, tz = cur[0] + dx, cur[1] + dz
            if 0 <= tx < self.GRID and 0 <= tz < self.GRID:
                self.blocked_cells[(tx, tz)] = self.BLOCKED_TTL

        self._prev_pos = cur

    # ------------------------------------------------------------------
    # Layer 2: Battery emergency override with hysteresis
    # ------------------------------------------------------------------

    def get_override(self, prep, legal_action, last_action=-1):
        """Override for battery emergency with A* path distance + hysteresis.

        Returns (should_override, expert_action).
        Once return_mode is activated, it persists until battery >= 95% AND on charger.
        """
        self.update_chargers(prep)
        self.update_blocked(prep, last_action)

        hx, hz = prep.cur_pos
        battery_ratio = prep.battery / max(prep.battery_max, 1.0)
        on_charger = self._is_on_charger(hx, hz)

        # Exit return_mode: fully charged and on charger
        if self.return_mode and on_charger and battery_ratio >= self.EXIT_RETURN_RATIO:
            self.return_mode = False
            self._cached_path = []
            self._cached_target = None

        # Try to get A* path to nearest charger (with caching)
        charger_path, charger_dist, charger_target = self._plan_to_charger_cached(prep)

        # If no A* path found, fall back to Chebyshev distance
        if not charger_path:
            charger_dist = prep.nearest_charger_dist

        # Compute dynamic margin based on path complexity
        margin = self._charge_margin(charger_path)

        # Trigger conditions
        should_return = (
            self.return_mode
            or (charger_dist < float('inf') and prep.battery <= charger_dist + margin)
            or battery_ratio <= self.LOW_BATTERY_RATIO
        )

        if should_return:
            self.return_mode = True

            # Already on charger — don't override, charging happens automatically
            if on_charger and battery_ratio < self.EXIT_RETURN_RATIO:
                return False, None

            # If we have a cached path, use it
            if charger_path and len(charger_path) >= 2:
                act = self._path_to_action(hx, hz, charger_path[1])
                if act is not None and legal_action[act]:
                    return True, act

            # Fall back to A* planner (3-level NPC avoidance)
            act = self._plan_to_charger(prep)
            if act is not None and legal_action[act]:
                return True, act

            # Last resort: move toward charger greedily
            if prep.nearest_charger_dx != 0 or prep.nearest_charger_dz != 0:
                best_act = self._greedy_toward_charger(prep, legal_action)
                if best_act is not None:
                    return True, best_act

        return False, None

    def _greedy_toward_charger(self, prep, legal_action):
        """Fallback: pick legal action that moves closest to nearest charger."""
        hx, hz = prep.cur_pos
        best_act = None
        best_dist = float('inf')
        for idx, (dx, dz) in enumerate(self.DELTAS):
            if not legal_action[idx]:
                continue
            nx, nz = hx + dx, hz + dz
            dist = self._charger_heuristic(nx, nz)
            if dist < best_dist:
                best_dist = dist
                best_act = idx
        return best_act

    def _charge_margin(self, path):
        """Dynamic safety margin based on path complexity."""
        if len(path) < 2:
            return 20.0
        turns = 0
        blocked_count = 0
        prev_delta = None
        for i in range(1, len(path)):
            cur = path[i - 1]
            nxt = path[i]
            delta = (int(np.clip(nxt[0] - cur[0], -1, 1)),
                     int(np.clip(nxt[1] - cur[1], -1, 1)))
            if prev_delta is not None and delta != prev_delta:
                turns += 1
            prev_delta = delta
            if nxt in self.blocked_cells:
                blocked_count += 1
        margin = self.BASE_RETURN_MARGIN + 0.35 * float(turns) + 1.2 * float(blocked_count)
        return min(margin, 40.0)  # Cap at 40

    # ------------------------------------------------------------------
    # Path caching
    # ------------------------------------------------------------------

    def _plan_to_charger_cached(self, prep):
        """Use cached charger path if still valid, otherwise replan."""
        hx, hz = prep.cur_pos

        if self._cached_path:
            # Try to find current position in cached path
            try:
                idx = self._cached_path.index((hx, hz))
                remaining = self._cached_path[idx:]
                # Check next step is not blocked
                if len(remaining) >= 2:
                    nxt = remaining[1]
                    if nxt not in self.blocked_cells:
                        return remaining, float(len(remaining) - 1), self._cached_target
            except ValueError:
                pass
            # Cache miss — fall through to replan

        # Plan fresh path using A*
        chargers = self._charger_list
        if not chargers:
            return [], float('inf'), None

        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h
                       for cx, cz, w, h in chargers)

        h_func = self._charger_heuristic

        # Try with full NPC danger
        cost_map = self._build_cost_map(prep, npc_weight=1.0)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = path[-1] if path else None
            return path, dist, self._cached_target

        # Fallback: reduced NPC danger
        cost_map = self._build_cost_map(prep, npc_weight=0.3)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = path[-1] if path else None
            return path, dist, self._cached_target

        # Fallback: no NPC avoidance
        cost_map = self._build_cost_map(prep, npc_weight=0.0)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = path[-1] if path else None
            return path, dist, self._cached_target

        return [], float('inf'), None

    @staticmethod
    def _path_to_action(hx, hz, next_pos):
        """Convert a path step to action index."""
        dx = int(np.clip(next_pos[0] - hx, -1, 1))
        dz = int(np.clip(next_pos[1] - hz, -1, 1))
        for idx, (adx, adz) in enumerate(ExpertPolicy.DELTAS):
            if adx == dx and adz == dz:
                return idx
        return None

    # ------------------------------------------------------------------
    # Cost-map construction (with visit_count + blocked penalties)
    # ------------------------------------------------------------------

    def _build_cost_map(self, prep, npc_weight=1.0):
        """Build 128x128 weighted cost map.

        Cell costs:
          - 1.0 + visit_penalty    : explored + passable (visit penalty encourages unexplored paths)
          - _UNEXPLORED_COST       : unexplored (passable but uncertain)
          - _INF_COST              : explored + impassable (known wall)
          - + blocked_penalty      : cells that previously blocked movement
          - + danger               : passable cell near NPC (danger decays exponentially)
        """
        G = self.GRID
        cost = np.full((G, G), self._UNEXPLORED_COST, dtype=np.float32)

        explored = prep.explored_map >= 0.5
        base_passable = explored & (prep.passable_map >= 0.5)
        cost[base_passable] = 1.0

        # Visit count penalty: encourage taking less-visited paths
        visit_penalty = np.clip(prep.visit_count * 0.15, 0, 0.75)
        cost[base_passable] += visit_penalty[base_passable]

        # Explored + impassable
        cost[explored & (prep.passable_map < 0.5)] = self._INF_COST

        # Blocked cell penalty
        for (bx, bz), ttl in self.blocked_cells.items():
            if 0 <= bx < G and 0 <= bz < G and cost[bx, bz] < self._INF_COST:
                cost[bx, bz] += 4.0

        # NPC danger
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
    # Weighted A* search — returns full path for caching
    # ------------------------------------------------------------------

    def _weighted_astar_full(self, prep, cost_map, is_goal, h_func):
        """A* on weighted cost map. Returns (first_action, path, distance)."""
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return 0, [(sx, sz)], 0.0

        G = self.GRID
        INF = 1e9
        dist_arr = np.full((G, G), INF, dtype=np.float32)
        dist_arr[sx, sz] = 0.0
        first_act = np.full((G, G), -1, dtype=np.int8)
        parent = {}  # Dict[(x,z), (x,z)] for path reconstruction
        closed = np.zeros((G, G), dtype=np.bool_)

        counter = 0
        heap = [(h_func(sx, sz), counter, sx, sz)]

        while heap:
            f, _, x, z = heapq.heappop(heap)
            if closed[x, z]:
                continue
            closed[x, z] = True
            if is_goal(x, z):
                # Reconstruct path
                path = self._reconstruct_path(parent, (sx, sz), (x, z))
                fa = int(first_act[x, z])
                return fa if fa >= 0 else 0, path, float(dist_arr[x, z])

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
                    parent[(nx, nz)] = (x, z)
                    counter += 1
                    heapq.heappush(heap, (new_d + h_func(nx, nz), counter, nx, nz))

        return None, [], float('inf')

    @staticmethod
    def _reconstruct_path(parent, start, goal):
        """Trace back from goal to start using parent dict."""
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Original A* (returns first_action only — used by fallback)
    # ------------------------------------------------------------------

    def _weighted_astar(self, prep, cost_map, is_goal, h_func):
        """A* on weighted cost map. Returns the first-action index to reach goal."""
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
    # Charger heuristic
    # ------------------------------------------------------------------

    def _charger_heuristic(self, x, z):
        """Chebyshev distance to nearest charger boundary (admissible)."""
        if not self._charger_list:
            return 0.0
        return min(
            max(abs(x - cx) - w, abs(z - cz) - h, 0)
            for cx, cz, w, h in self._charger_list
        )

    # ------------------------------------------------------------------
    # Charger path planning (fallback, returns first_action only)
    # ------------------------------------------------------------------

    def _plan_to_charger(self, prep):
        """Plan path to nearest charger with graduated fallback.

        Returns first action index only (for non-cached fallback).
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
