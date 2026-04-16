"""LTSPPO expert utilities: NPC safety filter + charger planning signals + emergency fallback.

This module intentionally no longer provides regular charging policy guidance.

What remains:
  - NPC collision safety filter for legal-action masking
  - Weighted A* charger planning as a signal / fallback utility
  - Blocked-cell memory to stabilize replanning
  - Explicit extreme-emergency fallback path for agent runtime use
  - Helper methods for teacher / target reliability checks

What is removed from policy guidance:
  - No regular charging logit bias
  - No regular return-mode hysteresis controlling normal policy behavior

`return_mode` is kept only as a compatibility flag for "extreme emergency fallback
currently active", so existing runtime code can still suppress anti-stuck logic.
"""
from __future__ import annotations

import heapq
import numpy as np

from agent_ppo.conf.conf import Config


class ExpertPolicy:
    GRID = 128
    DELTAS = (
        (1, 0), (1, -1), (0, -1), (-1, -1),
        (-1, 0), (-1, 1), (0, 1), (1, 1),
    )

    # Compatibility / fallback control
    EXIT_EMERGENCY_RATIO = 0.12
    EMERGENCY_RATIO = float(getattr(Config, "EXPERT_EMERGENCY_BATTERY_RATIO", 0.05))
    EMERGENCY_SLACK_MARGIN = 0.0
    EMERGENCY_PATH_MARGIN = 2.0

    # Reliability thresholds for teacher signals
    RELIABLE_NPC_DIST = 4
    RELIABLE_SLACK_BUFFER = 6.0
    RELIABLE_RETURN_RATIO = 0.35
    RELIABLE_PREPARE_RETURN_RATIO = 0.50

    # Blocked cell memory
    BLOCKED_TTL = 8

    # Cost-map parameters
    _INF_COST = 1e6
    _NPC_DANGER_MAX = 15.0
    _NPC_DANGER_DECAY = 2.0
    _NPC_DANGER_RADIUS = 8
    _UNEXPLORED_COST = 1.8

    def __init__(self):
        self._charger_list = []
        self.return_mode = False
        self._cached_path = []
        self._cached_distance = float("inf")
        self._cached_target = None
        self.blocked_cells = {}
        self._prev_pos = None
        self._last_emergency_reason = None

    def reset(self):
        """Reset per-episode state."""
        self._charger_list = []
        self.return_mode = False
        self._cached_path = []
        self._cached_distance = float("inf")
        self._cached_target = None
        self.blocked_cells = {}
        self._prev_pos = None
        self._last_emergency_reason = None

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
        return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in self._charger_list)

    # ------------------------------------------------------------------
    # Layer 1: NPC safety filter
    # ------------------------------------------------------------------

    def filter_actions(self, prep, legal_action):
        """Block stepping onto NPC and moving toward nearby NPCs."""
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
                if nx2 == nx and nz2 == nz:
                    legal[idx] = 0
                    continue
                if npc_dist <= 3:
                    new_dist = max(abs(nx2 - nx), abs(nz2 - nz))
                    if new_dist < npc_dist:
                        legal[idx] = 0
                elif npc_dist <= 5:
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

        expired = [k for k, v in self.blocked_cells.items() if v <= 1]
        for k in expired:
            del self.blocked_cells[k]
        for k in list(self.blocked_cells):
            self.blocked_cells[k] -= 1

        if self._prev_pos is not None and cur == self._prev_pos and last_action is not None and last_action >= 0:
            dx, dz = self.DELTAS[last_action]
            tx, tz = cur[0] + dx, cur[1] + dz
            if 0 <= tx < self.GRID and 0 <= tz < self.GRID:
                self.blocked_cells[(tx, tz)] = self.BLOCKED_TTL

        self._prev_pos = cur

    # ------------------------------------------------------------------
    # Public planning / reliability helpers
    # ------------------------------------------------------------------

    def get_charger_signal(self, prep, legal_action=None, last_action=-1, refresh_state=False):
        """Return structured charger-planning information without controlling policy."""
        self.update_chargers(prep)
        if refresh_state:
            self.update_blocked(prep, last_action)

        hx, hz = prep.cur_pos
        battery = float(getattr(prep, "battery", 0.0))
        battery_max = max(float(getattr(prep, "battery_max", 1.0)), 1.0)
        battery_ratio = battery / battery_max
        on_charger = self._is_on_charger(hx, hz)

        charger_path, charger_dist, charger_target = self._plan_to_charger_cached(prep)
        if not charger_path:
            charger_dist = float(getattr(prep, "nearest_charger_dist", float("inf")))
            charger_target = charger_target or self._nearest_charger_center(hx, hz)

        margin = self._charge_margin(charger_path)
        slack = self._estimate_slack(prep, charger_dist)
        min_npc_dist = self._min_npc_dist(prep)
        suggested_action = None

        if charger_path and len(charger_path) >= 2:
            suggested_action = self._path_to_action(hx, hz, charger_path[1])

        if suggested_action is None:
            suggested_action = self._plan_to_charger(prep)

        if suggested_action is None and legal_action is not None:
            suggested_action = self._greedy_toward_charger(prep, legal_action)

        legal_and_safe = False
        if legal_action is not None and suggested_action is not None:
            legal_and_safe = bool(legal_action[suggested_action])

        reachable = bool(charger_path) or np.isfinite(charger_dist)
        target_reliable = self._is_target_signal_reliable(
            prep,
            legal_and_safe=legal_and_safe,
            reachable=reachable,
            charger_target=charger_target,
            charger_dist=charger_dist,
            slack=slack,
            min_npc_dist=min_npc_dist,
        )
        mode_reliable = self._is_mode_signal_reliable(
            battery_ratio=battery_ratio,
            slack=slack,
            min_npc_dist=min_npc_dist,
            target_reliable=target_reliable,
            on_charger=on_charger,
        )

        return {
            "battery_ratio": battery_ratio,
            "battery": battery,
            "on_charger": on_charger,
            "charger_dist": float(charger_dist),
            "charger_target": charger_target,
            "charger_path": charger_path,
            "margin": float(margin),
            "slack": float(slack),
            "min_npc_dist": float(min_npc_dist),
            "suggested_action": suggested_action,
            "suggested_action_legal": legal_and_safe,
            "reachable": reachable,
            "target_reliable": bool(target_reliable),
            "mode_reliable": bool(mode_reliable),
        }

    def is_target_teacher_reliable(self, prep, legal_action=None, last_action=-1):
        signal = self.get_charger_signal(prep, legal_action, last_action)
        return bool(signal["target_reliable"])

    def is_mode_teacher_reliable(self, prep, legal_action=None, last_action=-1):
        signal = self.get_charger_signal(prep, legal_action, last_action)
        return bool(signal["mode_reliable"])

    def get_teacher_guidance(self, prep, legal_action=None, last_action=-1):
        """Return optional teacher guidance payload for future mode/target supervision."""
        signal = self.get_charger_signal(prep, legal_action, last_action)
        if not signal["target_reliable"] and not signal["mode_reliable"]:
            return None

        battery_ratio = signal["battery_ratio"]
        slack = signal["slack"]
        on_charger = signal["on_charger"]

        if signal["min_npc_dist"] <= 2:
            mode = "evade"
        elif on_charger and battery_ratio < self.RELIABLE_RETURN_RATIO:
            mode = "return"
        elif battery_ratio <= self.RELIABLE_RETURN_RATIO or slack <= 0:
            mode = "return"
        elif battery_ratio <= self.RELIABLE_PREPARE_RETURN_RATIO or slack <= self.RELIABLE_SLACK_BUFFER:
            mode = "prepare_return"
        else:
            mode = "clean"

        return {
            "mode": mode,
            "target": signal["charger_target"],
            "action": signal["suggested_action"],
            "teacher_mask": float(signal["target_reliable"] or signal["mode_reliable"]),
            "signal": signal,
        }

    # ------------------------------------------------------------------
    # Extreme emergency fallback
    # ------------------------------------------------------------------

    def get_emergency_fallback(self, prep, legal_action, last_action=-1):
        """Return explicit charger fallback only in extreme emergency.

        This is runtime-oriented and intentionally much narrower than the old
        return-mode controller. It only activates when the robot is close to
        battery death and charger reachability is already critical.
        """
        legal_action = list(legal_action)
        signal = self.get_charger_signal(prep, legal_action, last_action, refresh_state=True)

        battery_ratio = signal["battery_ratio"]
        on_charger = signal["on_charger"]
        slack = signal["slack"]
        charger_dist = signal["charger_dist"]
        margin = signal["margin"]
        suggested_action = signal["suggested_action"]

        if self.return_mode:
            should_keep = (
                not on_charger
                and battery_ratio < self.EXIT_EMERGENCY_RATIO
                and (
                    slack <= self.RELIABLE_SLACK_BUFFER
                    or battery_ratio <= self.EMERGENCY_RATIO * 1.5
                )
            )
            if not should_keep:
                self.return_mode = False
                self._last_emergency_reason = None

        should_trigger = (
            not on_charger
            and suggested_action is not None
            and legal_action[suggested_action]
            and (
                (battery_ratio <= self.EMERGENCY_RATIO and slack <= self.EMERGENCY_SLACK_MARGIN)
                or (battery_ratio <= self.EMERGENCY_RATIO * 0.8)
                or (np.isfinite(charger_dist) and prep.battery <= charger_dist + self.EMERGENCY_PATH_MARGIN)
            )
        )

        if should_trigger:
            self.return_mode = True
            if battery_ratio <= self.EMERGENCY_RATIO and slack <= self.EMERGENCY_SLACK_MARGIN:
                reason = "battery_and_slack_critical"
            elif battery_ratio <= self.EMERGENCY_RATIO * 0.8:
                reason = "battery_ratio_critical"
            else:
                reason = "path_margin_critical"
            self._last_emergency_reason = reason
            return {
                "active": True,
                "action": suggested_action,
                "reason": reason,
                "signal": signal,
            }

        return {
            "active": False,
            "action": None,
            "reason": self._last_emergency_reason,
            "signal": signal,
        }

    # ------------------------------------------------------------------
    # Legacy-compatible interfaces used by runtime
    # ------------------------------------------------------------------

    def get_override(self, prep, legal_action, last_action=-1):
        """Legacy compatibility: only returns extreme-emergency fallback now."""
        fallback = self.get_emergency_fallback(prep, legal_action, last_action)
        if fallback["active"] and fallback["action"] is not None:
            return True, fallback["action"]
        return False, None

    def get_logit_bias(self, prep, legal_action, last_action=-1):
        """Return only NPC avoidance bias; no regular charging / return bias."""
        # Keep compatibility flag aligned with emergency state only.
        _ = self.get_emergency_fallback(prep, legal_action, last_action)

        bias = np.zeros(8, dtype=np.float32)
        hx, hz = prep.cur_pos

        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            ndx, ndz = nx - hx, nz - hz
            npc_dist = max(abs(ndx), abs(ndz))
            if npc_dist > 6 or npc_dist < 1:
                continue
            for idx, (dx, dz) in enumerate(self.DELTAS):
                if not legal_action[idx]:
                    continue
                nlen = max(max(abs(ndx), abs(ndz)), 1.0)
                dot = (dx * ndx + dz * ndz) / nlen
                if dot > 0:
                    close = (6.0 - npc_dist) / 6.0
                    bias[idx] -= 2.0 * dot * close

        return bias

    # ------------------------------------------------------------------
    # Planner helpers
    # ------------------------------------------------------------------

    def _greedy_toward_charger(self, prep, legal_action):
        hx, hz = prep.cur_pos
        best_act = None
        best_dist = float("inf")
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
            delta = (
                int(np.clip(nxt[0] - cur[0], -1, 1)),
                int(np.clip(nxt[1] - cur[1], -1, 1)),
            )
            if prev_delta is not None and delta != prev_delta:
                turns += 1
            prev_delta = delta
            if nxt in self.blocked_cells:
                blocked_count += 1
        margin = 18.0 + 0.35 * float(turns) + 1.2 * float(blocked_count)
        return min(margin, 40.0)

    # ------------------------------------------------------------------
    # Path caching
    # ------------------------------------------------------------------

    def _plan_to_charger_cached(self, prep):
        """Use cached charger path if still valid, otherwise replan."""
        hx, hz = prep.cur_pos

        if self._cached_path:
            try:
                idx = self._cached_path.index((hx, hz))
                remaining = self._cached_path[idx:]
                if len(remaining) >= 2:
                    nxt = remaining[1]
                    if nxt not in self.blocked_cells:
                        return remaining, float(len(remaining) - 1), self._cached_target
            except ValueError:
                pass

        chargers = self._charger_list
        if not chargers:
            return [], float("inf"), None

        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in chargers)

        h_func = self._charger_heuristic

        cost_map = self._build_cost_map(prep, npc_weight=1.0)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = self._match_target_to_charger(path[-1])
            return path, dist, self._cached_target

        cost_map = self._build_cost_map(prep, npc_weight=0.3)
        act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
        if act is not None and path:
            self._cached_path = path
            self._cached_distance = dist
            self._cached_target = self._match_target_to_charger(path[-1])
            return path, dist, self._cached_target

        return [], float("inf"), None

    @staticmethod
    def _path_to_action(hx, hz, next_pos):
        dx = int(np.clip(next_pos[0] - hx, -1, 1))
        dz = int(np.clip(next_pos[1] - hz, -1, 1))
        for idx, (adx, adz) in enumerate(ExpertPolicy.DELTAS):
            if adx == dx and adz == dz:
                return idx
        return None

    # ------------------------------------------------------------------
    # Cost-map construction
    # ------------------------------------------------------------------

    def _build_cost_map(self, prep, npc_weight=1.0):
        G = self.GRID
        cost = np.full((G, G), self._UNEXPLORED_COST, dtype=np.float32)

        explored = prep.explored_map >= 0.5
        base_passable = explored & (prep.passable_map >= 0.5)
        cost[base_passable] = 1.0

        visit_penalty = np.clip(prep.visit_count * 0.15, 0, 0.75)
        cost[base_passable] += visit_penalty[base_passable]

        cost[explored & (prep.passable_map < 0.5)] = self._INF_COST

        for (bx, bz), ttl in self.blocked_cells.items():
            if 0 <= bx < G and 0 <= bz < G and cost[bx, bz] < self._INF_COST:
                cost[bx, bz] += 4.0

        if npc_weight > 0:
            for npc in prep._npcs:
                pos = npc.get("pos") or {}
                nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
                radius = self._NPC_DANGER_RADIUS
                x0, x1 = max(nx - radius, 0), min(nx + radius + 1, G)
                z0, z1 = max(nz - radius, 0), min(nz + radius + 1, G)

                lx = np.arange(x0, x1, dtype=np.float32)
                lz = np.arange(z0, z1, dtype=np.float32)
                xx, zz = np.meshgrid(lx, lz, indexing="ij")
                dist = np.maximum(np.abs(xx - nx), np.abs(zz - nz))
                danger = npc_weight * self._NPC_DANGER_MAX * np.exp(-dist / self._NPC_DANGER_DECAY)
                danger[dist > radius] = 0.0

                region = cost[x0:x1, z0:z1]
                safe = region < self._INF_COST
                region[safe] += danger[safe]

        return cost

    # ------------------------------------------------------------------
    # Weighted A* search
    # ------------------------------------------------------------------

    def _weighted_astar_full(self, prep, cost_map, is_goal, h_func):
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return 0, [(sx, sz)], 0.0

        G = self.GRID
        inf = 1e9
        dist_arr = np.full((G, G), inf, dtype=np.float32)
        dist_arr[sx, sz] = 0.0
        first_act = np.full((G, G), -1, dtype=np.int8)
        parent = {}
        closed = np.zeros((G, G), dtype=np.bool_)

        counter = 0
        heap = [(h_func(sx, sz), counter, sx, sz)]

        while heap:
            _, _, x, z = heapq.heappop(heap)
            if closed[x, z]:
                continue
            closed[x, z] = True
            if is_goal(x, z):
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

        return None, [], float("inf")

    @staticmethod
    def _reconstruct_path(parent, start, goal):
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def _weighted_astar(self, prep, cost_map, is_goal, h_func):
        sx, sz = prep.cur_pos
        if is_goal(sx, sz):
            return None

        G = self.GRID
        inf = 1e9
        dist_arr = np.full((G, G), inf, dtype=np.float32)
        dist_arr[sx, sz] = 0.0
        first_act = np.full((G, G), -1, dtype=np.int8)
        closed = np.zeros((G, G), dtype=np.bool_)

        counter = 0
        heap = [(h_func(sx, sz), counter, sx, sz)]

        while heap:
            _, _, x, z = heapq.heappop(heap)
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
    # Charger heuristic / planning
    # ------------------------------------------------------------------

    def _charger_heuristic(self, x, z):
        if not self._charger_list:
            return 0.0
        return min(max(abs(x - cx) - w, abs(z - cz) - h, 0) for cx, cz, w, h in self._charger_list)

    def _plan_to_charger(self, prep):
        chargers = self._charger_list
        if not chargers:
            return None

        def is_goal(x, z):
            return any(abs(x - cx) <= w and abs(z - cz) <= h for cx, cz, w, h in chargers)

        h_func = self._charger_heuristic

        cost_map = self._build_cost_map(prep, npc_weight=1.0)
        act = self._weighted_astar(prep, cost_map, is_goal, h_func)
        if act is not None:
            return act

        cost_map = self._build_cost_map(prep, npc_weight=0.3)
        act = self._weighted_astar(prep, cost_map, is_goal, h_func)
        if act is not None:
            return act

        return None

    # ------------------------------------------------------------------
    # Internal reliability helpers
    # ------------------------------------------------------------------

    def _estimate_slack(self, prep, charger_dist):
        if hasattr(prep, "charger_slack"):
            try:
                return float(prep.charger_slack)
            except (TypeError, ValueError):
                pass
        if not np.isfinite(charger_dist):
            return float("-inf")
        return float(getattr(prep, "battery", 0.0) - charger_dist)

    def _min_npc_dist(self, prep):
        hx, hz = prep.cur_pos
        min_npc_dist = float("inf")
        for npc in prep._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            min_npc_dist = min(min_npc_dist, max(abs(nx - hx), abs(nz - hz)))
        return min_npc_dist

    def _nearest_charger_center(self, hx, hz):
        if not self._charger_list:
            return None
        return min(
            ((cx, cz) for cx, cz, _, _ in self._charger_list),
            key=lambda item: max(abs(item[0] - hx), abs(item[1] - hz)),
        )

    def _match_target_to_charger(self, pos):
        if pos is None:
            return None
        x, z = pos
        for cx, cz, w, h in self._charger_list:
            if abs(x - cx) <= w and abs(z - cz) <= h:
                return (cx, cz)
        return self._nearest_charger_center(x, z)

    def _is_target_signal_reliable(
        self,
        prep,
        legal_and_safe,
        reachable,
        charger_target,
        charger_dist,
        slack,
        min_npc_dist,
    ):
        if not reachable or charger_target is None or not legal_and_safe:
            return False
        if not np.isfinite(charger_dist):
            return False
        if min_npc_dist <= self.RELIABLE_NPC_DIST:
            return False
        if slack <= -self.RELIABLE_SLACK_BUFFER:
            return False
        return True

    def _is_mode_signal_reliable(self, battery_ratio, slack, min_npc_dist, target_reliable, on_charger):
        if min_npc_dist <= 2:
            return True
        if on_charger and battery_ratio < self.RELIABLE_RETURN_RATIO:
            return True
        if battery_ratio <= self.RELIABLE_PREPARE_RETURN_RATIO:
            return target_reliable
        if slack <= self.RELIABLE_SLACK_BUFFER:
            return target_reliable
        return False
