"""
Mini Food Delivery Assistant  —  AI Fundamentals Final Project (team self1sh)

A courier must deliver food across a small 6x6 neighbourhood.
Three course algorithms work as ONE pipeline:

  1. A* search          -> plans the fastest route (restaurant -> customer)
  2. Bayesian inference -> a traffic alert fires on a road ON that route;
                           update P(road is blocked)
  3. MDP (Bellman)      -> using that probability, choose Continue vs Reroute
     (+ A* again)       -> if Reroute, A* replans with the suspect road avoided

All numbers are hypothetical toy values (allowed by the guidelines).
Console:  python delivery_assistant.py            (alert received)
          python delivery_assistant.py --no-alert (sensor stayed silent)
          python delivery_assistant.py --prior 0.05
Web UI:   python app.py
"""

import argparse
import heapq

# ---------------------------------------------------------------- 1. MAP
# S = restaurant, C = customer, '.' = main road (1 min per cell),
# '~' = back alley (3 min per cell), '#' = building (impassable)
CITY_MAP = [
    "S...#.",
    "~~~.~~",
    "~#~.~~",
    "~~~...",
    "~~#~~.",
    "~~~~~C",
]
CELL_COST = {".": 1, "S": 1, "C": 1, "~": 3}


def parse_map(rows):
    start = goal = None
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "C":
                goal = (r, c)
    return start, goal


def manhattan(a, b):
    # admissible: every move costs >= 1 minute
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star(rows, start, goal, avoid=frozenset(), verbose=False, trace=None):
    """A* on a weighted grid. Returns (path, total_cost, expanded_count).
    If `trace` is a list, every expansion (cell, g, h, f) is appended to it."""
    n_rows, n_cols = len(rows), len(rows[0])
    open_heap = [(manhattan(start, goal), 0, start)]   # (f, g, cell)
    came_from = {start: None}
    g_score = {start: 0}
    closed = set()
    step = 0

    while open_heap:
        f, g, cell = heapq.heappop(open_heap)
        if cell in closed:
            continue
        closed.add(cell)
        step += 1
        if verbose:
            print(f"   expand #{step:<2} {cell}  g={g:<2} h={f - g:<2} f={f}")
        if trace is not None:
            trace.append((cell, g, f - g, f))
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1], g, step

        r, c = cell
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            ch = rows[nr][nc]
            if ch == "#" or (nr, nc) in avoid:
                continue
            new_g = g + CELL_COST[ch]
            if new_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = new_g
                came_from[(nr, nc)] = cell
                heapq.heappush(open_heap, (new_g + manhattan((nr, nc), goal), new_g, (nr, nc)))
    return None, float("inf"), step


def draw(rows, path, alert_cell=None, avoid=frozenset()):
    on_path = set(path or [])
    for r, row in enumerate(rows):
        line = []
        for c, ch in enumerate(row):
            if (r, c) == alert_cell:
                line.append("X" if (r, c) in avoid else "!")
            elif ch in "SC":
                line.append(ch)
            elif (r, c) in on_path:
                line.append("*")
            else:
                line.append(ch)
        print("   " + " ".join(line))


# ---------------------------------------------------------- 2. BAYES
# Hypothesis B = "road cell is blocked".  Evidence A = "traffic alert".
P_BLOCKED_PRIOR = 0.20       # 1 in 5 rush-hour roads gets jammed
P_ALERT_IF_BLOCKED = 0.90    # sensor sensitivity
P_ALERT_IF_CLEAR = 0.15      # false-alarm rate


def bayes_update(prior, alert, p_a_b=P_ALERT_IF_BLOCKED, p_a_nb=P_ALERT_IF_CLEAR):
    """Returns posterior P(Blocked | observation) and the pieces for display."""
    like_b = p_a_b if alert else 1 - p_a_b
    like_nb = p_a_nb if alert else 1 - p_a_nb
    num = like_b * prior
    evidence = num + like_nb * (1 - prior)
    return num / evidence, like_b, like_nb, evidence


# ------------------------------------------------------------ 3. MDP
# States: Clear, Blocked (status of the suspect road)
# Actions: Continue (keep A* route), Reroute (A* detour avoiding it)
# Reward = -(minutes to customer). Minutes come straight from A* costs.
JAM_WAIT = 15     # minutes lost if you drive into a blocked road and wait
GAMMA = 0.9

# Transition model T[s][a] = {s': prob}: road state for the NEXT delivery
# (2-step lookahead: this delivery + the next one, gamma-discounted).
# Rerouting lets traffic clear; continuing into a jam keeps it jammed.
T = {
    "Clear":   {"Continue": {"Clear": 0.9, "Blocked": 0.1},
                "Reroute":  {"Clear": 0.9, "Blocked": 0.1}},
    "Blocked": {"Continue": {"Clear": 0.3, "Blocked": 0.7},
                "Reroute":  {"Clear": 0.6, "Blocked": 0.4}},
}


def build_rewards(main_cost, detour_cost, jam_wait=JAM_WAIT):
    return {
        "Clear":   {"Continue": -main_cost,              "Reroute": -detour_cost},
        "Blocked": {"Continue": -(main_cost + jam_wait), "Reroute": -detour_cost},
    }


def bellman_q(R, V, gamma=GAMMA):
    """One Bellman update: Q(s,a) = R(s,a) + gamma * sum_s' T(s'|s,a) V(s')."""
    return {s: {a: R[s][a] + gamma * sum(p * V[s2] for s2, p in T[s][a].items())
                for a in R[s]} for s in R}


def initial_values(R):
    """V0(s) = best immediate reward in s (value of the NEXT delivery alone)."""
    return {s: max(R[s].values()) for s in R}


def break_even(R, V, gamma=GAMMA):
    """P(Blocked) at which Continue and Reroute have equal expected value."""
    Q = bellman_q(R, V, gamma)
    dc = Q["Clear"]["Continue"] - Q["Clear"]["Reroute"]       # gain of Continue if clear
    db = Q["Blocked"]["Reroute"] - Q["Blocked"]["Continue"]   # gain of Reroute if blocked
    if dc <= 0:
        return 0.0          # rerouting is never worse
    if db <= 0:
        return 1.0          # continuing is never worse
    return dc / (dc + db)


def decide(R, p_blocked, V, gamma=GAMMA):
    """Belief-weighted action values (the courier can't SEE the road)."""
    Q = bellman_q(R, V, gamma)
    belief = {"Clear": 1 - p_blocked, "Blocked": p_blocked}
    qb = {a: sum(belief[s] * Q[s][a] for s in Q) for a in ("Continue", "Reroute")}
    return max(qb, key=qb.get), qb, Q


# ------------------------------------------------------------ PIPELINE
def solve(rows=CITY_MAP, alert=True, prior=P_BLOCKED_PRIOR, alert_cell=None,
          p_a_b=P_ALERT_IF_BLOCKED, p_a_nb=P_ALERT_IF_CLEAR,
          jam_wait=JAM_WAIT, gamma=GAMMA):
    """Runs the whole A* -> Bayes -> MDP -> A* pipeline and returns every
    intermediate value (used by the console demo, the web UI and the tests)."""
    start, goal = parse_map(rows)
    if start is None or goal is None:
        return {"error": "Map needs exactly one S (restaurant) and one C (customer)."}

    trace = []
    path, cost, expanded = a_star(rows, start, goal, trace=trace)
    if path is None:
        return {"error": "No route: the customer is walled off.", "trace": trace}

    inner = path[1:-1]
    if not inner:
        return {"error": "Restaurant and customer are neighbours - nothing to decide."}
    if alert_cell is None or alert_cell not in inner:
        # default: a road cell around the middle of the route (never S or C)
        alert_cell = inner[min(max(len(path) // 2 - 2, 0), len(inner) - 1)]

    post, lb, lnb, ev = bayes_update(prior, alert, p_a_b, p_a_nb)

    d_path, d_cost, _ = a_star(rows, start, goal, avoid=frozenset({alert_cell}))
    if d_path is None:
        # no detour exists: the only option is to keep the route
        return {"start": start, "goal": goal, "path": path, "cost": cost,
                "expanded": expanded, "trace": trace, "alert_cell": alert_cell,
                "alert": alert, "prior": prior, "posterior": post,
                "like_b": lb, "like_nb": lnb, "evidence": ev,
                "detour_path": None, "detour_cost": None,
                "action": "Continue", "final_path": path,
                "note": "No detour exists around this cell, so Continue is forced."}

    R = build_rewards(cost, d_cost, jam_wait)
    V = initial_values(R)
    action, qb, Q = decide(R, post, V, gamma)
    return {"start": start, "goal": goal, "path": path, "cost": cost,
            "expanded": expanded, "trace": trace, "alert_cell": alert_cell,
            "alert": alert, "prior": prior, "posterior": post,
            "like_b": lb, "like_nb": lnb, "evidence": ev,
            "detour_path": d_path, "detour_cost": d_cost,
            "R": R, "V0": V, "Q": Q, "qb": qb, "gamma": gamma,
            "break_even": break_even(R, V, gamma),
            "action": action,
            "final_path": d_path if action == "Reroute" else path}


def run(alert=True, prior=P_BLOCKED_PRIOR, quiet=False):
    say = (lambda *a, **k: None) if quiet else print
    res = solve(CITY_MAP, alert, prior)
    start, goal, path, cost = res["start"], res["goal"], res["path"], res["cost"]

    say("=" * 60)
    say(" STEP 1 - A* route planning")
    say("=" * 60)
    say(f" Input: 6x6 map, restaurant S={start}, customer C={goal}")
    say("        '.'=1 min, '~'=3 min, '#'=building, h = Manhattan distance")
    for i, (cell, g, h, f) in enumerate(res["trace"], 1):
        say(f"   expand #{i:<2} {cell}  g={g:<2} h={h:<2} f={f}")
    say(f"\n Output: route cost {cost} min, {len(path) - 1} moves, {res['expanded']} nodes expanded")
    say(f" Route: {' -> '.join(map(str, path))}")
    if not quiet:
        draw(CITY_MAP, path)

    cell, post = res["alert_cell"], res["posterior"]
    lb, lnb, ev = res["like_b"], res["like_nb"], res["evidence"]
    say("\n" + "=" * 60)
    say(" STEP 2 - Bayesian update on the route")
    say("=" * 60)
    say(f" Suspect road cell on our route: {cell}")
    say(f" Observation: {'TRAFFIC ALERT' if alert else 'no alert'}")
    say(f" P(B)={prior:.2f}  P(obs|B)={lb:.2f}  P(obs|~B)={lnb:.2f}")
    say(f" P(obs) = {lb:.2f}*{prior:.2f} + {lnb:.2f}*{1 - prior:.2f} = {ev:.3f}")
    say(f" Output: P(Blocked | obs) = {lb:.2f}*{prior:.2f} / {ev:.3f} = {post:.3f}")

    say("\n" + "=" * 60)
    say(" STEP 3 - MDP decision: Continue vs Reroute")
    say("=" * 60)
    R, V, Q, qb = res["R"], res["V0"], res["Q"], res["qb"]
    say(f" A* detour (avoiding {cell}): {res['detour_cost']} min")
    say(f" Rewards (=-minutes): {R}")
    say(f" V0 = best immediate reward: V0(Clear)={V['Clear']}, V0(Blocked)={V['Blocked']}")
    say(f" One Bellman update: Q(s,a) = R(s,a) + {GAMMA}*sum T(s'|s,a)*V0(s')")
    for s in Q:
        say(f"   Q({s:7}, Continue)={Q[s]['Continue']:7.2f}   Q({s:7}, Reroute)={Q[s]['Reroute']:7.2f}")
    for a in qb:
        say(f"   E[Q({a:8})] = {1 - post:.3f}*{Q['Clear'][a]:.2f} + {post:.3f}*{Q['Blocked'][a]:.2f} = {qb[a]:.2f}")
    say(f" Break-even: reroute whenever P(Blocked) > {res['break_even']:.3f}")
    action = res["action"]
    say(f"\n Output: RECOMMENDED ACTION = {action.upper()}")

    say("\n" + "=" * 60)
    say(" FINAL PLAN")
    say("=" * 60)
    if not quiet:
        avoid = frozenset({cell}) if action == "Reroute" else frozenset()
        draw(CITY_MAP, res["final_path"], cell, avoid)
        print("   (! = suspect road on route, X = suspect road avoided)")
    say(f" Route: {' -> '.join(map(str, res['final_path']))}")
    say(f" Blockage probability: {post:.1%}   Action: {action}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mini Food Delivery Assistant")
    ap.add_argument("--no-alert", action="store_true", help="sensor did NOT fire")
    ap.add_argument("--prior", type=float, default=P_BLOCKED_PRIOR, help="prior P(blocked)")
    args = ap.parse_args()
    run(alert=not args.no_alert, prior=args.prior)
