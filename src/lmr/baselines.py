"""External baselines. OR-Tools is the baseline we target to beat (LKH3 is not available here)."""
import numpy as np


def ortools_route(C, seconds=10, scale=10):
    """Open asymmetric path from the depot (node 0) through all stops, minimising sum of C along the path.
    Modelled as a vehicle route that ends at a dummy node with zero-cost arcs.
    Search: PATH_CHEAPEST_ARC first solution + GUIDED_LOCAL_SEARCH for `seconds` (OR-Tools default tuning otherwise).
    Costs are rounded to 1/scale seconds because OR-Tools needs integers."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    n = len(C); M = np.zeros((n + 1, n + 1), dtype=np.int64)
    M[:n, :n] = np.rint(np.asarray(C) * scale).astype(np.int64)
    mgr = pywrapcp.RoutingIndexManager(n + 1, 1, [0], [n])
    routing = pywrapcp.RoutingModel(mgr)
    cb = routing.RegisterTransitCallback(lambda i, j: int(M[mgr.IndexToNode(i), mgr.IndexToNode(j)]))
    routing.SetArcCostEvaluatorOfAllVehicles(cb)
    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    p.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    p.time_limit.seconds = int(seconds)
    sol = routing.SolveWithParameters(p)
    idx, order = routing.Start(0), []
    while not routing.IsEnd(idx):
        order.append(mgr.IndexToNode(idx)); idx = sol.Value(routing.NextVar(idx))
    return order


def _hint_next(path, n):
    """successor map of an open path 0 -> ... -> last, closed through the dummy node n (last -> n -> 0)."""
    nxt = {a: b for a, b in zip(path[:-1], path[1:])}; nxt[path[-1]] = n; nxt[n] = 0
    return nxt


def cpsat_route(C, seconds=10.0, workers=1, hint=None, scale=10):
    """Exact-method baseline: OR-Tools CP-SAT with a circuit constraint (AddCircuit), asymmetric costs, open path from
    the depot via a dummy end node (0 -> dummy forbidden, dummy -> 0 forced). Optionally warm-started with `hint`
    (a route). Returns (route, status, objective, best_bound) with objective/bound in the units of C: bound is a PROVEN
    lower bound on the best possible route cost; status 'OPTIMAL' means the route is proven optimal."""
    from ortools.sat.python import cp_model
    C = np.asarray(C); n = len(C); Ci = np.rint(C * scale).astype(np.int64)
    m = cp_model.CpModel(); arcs = []; lit = {}; terms = []
    for i in range(n):
        for j in range(n):
            if i != j:
                v = m.NewBoolVar(f"x{i}_{j}"); lit[i, j] = v; arcs.append((i, j, v)); terms.append(int(Ci[i, j]) * v)
    for i in range(1, n):
        v = m.NewBoolVar(f"e{i}"); lit[i, n] = v; arcs.append((i, n, v))             # last stop -> dummy, cost 0
    v = m.NewBoolVar("d0"); lit[n, 0] = v; arcs.append((n, 0, v)); m.Add(v == 1)         # dummy -> depot, forced
    m.AddCircuit(arcs); m.Minimize(sum(terms))
    if hint is not None:
        nx = _hint_next(list(hint), n)
        for (i, j), var in lit.items(): m.AddHint(var, 1 if nx.get(i) == j else 0)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = float(seconds); s.parameters.num_workers = int(workers)
    st = s.Solve(m); name = s.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, name, None, s.BestObjectiveBound() / scale
    nxt = {i: j for (i, j), var in lit.items() if s.Value(var)}
    route, cur = [0], 0
    while nxt[cur] != n:
        cur = nxt[cur]; route.append(cur)
    return route, name, s.ObjectiveValue() / scale, s.BestObjectiveBound() / scale


def mip_route(C, seconds=10.0, solver_name="SCIP", scale=10):
    """MIP baseline: asymmetric TSP path with the Miller-Tucker-Zemlin subtour-elimination formulation (the textbook
    compact MIP), solved by SCIP through OR-Tools' linear-solver wrapper. Same dummy-end-node trick as above.
    Returns (route or None, status, objective, best_bound)."""
    from ortools.linear_solver import pywraplp
    C = np.asarray(C); n = len(C); Ci = np.rint(C * scale)
    sol = pywraplp.Solver.CreateSolver(solver_name)
    if sol is None: raise RuntimeError(f"solver {solver_name} not available")
    x = {}
    for i in range(n):
        for j in range(1, n):
            if i != j: x[i, j] = sol.BoolVar(f"x{i}_{j}")
    for i in range(1, n): x[i, n] = sol.BoolVar(f"e{i}")                       # to the dummy end node (cost 0)
    x[n, 0] = sol.BoolVar("d0"); sol.Add(x[n, 0] == 1)
    for i in range(n + 1):                                                      # one successor and one predecessor each
        sol.Add(sum(v for (a, b), v in x.items() if a == i) == 1)
        sol.Add(sum(v for (a, b), v in x.items() if b == i) == 1)
    u = {i: sol.NumVar(1, n, f"u{i}") for i in range(1, n + 1)}                # MTZ order variables
    for (i, j), v in x.items():
        if i >= 1 and j >= 1: sol.Add(u[i] - u[j] + n * v <= n - 1)
    sol.Minimize(sum(float(Ci[i, j]) * v for (i, j), v in x.items() if i < n and j < n))   # arcs touching the dummy cost 0
    sol.SetTimeLimit(int(seconds * 1000))
    st = sol.Solve()
    name = {0: "OPTIMAL", 1: "FEASIBLE", 2: "INFEASIBLE", 3: "UNBOUNDED", 6: "NOT_SOLVED"}.get(st, str(st))
    if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, name, None, sol.Objective().BestBound() / scale
    nxt = {i: j for (i, j), v in x.items() if v.solution_value() > 0.5}
    route, cur = [0], 0
    while nxt[cur] != n:
        cur = nxt[cur]; route.append(cur)
    return route, name, sol.Objective().Value() / scale, sol.Objective().BestBound() / scale
