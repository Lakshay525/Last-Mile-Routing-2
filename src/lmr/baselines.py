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
