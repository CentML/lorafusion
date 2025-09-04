# ruff: noqa: N806
"""Proposed solver for the micro-batch scheduling problem, minimizing latency."""

import multiprocessing

import numpy as np
import pulp
from loguru import logger

M = 10000000
SMALL_NUMBER = 0.0000001
ZERO_POINT_FIVE = 0.5


def solve_schedule_milp(  # noqa: C901, PLR0912, PLR0915
    data: list[list[list[int]]],
    num_microbatches: int,
    num_stages: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    *,
    time_limit: int = 600,
    verbose: bool = False,
) -> list[list[tuple[int, int, int]]]:
    """Solve the micro-batch scheduling MILP to minimize end-to-end pipeline latency.

    Args:
        data:
            nested list data[i][j][k] = token-count of sample k of batch j of adapter i.
        num_microbatches: int, number of micro-batches (L).
        num_stages: int, number of pipeline stages (S).
        alpha: float, forward-time per token.
        beta: float, backward-time per token.
        time_limit: int, solve time limit (seconds).
        verbose: bool, whether to show solver logs.

    Returns:
        schedule: list of length L, each micro-batch is list[(i,j,k)].
    """
    # flatten
    samples = []
    for i, adapter in enumerate(data):
        for j, batch in enumerate(adapter):
            for k, d in enumerate(batch):
                samples.append((i, j, k, d))

    # Check if the problem is feasible
    total_samples = len(samples)
    if total_samples == 0:
        logger.warning("No samples to schedule!")
        return [[] for _ in range(num_microbatches)]

    # Ensure num_microbatches is sufficient for the number of samples and strict
    # constraints
    required_microbatches = max(total_samples, num_stages * len(data))
    if num_microbatches < required_microbatches:
        old_num_microbatches = num_microbatches
        num_microbatches = required_microbatches
        logger.warning(
            f"Increased num_microbatches from {old_num_microbatches} to "
            f"{num_microbatches} to ensure feasibility"
        )

    L, S = num_microbatches, num_stages
    logger.info(
        f"Solving MILP with {total_samples} samples, {L} microbatches, {S} stages"
    )

    prob = pulp.LpProblem("Pipeline_Latency_Minimization", pulp.LpMinimize)

    # 1) Assignment binaries x[i,j,k,t]
    x = {
        (i, j, k, t): pulp.LpVariable(f"x_{i}_{j}_{k}_{t}", cat="Binary")
        for (i, j, k, _) in samples
        for t in range(L)
    }

    # 1) Each sample must be assigned to exactly one microbatch
    for i, j, k, _ in samples:
        prob += pulp.lpSum(x[(i, j, k, t)] for t in range(L)) == 1

    # 2) Strict S-delayed draining
    for i, adapter in enumerate(data):
        for j in range(len(adapter) - 1):
            B_j = len(adapter[j])
            for k2 in range(len(adapter[j + 1])):
                for t in range(L):
                    if t < S:
                        prob += x[(i, j + 1, k2, t)] == 0
                    else:
                        # if x[(i, j + 1, k2, t)] == 1, then all B_j samples of batch j
                        # must have been scheduled in {0…t-S}
                        prob += (
                            pulp.lpSum(
                                x[(i, j, k, t2)]
                                for k in range(B_j)
                                for t2 in range(t - S + 1)
                            )
                            >= B_j * x[(i, j + 1, k2, t)]
                        )

    # 3) Micro-batch loads T[t]
    T = {t: pulp.LpVariable(f"T_{t}", lowBound=0) for t in range(L)}
    for t in range(L):
        prob += T[t] == pulp.lpSum(d * x[(i, j, k, t)] for (i, j, k, d) in samples)

    # 4) If T[t] == 0, then all T[t2] == 0 for t2 > t
    # Use binary variables to model this constraint
    has_tokens = {t: pulp.LpVariable(f"has_tokens_{t}", cat="Binary") for t in range(L)}

    # Link has_tokens with T
    for t in range(L):
        # If T[t] > 0, then has_tokens[t] must be 1
        # If T[t] = 0, then has_tokens[t] must be 0
        prob += T[t] <= M * has_tokens[t]
        prob += (
            has_tokens[t] <= T[t] * M
        )  # This ensures has_tokens[t] is 0 when T[t] is 0

    # If has_tokens[t] = 0, then has_tokens[t+1:] must all be 0
    for t in range(L - 1):
        prob += has_tokens[t + 1] <= has_tokens[t]

    # 4) Forward/backward timing variables
    Fs = {
        (s, t): pulp.LpVariable(f"Fs_{s}_{t}", lowBound=0)
        for s in range(S)
        for t in range(L)
    }
    Fe = {
        (s, t): pulp.LpVariable(f"Fe_{s}_{t}", lowBound=0)
        for s in range(S)
        for t in range(L)
    }
    Bs = {
        (s, t): pulp.LpVariable(f"Bs_{s}_{t}", lowBound=0)
        for s in range(S)
        for t in range(L)
    }
    Be = {
        (s, t): pulp.LpVariable(f"Be_{s}_{t}", lowBound=0)
        for s in range(S)
        for t in range(L)
    }
    # The ending time of the first microbatch should be larger than 0
    prob += Fs[(0, 0)] == 0
    prob += Fe[(0, 0)] >= SMALL_NUMBER
    prob += Be[(0, 0)] >= SMALL_NUMBER
    prob += Bs[(0, 0)] >= SMALL_NUMBER

    # 5) Link durations: Fe = Fs + alpha*T, Be = Bs + beta*T
    for s in range(S):
        for t in range(L):
            prob += Fe[(s, t)] == Fs[(s, t)] + alpha * T[t]
            prob += Be[(s, t)] == Bs[(s, t)] + beta * T[t]

    # 6) Pipeline dependencies
    # a) Forward/backward depends on previous microbatch forward/backward
    for s in range(S):
        for t in range(1, L):
            prob += Fs[(s, t)] >= Fe[(s, t - 1)]
            prob += Bs[(s, t)] >= Be[(s, t - 1)]
    # b) Forward depends on previous-stage forward
    for s in range(1, S):
        for t in range(L):
            prob += Fs[(s, t)] >= Fe[(s - 1, t)]
    # c) Backward depends on own forward (last stage) or next-stage backward
    for s in range(S - 1, -1, -1):
        for t in range(L):
            if s == S - 1:
                prob += Bs[(s, t)] >= Fe[(s, t)]
            else:
                prob += Bs[(s, t)] >= Be[(s + 1, t)]
    # d) Forward depends on memory-release: backward of same stage, t-S
    for s in range(S):
        for t in range(S - s, L):
            prob += Fs[(s, t)] >= Be[(s, t - (S - s))]
    # e) Backward of stage s microbatch t must be scheduled after stage s
    # microbatch t + (S - s - 1)
    for s in range(S):
        for t in range(L - (S - s - 1)):
            prob += Bs[(s, t)] >= Fe[(s, t + (S - s - 1))]

    # 7) End-to-end latency C_tot = finish time of last microbatch backward on stage 0
    C_tot = pulp.LpVariable("C_tot", lowBound=0)
    prob += C_tot >= Be[(0, L - 1)]

    # 8) Objective: minimize pipeline latency
    prob += C_tot

    # 9) Solve
    verbose = True
    solver = pulp.PULP_CBC_CMD(
        mip=True,
        msg=verbose,
        timeLimit=time_limit,
        threads=multiprocessing.cpu_count(),
        options=["cuts on", "presolve on", "heuristics on"],
    )

    try:
        prob.solve(solver)
    except pulp.LpSolverError as e:
        logger.error(f"Solver error after multiple attempts: {e}")
        msg = f"Solver error after multiple attempts: {e}"
        raise RuntimeError(msg) from e

    if prob.status != pulp.LpStatusOptimal:
        logger.warning(f"Warning: Non-optimal solution (status: {prob.status}).")

    # 10) Extract schedule
    schedule: list[list[tuple[int, int, int]]] = [[] for _ in range(L)]
    for i, j, k, _ in samples:
        for t in range(L):
            if pulp.value(x[(i, j, k, t)]) > ZERO_POINT_FIVE:
                schedule[t].append((i, j, k))
                break

    # Also extract the forward/backward times
    # Make numpy array of size (S, L)
    fwd_start = np.zeros((S, L))
    fwd_end = np.zeros((S, L))
    bwd_start = np.zeros((S, L))
    bwd_end = np.zeros((S, L))
    T_values = np.zeros(L)

    for s in range(S):
        for t in range(L):
            fwd_start[s, t] = pulp.value(Fs[(s, t)])
            fwd_end[s, t] = pulp.value(Fe[(s, t)])
            bwd_start[s, t] = pulp.value(Bs[(s, t)])
            bwd_end[s, t] = pulp.value(Be[(s, t)])
            T_values[t] = pulp.value(T[t])

    # Debug
    for s in range(S):
        for t in range(L):
            logger.debug(
                f"Stage {s}, microbatch {t}: "
                f"{fwd_start[s, t]=}, {fwd_end[s, t]=}, "
                f"{bwd_start[s, t]=}, {bwd_end[s, t]=}, "
                f"{T_values[t]=}"
            )

    return schedule, fwd_start, fwd_end, bwd_start, bwd_end
