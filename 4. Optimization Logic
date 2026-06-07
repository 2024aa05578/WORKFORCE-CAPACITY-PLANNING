
from pulp import *

def optimize_resources(demand_data, available_engineers):

    locations = list(demand_data.keys())

    problem = LpProblem("CapacityPlanning", LpMinimize)

    allocation = LpVariable.dicts(
        "Engineers",
        locations,
        lowBound=0,
        cat='Integer'
    )

    problem += lpSum([allocation[l] for l in locations])

    for loc in locations:
        problem += allocation[loc] >= demand_data[loc]

    problem += lpSum([allocation[l] for l in locations]) <= available_engineers

    problem.solve()

    result = {}

    for loc in locations:
        result[loc] = allocation[loc].varValue

    return result
