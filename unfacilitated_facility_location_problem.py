from cso import CSO
from cso.fitness import fitness_4
from random import randint, choice

from collections import namedtuple, defaultdict

Client = namedtuple('Client', ['x', 'y'])
Facility = namedtuple('Facility', ['x', 'y'])

CLIENTS = [
    Client(2,1),
    Client(3,2),
    Client(2,3),
    Client(4,4),
    Client(3,5),
    Client(4,6),
    Client(8,4),
    Client(9,5),
    Client(10,1),
    Client(6,10),
]
FACILITIES = [
    Facility(4,1),
    Facility(8,3),
    Facility(3,7),
]

def assert_are_coordinates(client_or_facility: list):
    assert isinstance(client_or_facility, list)
    assert all([
            isinstance(client, tuple) and client.x and client.y
            for client in client_or_facility
        ]
    )

def distance_of_coordinates(coord_1: tuple, coord_2: tuple) -> float:
    assert_are_coordinates([coord_1, coord_2])
    return ((coord_1.x - coord_2.x)**2 + (coord_1.y - coord_2.y)**2)**0.5


def _verify_solution(mapping: dict):
    assert isinstance(mapping, dict)
    assert len(mapping) == len(FACILITIES)
    assert all([
        isinstance(facility, Facility) for facility in mapping.keys()
    ])
    assert all([
        facility in mapping.keys() for facility in FACILITIES
    ])

    assert all([
        isinstance(clients_list, list) for clients_list in mapping.values()
    ])
    all_clients = []
    for client_list in mapping.values():
        assert all([
            isinstance(client, Client) for client in client_list
        ])
        all_clients.extend(client_list)
    assert len(all_clients) == len(CLIENTS)
    assert([
        client in client_list for client in CLIENTS
    ])


def fitness(mapping: dict):
    _verify_solution(mapping=mapping)
    # TODO make sure that client is not assigned to more than one facility

    _fitness = 0

    for facility, clients in mapping.items():
        for client in clients:
            _fitness += distance_of_coordinates(
                facility,
                client,
            )
    return _fitness


def _generate_random_solution() -> dict:
    solution = {}
    for f in FACILITIES:
        solution[f] = []

    for client in CLIENTS:
        assign_to_facility = choice(FACILITIES)
        if assign_to_facility in solution:
            solution[assign_to_facility].append(client) 
        else:
            solution[assign_to_facility] = [client]
    return solution


# def run():
#     CSO(fitness=fitness, )

# sample_mapping = {
#     FACILITIES[0]: [CLIENTS[0], CLIENTS[1]],
#     FACILITIES[1]: [CLIENTS[2], CLIENTS[3]],
#     FACILITIES[2]: CLIENTS[4:],
# }
# print(fitness(mapping=_generate_random_solution()))

# CSO(fitness=fitness, bound=[(-4,4),(-4,4)], min=True).execute()

POPULATION_SIZE = 10
MAX_GENERATION = 100_000
pa = 0.25
# Generate random solution
random_solutions = [_generate_random_solution() for _ in range(POPULATION_SIZE)]

iter = 0
global_best_solution = {
    "solution": None,
    "fitness": 999999,
}

while (iter < MAX_GENERATION): # OR STOP CRITERIA
    # generate random facility (our cuckoo) via levy flight
    fitness_fi = 999999 # TODO Levy flight generated solution

    # Choose a nest among the population randomly
    solution = choice(random_solutions)
    fitness_fj = fitness(mapping=solution)

    # if fitness_fi <= fitness_fj:
    #     # replace the nest with new solution
    #     random_solutions.remove(solution)
        # random_solutions.append(fitness_fi)

    # A fraction (pa) of worse nests are discovered with a probability pa
    solutions_with_fitness = []
    for s in random_solutions:
        # print(s)
        solutions_with_fitness.append({
            'solution': s,
            'fitness': fitness(mapping=s),
        })

    # ABANDON pa% of the worst solutions and generate new random ones    
    SOLUTIONS_TO_BE_DROPPED = int(pa*POPULATION_SIZE)
    remaining = sorted(solutions_with_fitness, key=lambda x: x['fitness'])[:SOLUTIONS_TO_BE_DROPPED]
    random_solutions = [s['solution'] for s in remaining]
    
    if global_best_solution['fitness'] > remaining[0]['fitness']:
        global_best_solution = remaining[0]

    while len(random_solutions) < (POPULATION_SIZE):
        random_solutions.append(_generate_random_solution())
    iter += 1


# Find the best solution
for s in random_solutions:
    solutions_with_fitness.append({
        'solution': s,
        'fitness': fitness(mapping=s),
    })

best_solution = sorted(solutions_with_fitness, key=lambda x: x['fitness'])[-1]
# BEST SOLUTION
print(best_solution['solution'])
print(best_solution['fitness'])

# GLOBAL BEST SOLUTION
print(global_best_solution['solution'])
print(global_best_solution['fitness'])
