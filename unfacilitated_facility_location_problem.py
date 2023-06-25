from cso import CSO
from cso.fitness import fitness_4
from random import randint, choice, uniform

from collections import namedtuple, defaultdict

import numpy as np
from math import gamma

# LEVY FLIGHT PARAMS
beta = 1.5
n = 1
pa = 0.25

POPULATION_SIZE = 100
MAX_GENERATION = 1_000

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



def _generate_levy_flight_walk(global_best: list) -> float:
    num = gamma(1+beta)*np.sin(np.pi*beta/2)
    den = gamma((1+beta)/2)*beta*(2**((beta-1)/2))
    σu = (num/den)**(1/beta)
    σv = 1
    u = np.random.normal(0, σu, n)
    v = np.random.normal(0, σv, n)
    S = u/(np.abs(v)**(1/beta))
    S = S[0]

    # dictify client positions based on facility
    # {
    #   Client(2,1): 2,
    #   Client(2,1): 0,
    #   ..
    # }
    random_solution = _generate_random_solution()
    clients = {}
    for index, (_, facility) in enumerate(random_solution.items()):
        for client in facility:
            clients[client] = index

    client_from_best = {}
    for index, (_, facility) in enumerate(global_best.items()):
        for client in facility:
            client_from_best[client] = index

    # Generate new solution from global best on random walk
    solution = {}
    for index, f in enumerate(FACILITIES):
        solution[index] = {
            "facility": f,
            "clients": []
        }

    # normalize solution    
    new_index = None
    for key, index in clients.items():
        while(new_index is None or (new_index < 0 or new_index >= len(FACILITIES))):
            new_index = round(index + (uniform(0, 1) * (0.1 * S * client_from_best[key])))
        solution[new_index]["clients"].append(key)

    normalized_solution = {}
    for index, soln in solution.items():
        normalized_solution[soln["facility"]] = soln["clients"]

    return normalized_solution


def run():
    # Generate random solution
    random_solutions = [_generate_random_solution() for _ in range(POPULATION_SIZE)]

    iter = 0

    # FIND GLOBAL BEST FROM RANDOM SOLUTIONS
    # A fraction (pa) of worse nests are discovered with a probability pa
    solutions_with_fitness = []
    for s in random_solutions:
        solutions_with_fitness.append({
            'solution': s,
            'fitness': fitness(mapping=s),
        })
    global_best_solution = sorted(solutions_with_fitness, key=lambda x: x['fitness'])[0]

    while (iter < MAX_GENERATION): # OR STOP CRITERIA
        # generate random facility (our cuckoo) via levy flight
        levy_flight_solution = _generate_levy_flight_walk(global_best=global_best_solution["solution"])
        fitness_fi = fitness(mapping=levy_flight_solution) # TODO Levy flight generated solution

        # Choose a nest among the population randomly
        solution = choice(random_solutions)
        fitness_fj = fitness(mapping=solution)

        if fitness_fi <= fitness_fj:
            # replace the nest with new solution
            random_solutions.remove(solution)
            random_solutions.append(levy_flight_solution)

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

        while len(remaining) < POPULATION_SIZE:
            current_solution = _generate_random_solution()
            remaining.append({
                "solution": current_solution,
                "fitness": fitness(mapping=current_solution)
            })
        
        current_best = sorted(remaining, key=lambda x: x['fitness'])[0]
        if global_best_solution['fitness'] > current_best['fitness']:
            global_best_solution = remaining[0]

        print(global_best_solution['fitness'])
        random_solutions = [s['solution'] for s in remaining]

        iter += 1

    return global_best_solution

solution = run()
print()
print("********* OUTPUT ***********")
print(solution['solution'])
print(solution['fitness'])
