from cso import CSO
from cso.fitness import fitness_4
from random import randint

CLIENTS = [
    (2,1),
    (3,2),
    (2,3),
    (4,4),
    (3,5),
    (4,6),
    (8,4),
    (9,5),
    (10,1),
    (6,10),
]
FACILITIES = [
    (4,1),
    (8,3),
    (3,7),
]

def assert_is_coordinates(client_or_facility: list):
    assert isinstance(client_or_facility, list)
    assert all([
            isinstance(client, tuple) and (len(client) == 2)
            for client in client_or_facility
        ]
    )
    return randint(0, 10)

def fitness(clients: list, facilities: list):
    assert_is_coordinates(clients)
    assert_is_coordinates(facilities)

# def run():
#     CSO(fitness=fitness, )

CSO(fitness=fitness_4, bound=[(-4,4),(-4,4)], min=False).execute()
