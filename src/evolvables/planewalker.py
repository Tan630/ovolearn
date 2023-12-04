from __future__ import annotations
from typing import Generic
from typing import Self
import random
from abc import ABC
import numpy

from abc import abstractmethod

from core.evaluator import Evaluator
from core.population import Genome
from core.population import Population
from core.selector import Elitist
from core.selector import TournamentSelector
from core.controller import Controller
from typing import Callable
from typing import List

from core.population import Tuple
from core.population import Iterable
from core.variator import Variator
import math

class Position(Genome[float]):
    """!A solution
        A genotypical representation of a solution that specifies the capabilities a solution must have in order to work with evolutionary operators.
    """
    def __init__(self, pos: List[float]) -> None:
        super().__init__()
        self.pos = pos
    
    def copy(self) -> Self:
        return self.__class__(self.pos)
    
    @classmethod
    def create_random(cls, bounds: Tuple[Tuple[float, float], ...])-> Self:
        return cls([random.random() * (max(x) - min(x)) + min(x) for x in bounds])
    
    def __len__(self) -> int:
        return len(self.pos)
    
    def __str__(self)-> str:
        return str(self.pos)

class FunctionEvaluator(Evaluator[Position]):
    def __init__(self, function: Callable):
        self.function = function

    @Evaluator.evaluate_shortcut
    def evaluate(self: Self, s1: Position)-> float:
        return -self.function(*s1.pos)

class StepMutator(Variator[Position]):
    def __init__ (self, neighbour_size):
        super().__init__(1, neighbour_size, False)

    def vary(self, parents: Tuple[Position, ...], step:float = 1) -> Tuple[Position, ...]:
        # Create a random unit vector: create a random vector then normalize it
        direction1 = StepMutator.normalise(
            StepMutator.random_vector(len(parents[0])))
        # Scale the random unit vector by the given step size
        destination1 = [x * step for x in direction1]

        direction2 = StepMutator.normalise(
            StepMutator.random_vector(len(parents[0])))
        destination2 = [x * step for x in direction2]

        return (Position(destination1), Position(destination2))
        
    @staticmethod
    def normalise(pos: Iterable[float]) -> List[float]:
        len = sum(a**2 for a in pos)
        return [a / len for a in pos]
    
    @staticmethod
    def random_vector(d: int) -> List[float]:
        """Implementation fault: only returns an 2D vector.
        """
        return [numpy.random.normal() for i in range(d)]


def himmelblau(x:float, y:float):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

init_pop = Population[Position]()


for i in range (100):
    init_pop.append(Position.create_random(((-5,5), (-5,5))))



evaluator = FunctionEvaluator(himmelblau)
selector = Elitist(TournamentSelector[Position](1, 10))
variator = StepMutator(3)

ctrl = Controller[Position](
    population = init_pop,
    evaluator = evaluator,
    variator = variator,
    survivor_selector = selector,
    parent_selector = selector
)

dicts = {}
for i in range(0, 1000):
    ctrl.step()
    dicts[i] = ctrl.population[0].score

print (dicts)



X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = himmelblau(X, Y)
#Z = rosenbrock(X, Y)

fig = plt.figure(figsize=(8, 6))
cs = plt.contour(X, Y, Z, levels=100, cmap='Spectral',
                 norm=colors.Normalize(vmin=Z.min(), vmax=Z.max()), alpha=0.4)
fig.colorbar(cs)


lists = [x.pos for x in ctrl.population]
xs = [x[0] for x in lists]
ys = [x[1] for x in lists]
plt.scatter(xs, ys)
plt.show()


plt.show()





# print (xs)
# print (ys)

# plt.scatter(xs, ys)
# plt.show()


# evalA = FunctionEvaluator(himmelblau)
# posA = Position([5,40])







# for i in range (0, 10):
#     init_pop.append(Binary.create_random(10))

# evaluator = BitDistanceEvaluator()
# selector = Elitist(TournamentSelector[Binary](1, 10))
# variator = RandomBitMutator()

# ctrl = Controller[Binary](
#     population = init_pop,
#     evaluator = evaluator,
#     variator = variator,
#     survivor_selector = selector,
#     parent_selector = selector
# )

# dicts : typing.Dict[int, Optional[float]]= {}

# for i in range(0, 100):
#     ctrl.step()
#     dicts[i] = ctrl.population[0].score

# print (dicts)



