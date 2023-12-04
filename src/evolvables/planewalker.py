from typing import Generic
from typing import Self
import random
from abc import ABC
import numpy

from abc import abstractmethod

from core.evaluator import Evaluator
from core.population import Genome
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
    
    def __len__(self) -> int:
        return len(self.pos)

class FunctionEvaluator(Position):
    def __init__(self, function: Callable):
        self.function = function

    @Evaluator.evaluate_shortcut
    def evaluate(self: Self, s1: Position)-> float:
        return self.function(*s1.pos)

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
    return (x**2+y-11) + (x + y**2 - 7)




evalA = FunctionEvaluator(himmelblau)
posA = Position([5,40])



print (evalA.evaluate(posA))


