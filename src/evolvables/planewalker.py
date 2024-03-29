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
    def __init__(self, pos: tuple[float, ...]) -> None:
        super().__init__()
        self.pos = pos
    
    def copy(self) -> Self:
        result = self.__class__(self.pos)
        result.score = self.score
        return result
    
    @classmethod
    def create_random(cls, bounds: Tuple[Tuple[float, float], ...])-> Self:
        return cls(tuple(random.random() * (max(x) - min(x)) + min(x) for x in bounds))
    
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
    def __init__ (self, neighbour_count, step:float = 1):
        super().__init__(1, neighbour_count)
        self.step = step

    def vary(self, parents: Tuple[Position, ...]) -> Tuple[Position, ...]:
        results : list[Position] = []
        
        for i in range(self.coarity):
            direction1 = StepMutator.normalise(
                StepMutator.random_vector(len(parents[0])))
            results.append(Position(tuple(x * self.step for x in direction1)))
        return (tuple(results))
        
    @staticmethod
    def normalise(pos: Iterable[float]) -> List[float]:
        len = sum(a**2 for a in pos)
        return [a / len for a in pos]
    
    @staticmethod
    def random_vector(d: int) -> List[float]:
        """Implementation fault: only returns an 2D vector.
        """
        return [numpy.random.normal() for i in range(d)]

class FunctionalStepMutator(Variator[Position]):
    def __init__ (self, neighbour_count, func:Callable[[float], float]):
        super().__init__(1, neighbour_count)
        self.func = func

    def vary(self, parents: Tuple[Position, ...]) -> Tuple[Position, ...]:
        results : list[Position] = []
        
        
        step_size = self.func(parents[0].score)
        parent:Position = parents[0]
        parent_pos:tuple[float, ...] = parent.pos

        
        for i in range(self.coarity):
            direction = StepMutator.normalise(StepMutator.random_vector(len(parents[0])))
            direction_scaled = [x * step_size for x in direction]
            
            new_point:tuple[float, ...] = tuple(sum(x) for x in zip(direction_scaled, parent_pos))
                
            results.append(Position(new_point))
        return (tuple(results))
        
    @staticmethod
    def normalise(pos: Iterable[float]) -> List[float]:
        len = sum(a**2 for a in pos)
        return [a / len for a in pos]
    
    @staticmethod
    def random_vector(d: int) -> List[float]:
        """Implementation fault: only returns an 2D vector.
        """
        return [numpy.random.normal() for i in range(d)]


