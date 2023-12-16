from __future__ import annotations
from typing import Generic
from typing import Self
import random
from abc import ABC
import numpy

from abc import abstractmethod

from typing import Sequence
from core.evaluator import Evaluator
from core.population import Genome, GenomePool
from core.population import Population
from core.selector import Elitist
from core.selector import TournamentSelector
from core.controller import Controller
from typing import Callable, Optional
from typing import List, Any

from core.population import Tuple
from core.population import Iterable
from core.variator import Variator
import math


from core.controller import Controller
from core.population import Population
from evolvables.expression import Program
from core.evaluator import Evaluator
from core.selector import Elitist, TournamentSelector, SimpleSelector
from typing import Callable

from typing import Any


class Position(Genome[float]):
    """!A position
        A genotypical representation of a n-dimensional vector
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
        return f"Pos{str(self.pos)}"

class FunctionEvaluator(Evaluator[Position]):
    def __init__(self, function: Callable):
        self.function = function

    @Evaluator.evaluate_shortcut
    def evaluate(self: Self, s1: Position)-> float:
        return -self.function(*s1.pos)

class FunctionalStepMutator(Variator[Position]):
    def __init__ (self, neighbour_count, func:Callable[[float, float, float], float]):
        super().__init__(1, neighbour_count)
        self.func = func
        self.p_best = None
        self.p_p_best = None

    def vary(self, parents: Tuple[Position, ...]) -> Tuple[Position, ...]:
        # Define the past best 
        

        results : list[Position] = []
        step_size = self.func(parents[0].score, self.p_best, self.p_p_best)
        parent:Position = parents[0]
        parent_pos:tuple[float, ...] = parent.pos

        for i in range(self.coarity):
            direction = normalise(random_vector(len(parents[0])))
            direction_scaled = [x * step_size for x in direction]
            new_point = tuple(sum(x) for x in zip(direction_scaled, parent_pos))
            results.append(Position(new_point))
        return (tuple(results))
    
    
    def vary_pool(self, pool: GenomePool[Position], rate: Any = None, p_best = None, p_p_best = None) -> Population[Position] :
        """!Apply the variator to a pool of tuples of parents.
            Also applies the "dynamic scoring" heuristic: the score of a genome is assigned to it as a field.
            Inspecting this field instead of evaluating the solution may save cost.
            @TODO This heuristic has caused several problems in implementation. One more moving piece.
        """
        self.p_best = p_best
        self.p_p_best = p_p_best

        pool_result = super().vary_pool(pool, rate)
        
        return pool_result


def normalise(pos: Sequence[float])-> Tuple[float, ...]:
    len = math.sqrt(pos[0]**2+pos[1]**2)
    return tuple(a / len for a in pos)


def random_vector(d: int) -> Tuple[float, ...]:
    return tuple(numpy.random.normal() for i in range(d))

def score_function(objective: Callable[..., float],
                      stepper: Callable[[float, float, float], float],
                      pop_size: int,
                      init_point: tuple[float, float],
                      episode_count:int,
                      step_count:int) -> float:
    
    total_score : float = 0
    for i in range (0, episode_count):
        episode_score = score_episode(objective, stepper, pop_size, init_point, step_count)
        
        total_score += episode_score
        
    total_score /= episode_count
    return total_score

def score_episode(objective: Callable[..., float],
                      stepper: Callable[[float, float, float], float],
                      pop_size: int,
                      init_point: tuple[float, float],
                      step_count:int)-> float:
    init_pop = Population[Position]()
    
    for i in range (pop_size):
        init_pop.append(Position(init_point))

    evaluator = FunctionEvaluator(objective)
    parentselector = Elitist(TournamentSelector[Position](1, 10))
    child = SimpleSelector[Position](1, pop_size)
    variator = FunctionalStepMutator(10, stepper)

    ctrl = Controller[Position](
        population = init_pop,
        evaluator = evaluator,
        variator = variator,
        survivor_selector = parentselector,
        parent_selector = child
    )

    episode_score: float = 0
    
    for i in range(0, step_count):
        ctrl.step()
        best_genome = max(ctrl.population, key = lambda x : x.score)
        
        if best_genome.score > -0.1:
            break
        elif best_genome.score < -300:
            return -1000
        else:
            episode_score -= 1
            
    return episode_score

class WalkerEvaluator(Evaluator[Program[float]]):
    def __init__(self, objective, episode_count: int, step_count: int):
        super().__init__()
        self.objective = objective
        self.episode_count = episode_count
        self.step_count = step_count

    @Evaluator.evaluate_shortcut
    def evaluate(self, s1: Program[float]) -> float:
        return score_function(self.objective,
                      s1,
                      pop_size = 1,
                      init_point = (0,0),
                      episode_count = self.episode_count,
                      step_count = self.step_count)
    


# def himmelblau(x:float, y:float)-> float:
#     if x < -5 or x > 5 or y < -5 or y > 5:
#         return 1000
#     else:
#         return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    

    
# episode_count = 500
# step_count = 1000

# test_function = lambda x : 2
# import random
# def random_point():
#     return (random.random() * 10 - 5, random.random() * 10 - 5)
    
# import numpy as np
# dicts = {}
# for i in np.arange(1, 4, 0.1):
#     dicts[i] = score_function(himmelblau,
#                       lambda x : i,
#                       pop_size = 1,
#                       init_point = random_point(),
#                       episode_count = episode_count,
#                       step_count = step_count)
# print (i)




# def analyse_function(objective: Callable[..., float],
#                      stepper: Callable[[float], float],
#                      pop_size: int,
#                      init_point: tuple[float, float],
#                      episode_count:int,
#                      step_count:int) -> list[list[Position]]:
#     init_pop = Population[Position]()
#     for i in range (pop_size):
#         init_pop.append(Position(init_point))

#     evaluator = FunctionEvaluator(objective)
#     parentselector = Elitist(TournamentSelector[Position](1, 10))
#     child = SimpleSelector[Position](1, pop_size)
#     variator = FunctionalStepMutator(10, stepper)

#     ctrl = Controller[Position](
#         population = init_pop,
#         evaluator = evaluator,
#         variator = variator,
#         survivor_selector = parentselector,
#         parent_selector = child
#     )

#     list_of_lists_of_bests : list[list[Position]]  = []
#     for i in range (0, episode_count):
#         bests: list[Position] = []
#         for ii in range(0, step_count):
#             ctrl.step()
#             bests.append(ctrl.population[0])
#         list_of_lists_of_bests.append(bests)
#     return list_of_lists_of_bests

