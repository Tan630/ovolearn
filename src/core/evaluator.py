import typing
from typing import Self
import abc
from core.population import Population
from core.population import Genome

T = typing.TypeVar("T", bound=Genome)

class Evaluator(abc.ABC, typing.Generic[T]):
    @staticmethod
    def evaluate_shortcut(func):
        """!Apply the "dynamic scoring" heuristic to the evaluate(.) method. 
            @TODO The same heuristic is implemented by default in the selector. It is prone to cause problems.
        """
        def wrapper(*args, **kwargs) -> float:
            genome = args[0]
            if (isinstance(genome, Genome) and genome.score is not None):
                return genome.score
            else:
                score: float = func(*args, **kwargs)
                genome.score = score
                return score
        return wrapper
    
    
    @abc.abstractmethod
    @evaluate_shortcut
    def evaluate(self: Self, s1: T)-> float:
        """Evaluate an individual and return the score.
            Higher scores are better.
        """
        pass

    

    # @abc.abstractmethod
    # def truncate(self: Self) -> bool:
    #     """Return if a termination condition has been reached.
    #     """
    #     return False

    def evaluate_population(self: Self, pop: Population[T])-> float:
        """Return the score of a population.
            Default implementation: evaluate the sum of the scores of individuals.
        """
        return_float = 0.
        for solution in pop:
            return_float = return_float + self.evaluate(solution)
        return return_float
    
    



    
