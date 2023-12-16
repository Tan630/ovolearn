import typing
from typing import Self, Callable
import abc
from core.population import Population
from core.population import Genome

T = typing.TypeVar("T", bound=Genome)

class ScoringException(Exception): ...

class Evaluator(abc.ABC, typing.Generic[T]):
    @staticmethod
    def evaluate_shortcut(func):
        """!Apply the "dynamic scoring" heuristic to the evaluate(.) method. 
        """
        
        def wrapper(*args, **kwargs) -> float:
            genome = args[1]
            if (not isinstance(genome, Genome)):
                raise (ScoringException("This thing is not a genome!"))
            elif (genome.is_scored()):
                return genome.score
            else:
                score: float = func(*args, **kwargs)
                genome.score = score
                return score
        return wrapper
    
    
    @abc.abstractmethod
    @evaluate_shortcut
    def evaluate(self: Self, s1: T)-> float:
        """!Evaluate an individual and return the score.
            Higher scores are better.
        """
        pass


    # @abc.abstractmethod
    # def truncate(self: Self) -> bool:
    #     """Return if a termination condition has been reached.
    #     """
    #     return False

    def evaluate_population(self: Self, pop: Population[T])-> Population[T]:
        """!Score every individual of a population
        """
        for x in pop:
            x.score = self.evaluate(x)
        return pop
    
    



    
