import abc
from abc import abstractmethod
import typing
import time
from typing import Optional
from typing import Self
from core.globals import report, LogLevel
from core.population import Population
from core.population import Genome
from core.population import Tuple
from core.population import GenomePool

T = typing.TypeVar("T", bound=Genome)

class ArityException(Exception):
    pass

class Variator(abc.ABC, typing.Generic[T]):
    def __init__(self, arity, coarity) -> None:
        self.arity = arity
        self.coarity = coarity

    @abc.abstractmethod
    def vary(self, parents: Tuple[T, ...]) -> Tuple[T, ...]:
        """!Appy the variator to a tuple of parents
            Produce a tuple of genomes from a tuple of genomes.
            The input and output tuple sizes should match the arity and coarity of this selector, respectively.
            Implementation note: clear the score first.
        """
        pass

    def vary_pool(self, pool: GenomePool[T], rate: typing.Any = None) -> Population[T] :
        """!Apply the variator to a pool of tuples of parents.
            Also applies the "dynamic scoring" heuristic: the score of a genome is assigned to it as a field.
            Inspecting this field instead of evaluating the solution may save cost.
            @TODO This heuristic has caused several problems in implementation. One more moving piece.
        """
        parent_pool_arity = pool.arity
        empiricalparent_pool_arity = len(pool[0])
        my_arity = len(pool[0])
        
        if (parent_pool_arity != empiricalparent_pool_arity):
            report(LogLevel.WRN, f"Parent pool arity inconsistent with empirical arity. ({parent_pool_arity}) <> ({empiricalparent_pool_arity})")
        if (my_arity != empiricalparent_pool_arity):
            report(LogLevel.WRN, f"Selector arity inconsistent with empirical arity. ({my_arity}) <> ({empiricalparent_pool_arity})")
        
        new_population = Population[T]()
        for pair in pool:
            results = self.vary(pair)
            for individual in results:
                new_population.append(individual)

        return new_population
    
class DefaultVariator(Variator[T]):
    """!The default variator does not change anything
    """
    def __init__(self):
        super().__init__(1, 1, False)

    @abc.abstractmethod
    def vary(self, parents: Tuple[T, ...]) -> Tuple[T, ...]:
        e = []
        for x in parents:
            e.append(x)
        return tuple(e)
