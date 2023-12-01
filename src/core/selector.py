## @package evolearn
#  Module selector

import typing
import abc
from abc import abstractmethod
from typing import Tuple
from typing import List
from typing import Optional
from typing import Iterator
from core.globals import report
from core.globals import LogLevel
from core.population import Genome
from core.population import Iterable
from core.population import Population
from core.population import GenomePool
from core.evaluator import Evaluator

T = typing.TypeVar("T", bound=Genome)

class Selector(abc.ABC, typing.Generic[T]):
    """!An abstract selector
        A selector that can be applied as a parent selector or a survivor selector. 
    """

    def __init__(self: typing.Self, coarity: int, budget: int):
        self.coarity = coarity
        self.budget = budget

    def select_to_pool(self,
                       population: Population[T],
                       evaluator: Evaluator[T],
                       budget: Optional[int] = None) -> GenomePool[T]:
        """!Select to tuples of parents
            Select to a GenomePool instance, which can be passed to the variator. The arity of the returned value depends on the arity of the selector.
            If the population cannot exactly fill tuples of a given size, discard the left-over genomes.
        """
        selected = self.select_to_many(population, evaluator, budget)
        
        # Tuple magic. Zipping an iterable with itself extracts a tuple of that size. The "discarding" behaviour is implemented this way.
        ai:Iterator[T] = iter(selected)
        output_tuples:List[Tuple[T, ...]] = list(zip(*tuple(ai for i in range(self.coarity))))
        pool = GenomePool[T](self.coarity)

        for x in output_tuples:
            pool.append(x)

        return pool

    def select_to_population(self,
                             population: Population[T],
                             evaluator: Evaluator[T],
                             budget: Optional[int] = None) -> Population[T]:
        """!Select to a Population
            Select to a Population instance. This might happen, for example, while the selector is acting as a survivor selector.
        """
        selected = self.select_to_many(population, evaluator, budget)
        new_population = Population[T]()
        for x in selected:
            new_population.append(x)

        return new_population
    
    def select_to_many(self, population: Population[T], evaluator: Evaluator[T], budget: Optional[int] = None) -> Iterable[T]:
        """!Many-to-many selection strategy.
            Repeatedly apply select() to create a collection of solutions
            @param popoulation: the input population
            @param evaluator: the evaluator that selects from the input population
            @param budget: the size of the returned collection.
        """
        old_population: Population[T] = population
        return_list: List[T] = []

        if budget is None:
            budget = self.budget
        budget = min(budget, len(old_population))
        budget_used: int = 0
        while budget_used < budget:
            return_list.append(self.select(old_population, evaluator))
            budget_used = budget_used + 1

        return return_list

    @abc.abstractmethod
    def select(self, 
               parents: Population[T],
               evaluator: Evaluator[T]) -> T:
        """!Many-to-one selection strategy
            Select, possibly stochastically, a solution from the population
            @param parents: the input population
            @param evaluator: the evaluator that selects from the input population
            @sideeffect Each call takes one member from the input population
        """
        pass


class SimpleSelector(Selector[T]):
    def __init__(self: typing.Self, coarity:int, budget: int):
        super().__init__(coarity, budget)

    def select(self,
               population: Population[T],
               evaluator: Evaluator[T]) -> T:
        """!A one-to-one selection strategy.
            Select the solution with highest fitness.
        """
        solutions = population
        
        best_score: Optional[float] = None
        best_index: int
        best_solution: T

        for i in range(0, len(solutions)):
            current_genome = solutions[i]
            # If the genome is not scored, score it.
            # At this point, evaluating a genome also scores it. This might be redundant.
            if current_genome.score is None:
                
                current_genome_scpre = evaluator.evaluate(solutions[i])
                current_genome.score = current_genome_scpre

            if best_score is None or current_genome.score > best_score:
                best_index, best_score, best_solution = (i, current_genome.score, solutions[i])

        selected_solution = solutions.draw(best_index)
        report(LogLevel.TRC, f"Solution selected: {str(selected_solution)}")
        return selected_solution
    
class ElitistSimpleSelector(SimpleSelector[T]):
    def __init__(self: typing.Self, coarity:int, budget: int):
        super().__init__(coarity, budget-1)
        self.best_genome: Optional[T] = None

    def select_to_many(self, population: Population[T], evaluator: Evaluator[T], budget: Optional[int] = None) -> Iterable[T]:
        """!A many-to-many selection strategy.
            Preserve and update an elite, insert the elite to the resulted population.
        """

        results: Iterable[T] = super().select_to_many(population, evaluator, budget)
        best_genome: Optional[T] = self.best_genome

        for g in results:
            if best_genome is None:                
                best_genome = g.copy()
                best_genome.score = g.score
            elif best_genome.score is None:
                raise Exception("Evaluator scoring heuristic failed, best_genome exists without a score.")
            elif g.score is None:
                raise Exception("Evaluator scoring heuristic failed, a solution is evaluated but unscored.")
            elif best_genome.score < g.score:
                report(LogLevel.INF, f"Elitism activated: score {best_genome.score} -> {g.score} by {g.score - best_genome.score}")
                best_genome = g.copy()
                best_genome.score = g.score
                

        if best_genome is not None:
            self.best_genome = best_genome
            return [*results, best_genome]
        else:
            raise Exception("Mypi forces me to do this.")
    
# The idea of having a selector decorator is highly restrictive - if it cannot tap into the "inner working" of the selector,
#   then it cannot do much beyond pre- and post-processing.
# On the other hand, the evaluator heuristic of, for example, pre-evaluation, might be useful.
# Hope that does not come back to bite me again.

# class SelectorDecorator(Selector[T]):
#     """!Many-to-many selection strategy.
#         Repeatedly applying strategy to create a collection of solutions.
#         """
#     def __init__(self: typing.Self, selector: Selector[T]):
#         self.selector = selector

#     def select_to_pool(self, *args) -> GenomePool[T]:
#         return self.selector.select_to_pool(*args)
    
#     def select_to_population(self, *args) -> Population[T]:
#         return self.selector.select_to_population(*args)
    
#     def select(self, *args) -> T:
#         return self.select(*args)
    
# class ElitisSelectorDecorator(SelectorDecorator[T]):
#     def select(self, *args) -> T:
#         pop = self.select(*args)
#         best_score: float
#         best_genome: T
#         for genome in pop:
#             if genome.score is None:
#                 raise Exception("Genome score is none! The genome should have already been assigned a score. This should not happen.")
#             if best_score is None:
#                 best_score = genome.score
#             elif best_score < genome.score:
#                 best_score = genome.score
#                 best_genome = genome.copy()
#         pop.append(best_genome)
#         return pop