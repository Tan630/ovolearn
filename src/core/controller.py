

import typing
import abc

from typing import Callable
from typing import Self
from typing import Optional
from core.population import Population
from core.population import Genome
from core.evaluator import Evaluator
from core.variator import Variator
from core.selector import Selector
from core.globals import report
from core.globals import LogLevel


T = typing.TypeVar("T", bound = Genome)
TK = typing.TypeVar("TK", bound=Evaluator)
TV = typing.TypeVar("TV", bound=Variator)

class Controller(typing.Generic[T]):
    """Encapsulated environment where evolutionary learning takes place.
    """
    def __init__(self,
            population: Population[T],
            evaluator: Evaluator[T],
            parent_selector: Selector[T],
            variator: Variator[T],
            survivor_selector: Selector[T],
    ) -> None:
        """!Initialize an evolution controller with a set of configurable components.
            @param population: the initial population
            @param evaluator: The evaluator acts on an individual to determine its fitness.
            @param parent_selector: The parent selector applies to the population before variation. The range of the parent selector must match the domain of the variator.
            @param variator: The variator receives a collection of elements and outputs a list of genomes. These genomes are deposited into the population.
            @param survivor_selector: The parent selector that is applied before variation.
        """
        self.population = population
        self.evaluator = evaluator
        self.parent_selector = parent_selector
        self.variator = variator
        self.survivor_selector = survivor_selector
        self.is_done = False
        self.generation = 0

    def step(self, accountant: Optional[Callable[[Self], None]]= None) -> Self:
        """!Improve the population by one generation.
            @param accountant:  act on the population after the survivor selector
            @return: the population that results from the step
        """
        
        if (self.done()):
            return self
        self.generation = self.generation + 1
        
        # Apply the parent selector
        pop_before_parent_selection = len(self.population)
        parents = self.parent_selector.select_to_pool(self.population, self.evaluator)
        
        # Apply the child selector
        pop_before_variation = len(parents) * len(parents[0])
        children = self.variator.vary_pool(parents)
        
        # Apply the survivor selector
        pop_before_survivor_selection = len(children)
        survivors = self.survivor_selector.select_to_population(children, self.evaluator)

        # The survivor become the next population.
        pop_after_survivor_selection = len(survivors)
        self.population = survivors
        self.population.sort()
        
        # Accounting for error reporting purposes. 
        report(LogLevel.DBG, f"Population progress: [{pop_before_parent_selection}]"
               f" -> (...)>PAR>({self.parent_selector.coarity})"
               f" -> [({self.parent_selector.coarity})~{pop_before_variation}]"
               f" -> ({self.variator.arity}) VAR ({self.variator.coarity})"
               f" -> [{pop_before_survivor_selection}] -> SUR"
               f" -> [{pop_after_survivor_selection}]"
               f", gen {self.generation}."
               f" Best score is now {self.population[0].score}")
        
        empirical_best_score = self.evaluator.evaluate(self.population[0])
        
        if (self.population[0].score != empirical_best_score):
            report(LogLevel.WRN, f"Score inconsistent: cached {self.population[0].score}, actual {empirical_best_score}")

        if (pop_before_variation / self.variator.arity * self.variator.coarity != pop_before_survivor_selection):
            report(LogLevel.WRN, f"Variator arity inconsistent with population growth. {int(pop_before_variation / self.variator.arity * self.variator.coarity)} <> {pop_before_survivor_selection}")

        if (accountant is not None):
            accountant(self)

        print ([self.evaluator.evaluate(x) for x in self.population])

        report(LogLevel.WRN, f"Best solution is: {str(self.population[0])}")

        return self
    
    def varity_check(self):
        #     population: Population[T],
        #     evaluator: Evaluator[T],
        #     parent_selector: Selector[T],
        #     variator: Variator[T],
        #     survivor_selector: Selector[T],
        report(LogLevel.INF, f"parent selection: {self.parent_selector.coarity} -> {self.variator.arity}")


    def done(self) -> bool:
        """!@TODO This behaviour is not implemented.
        """
        return self.is_done
    
    @property
    def max_steps(self) -> int:
        return self._max_steps
    
    @max_steps.setter
    def max_steps(self, val: int) -> None:
        self._max_steps = val




