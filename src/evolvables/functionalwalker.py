from core.controller import Controller
from core.population import Population
from core.selector import Elitist, TournamentSelector
from core.selector import Elitist, TournamentSelector
from typing import Callable

from evolvables.planewalker import Position, FunctionEvaluator, StepMutator


class FunctionalStepMutator(StepMutator):
    def __init__ (self, neighbour_size):
        super().__init__(1, neighbour_size)

    def vary(self, parents: tuple[Position, ...], step:float|Callable[[Position], float] = 1) -> tuple[Position, ...]:
        if isinstance (step, float):
            return super().vary(parents, step)
        else:
            return super().vary(parents, step(parents[0]))

