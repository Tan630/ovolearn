from core.controller import Controller
from core.population import Population
from core.selector import Elitist, TournamentSelector, SimpleSelector
from typing import Callable

from evolvables.planewalker import Position, FunctionEvaluator, FunctionalStepMutator


    

from typing import Any

def analyse_function(objective: Callable[..., float],
                     stepper: Callable[[Position], float],
                     pop_size: int,
                     pop_range: tuple[tuple[float, float], ...],
                     episode_count:int,
                     step_count:int) -> list[list[Position]]:
    init_pop = Population[Position]()
    for i in range (pop_size):
        init_pop.append(Position.create_random(pop_range))

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

    list_of_lists_of_bests : list[list[Position]]  = []
    for i in range (0, episode_count):
        bests: list[Position] = []
        for ii in range(0, step_count):
            ctrl.step()
            bests.append(ctrl.population[0])
        list_of_lists_of_bests.append(bests)
    return list_of_lists_of_bests

    
    
def himmelblau(x:float, y:float)-> float:
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

pops1 = analyse_function(himmelblau, lambda x : 1, pop_size = 1, pop_range = ((-5,5), (-5,5)), episode_count = 5, step_count = 20)