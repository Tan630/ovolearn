from core.controller import Controller
from core.population import Population
from evolvables.expression import Program
from core.evaluator import Evaluator
from core.selector import Elitist, TournamentSelector, SimpleSelector
from typing import Callable

from evolvables.planewalker import Position, FunctionEvaluator, FunctionalStepMutator


    

from typing import Any

def analyse_function(objective: Callable[..., float],
                     stepper: Callable[[float], float],
                     pop_size: int,
                     init_point: tuple[float, float],
                     episode_count:int,
                     step_count:int) -> list[list[Position]]:
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

    list_of_lists_of_bests : list[list[Position]]  = []
    for i in range (0, episode_count):
        bests: list[Position] = []
        for ii in range(0, step_count):
            ctrl.step()
            bests.append(ctrl.population[0])
        list_of_lists_of_bests.append(bests)
    return list_of_lists_of_bests

def score_function(objective: Callable[..., float],
                      stepper: Callable[[float], float],
                      pop_size: int = 40,
                      init_point: tuple[float, float] = (-2.5, 2.5),
                      episode_count:int = 10,
                      step_count:int = 50) -> float:
    
    total_score : float = 0
    for i in range (0, episode_count):
        episode_score = score_episode(objective, stepper, pop_size, init_point, step_count)
        print (episode_score)
        total_score += episode_score
    total_score /= episode_count
    return total_score

def score_episode(objective: Callable[..., float],
                      stepper: Callable[[float], float],
                      pop_size: int = 40,
                      init_point: tuple[float, float] = (-2.5, 2.5),
                      step_count:int = 50)-> float:
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
        episode_score += ctrl.population[0].score
    episode_score /= step_count
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
                      init_point = (-2.5, 2.5),
                      episode_count = self.episode_count,
                      step_count = self.step_count)