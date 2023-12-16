from evolvables.expression import ProgramFactory, Program, ProgramCrossoverVariator
from core.evaluator import Evaluator
from core.population import Population
from core.selector import SimpleSelector, Elitist
from core.controller import Controller
from typing import Callable
from evolvables.planewalker import WalkerEvaluator, score_function
import math


def neg(x):
    return -x

def add(x, y):
    return x+y

def sub(x, y):
    return x-y

def mul(x, y):
    return x*y

def div(x, y):
    return 0 if y==0 else x/y

def log(x):
    abs_x = abs(x)
    return 0 if (abs_x == 0) else math.log(abs(x))

def lim(x, a, b):
    return min(max(min(a,b), x), max(a,b))

def avg(x, y):
    return (x+y)/2

def val0():
    return 0

def val1():
    return 1


def himmelblau(x:float, y:float)-> float:
    if x < -5 or x > 5 or y < -5 or y > 5:
        return 1000
    else:
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2




# ########## Begin setup :) ########## #

# Size of the population. Affects the size of the initial initial population, also enforced by selectors.
pop_size = 100
tree_depth = 3
node_budget = 10
episode_bound = 10
step_bound = 25

sub_episode_bound = 40
sub_step_bound = 25

progf = ProgramFactory((abs, neg, add, sub, mul, div, log, lim, avg, val0, val1), 3)

pops: Population[Program[float]] = Population()
for i in range(0, pop_size):
    pops.append(progf.build(tree_depth, node_budget))

variator = ProgramCrossoverVariator(arity = 2, coarity = 3)

evaluator = WalkerEvaluator(himmelblau, sub_episode_bound, sub_step_bound)

# Prepare the selector.
import gymnasium as gym
selc = Elitist(SimpleSelector[Program[float]](coarity = 2, budget = pop_size))
selp = SimpleSelector[Program[float]](coarity = 2, budget = pop_size)

ctrl = Controller[Program[float]](
    population = pops,
    evaluator = evaluator,
    parent_selector = selc,
    variator = variator,
    survivor_selector = selp
)

best_solutions: list[Program] = []
best_scores: list[Program] = []

def score_keeper(best_scores, best_solutions, c: Controller[Program]):
    best_solutions = best_solutions.append(c.population[0])
    best_scores = best_scores.append(c.population[0].score)

from functools import partial
for i in range(0, 20):
    print ("a")
    ctrl.step(partial(score_keeper, best_scores, best_solutions))
    print ("b")


print ([str(x) for x in best_solutions])
print (str(best_scores))


