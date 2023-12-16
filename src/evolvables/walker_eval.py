from evolvables.expression import ProgramFactory, Program, ProgramCrossoverVariator
from core.evaluator import Evaluator
from core.population import Population
from core.selector import SimpleSelector, Elitist
from core.controller import Controller
from typing import Callable
from evolvables.planewalker import WalkerEvaluator
import math
def add(x, y):
    return x+y

def sub(x, y):
    return x-y

def mul(x, y):
    return x-y

def div(x, y):
    return 0 if y==0 else x/y

def log(x):
    abs_x = abs(x)
    if (abs_x == 0):
        return 0
    else:
        return math.log(abs(x))

def lim(x, y):
    return max(min(x,2), 0)

def avg(x, y):
    return (x+y)/2

def val05():
    return 0.5
def val1():
    return 1
def val2():
    return 2
def val3():
    return 3



    

# tree_depth, node_budget = 5, 10

# a = progf.build(tree_depth, node_budget)

#print (a)
#progf = ProgramFactory((add, sub, mul, div, mul, div, lim, avg), 4)


# e = progf.build(10, 4, 0.3)


# import math
def himmelblau(x:float, y:float)-> float:
    if x < -5 or x > 5 or y < -5 or y > 5:
        return 1000
    else:
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
# eval = 

# print (eval.evaluate(e))
# print (e)




# ########## Begin setup :) ########## #

# Size of the population. Affects the size of the initial initial population, also enforced by selectors.
pop_size = 100
tree_depth = 3
node_budget = 10
episode_bound = 1
step_bound = 50

sub_episode_bound = 1
sub_step_bound = 50


progf = ProgramFactory((add, sub, mul, div, log, lim, avg, val1, val2, val3, val05), 1)

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
    ctrl.step(partial(score_keeper, best_scores, best_solutions))


print ([str(x) for x in best_solutions])
print (str(best_scores))