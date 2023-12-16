# %%
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


# %%
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
pop_size = 200
tree_depth = 4
node_budget = 6
step_bound = 30

sub_episode_bound = 80
sub_step_bound = 30

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
for i in range(0, step_bound):
    ctrl.step(partial(score_keeper, best_scores, best_solutions))


print ([str(x) for x in best_solutions])
print (str(best_scores))


# %% [markdown]
# Import

# %%
print ([str(x) for x in best_solutions])
print (best_scores)

['avg(lim(log(sym_1), sub(neg(sym_1), abs(val0)), val1), val0)', 'avg(lim(sub(neg(sym_1), abs(val0)), log(sym_1), val1), val0)', 'avg(lim(neg(val0), add(sym_2, val0), neg(val1)), val0)', 'avg(lim(neg(val1), lim(val1, val0, log(val0)), add(sym_2, val0)), val0)', 'avg(val0, lim(neg(val1), lim(val1, val0, log(val0)), add(sym_2, val0)))', 'avg(val0, lim(neg(val1), lim(val1, val0, log(val0)), add(sym_2, val0)))', 'avg(lim(log(val0), val1, neg(sym_3)), mul(val0, sub(abs(val0), val0)))', 'avg(lim(lim(log(val0), val1, neg(sym_3)), val1, neg(sym_3)), mul(val0, sub(abs(val0), val0)))', 'sub(lim(neg(val1), lim(sym_1, sym_3, val1), val0), avg(neg(val1), abs(val0)))', 'lim(div(lim(sym_3, lim(val1, mul(sym_2, sym_2), val0), sub(add(val0, val0), sym_2)), div(sub(sub(val1, val0), neg(val1)), val1)), val1, log(sym_1))', 'lim(avg(log(sym_1), val0), sym_1, neg(lim(sym_3, neg(val1), log(sym_3))))', 'mul(lim(lim(val1, sym_1, sym_1), neg(sym_3), lim(log(val0), val1, neg(sym_3))), avg(abs(val0), log(sym_1)))', 'mul(lim(lim(sym_3, lim(val1, mul(sym_2, sym_2), val0), sub(add(val0, val0), sym_2)), lim(val1, sym_1, sym_1), neg(sym_3)), avg(abs(val0), log(sym_1)))', 'avg(lim(lim(val1, sym_1, sym_1), avg(val0, neg(val1)), val0), val0)', 'avg(lim(val1, lim(abs(val0), abs(val1), neg(sym_1)), neg(sym_3)), mul(sub(abs(val0), val0), log(neg(lim(sym_3, neg(val1), log(sym_3))))))', 'avg(lim(neg(sym_3), val1, val0), mul(sub(abs(val0), val0), log(neg(lim(sym_3, neg(val1), log(sym_3))))))', 'lim(neg(mul(avg(sym_3, sym_3), sym_3)), log(avg(neg(sym_2), val1)), val1)', 'lim(neg(mul(avg(sym_3, sym_3), sym_3)), log(avg(neg(sym_2), val1)), val1)', 'lim(neg(lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))), lim(val0, sym_3, neg(val1)), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(neg(lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))), lim(val0, sym_3, neg(val1)), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(neg(lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))), lim(val0, sym_3, neg(val1)), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(neg(lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))), lim(val0, sym_3, neg(val1)), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(neg(lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))), mul(lim(log(sym_2), abs(sym_3), sym_3), sym_3), lim(log(add(val0, abs(log(sym_1)))), mul(avg(sym_3, sym_3), sym_3), lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))))', 'lim(val1, sub(abs(val0), mul(lim(lim(sym_2, sym_1, sym_2), val0, avg(sym_2, sym_1)), avg(sym_3, sym_3))), avg(abs(val0), log(sym_1)))', 'lim(log(sym_1), neg(avg(val0, log(sym_1))), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(sub(abs(val0), mul(lim(lim(sym_2, sym_1, sym_2), val0, avg(sym_2, sym_1)), avg(sym_3, sym_3))), neg(avg(val0, log(sym_1))), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(sub(abs(val0), mul(lim(lim(sym_2, sym_1, sym_2), val0, avg(sym_2, sym_1)), avg(sym_3, sym_3))), neg(avg(val0, log(sym_1))), lim(mul(avg(sym_3, sym_3), sym_3), log(add(val0, abs(log(sym_1)))), avg(val0, log(sym_1))))', 'lim(mul(sym_3, log(sym_1)), avg(val0, log(sym_1)), lim(log(add(val0, abs(log(sym_1)))), mul(avg(sym_3, sym_3), sym_3), lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))))', 'lim(mul(avg(sym_3, sym_3), sym_3), avg(lim(log(sym_1), add(val0, abs(log(sym_1))), val1), log(avg(neg(val1), val0))), lim(log(add(val0, abs(log(sym_1)))), mul(avg(sym_3, sym_3), sym_3), lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))))', 'lim(mul(avg(sym_3, sym_3), sym_3), avg(lim(log(sym_1), add(val0, abs(log(sym_1))), val1), log(avg(neg(val1), val0))), lim(log(add(val0, abs(log(sym_1)))), mul(avg(sym_3, sym_3), sym_3), lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))))']
[-13.1375, -13.8375, -14.375, -15.175, -14.8625, -15.2, -16.6125, -16.5375, -21.0125, -20.825, -16.6125, -9.3375, -6.3375, -16.825, -15.2125, -16.9, -11.8, -9.25, -9.0125, -7.9625, -8.55, -7.4125, -8.7625, -7.05, -7.175, -7.7125, -7.9, -7.5625, -4.575, -4.4375]


# %%
print ([str(x) for x in best_solutions])
print (str(best_scores))

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 15}

matplotlib.rc('font', **font)



exv = best_scores
exv = [-x for x in exv]
ekv = [i+1 for (i,e) in enumerate(exv)]

plt.title("Training Curve")
plt.xlabel("Generation")
plt.ylabel("Convergence Time (steps)")
plt.tight_layout()
plt.plot(ekv, exv)
plt.savefig("test.svg")
plt.show()

# %%

[str(x) for x in ctrlx.population]

# %%


# %%
['neg(div(abs(sym_2), mul(sym_3, val1)))', 'neg(div(abs(sym_2), mul(sym_3, val1)))', 'neg(div(abs(sym_2), mul(sym_3, val1)))', 'neg(div(abs(sym_2), mul(sym_3, val1)))', 'lim(div(sym_1, sym_3), lim(abs(sym_2), abs(sym_2), val1), sym_2)', 'lim(div(sym_1, sym_3), lim(abs(sym_2), abs(sym_2), val1), sym_2)', 'lim(div(sym_1, sym_3), lim(abs(sym_2), abs(sym_2), mul(sym_3, val1)), sym_2)', 'lim(sym_2, div(sym_1, sym_3), lim(abs(sym_2), abs(sym_2), val1))', 'lim(lim(abs(sym_2), abs(sym_2), mul(sym_3, val1)), div(sym_1, sym_3), sym_2)', 'lim(lim(abs(sym_2), abs(sym_2), mul(sym_3, val1)), div(sym_1, sym_3), sym_2)', 'lim(sym_2, div(sym_1, sym_3), lim(sym_3, abs(sym_2), val1))', 'lim(lim(abs(sym_2), sym_1, abs(sym_2)), div(sym_1, sym_3), sym_2)', 'lim(lim(abs(sym_2), sym_1, abs(sym_2)), div(sym_1, sym_3), sym_2)', 'lim(sym_2, div(sym_1, sym_3), lim(abs(sym_2), sym_1, abs(sym_2)))', 'lim(sym_2, div(sym_1, sym_3), lim(abs(sym_2), sym_1, abs(sym_2)))', 'lim(sym_2, div(sym_1, sym_3), lim(abs(sym_2), sym_1, abs(sym_2)))', 'lim(sym_2, div(sym_1, sym_3), lim(abs(sym_2), sym_1, abs(sym_2)))', 'lim(sym_2, div(sym_1, sym_3), lim(sym_1, abs(sym_2), log(sym_3)))', 'lim(sym_2, div(sym_1, sym_3), lim(sym_1, abs(sym_2), log(sym_3)))', 'lim(sym_2, div(sym_1, sym_3), lim(sym_1, abs(sym_2), log(sym_3)))']
[-17.975, -15.8625, -17.05, -18.5, -14.6, -12.6375, -13.0625, -11.925, -14.025, -14.2375, -14.275, -13.0375, -14.325, -14.6375, -14.3375, -14.875, -14.525, -10.725, -12.0125, -11.6625]

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 15}

matplotlib.rc('font', **font)



exv = best_scores
exv = [-x for x in exv]
ekv = [i+1 for (i,e) in enumerate(exv)]

plt.title("Training Curve")
plt.xlabel("Generation")
plt.ylabel("Convergence Time (steps)")
plt.tight_layout()
plt.plot(ekv, exv)
plt.savefig("test.svg")
plt.show()



# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 15}

matplotlib.rc('font', **font)



ex = best_solutions
ex = [-x for x in ex]
ek = [i+1 for (i,e) in enumerate(ex)]

plt.title("Training Curve")
plt.xlabel("Generation")
plt.ylabel("Convergence Time (steps)")
plt.tight_layout()
plt.plot(ek, ex)
plt.savefig("test.svg")
plt.show()



# %%


# %%


# %%


# %%


import numpy as np
        
def normalise(pos):
    len = math.sqrt(pos[0]**2+pos[1]**2)
    return tuple(a / len for a in pos)

def random_vector(d):
    return tuple(np.random.normal() for i in range(d))

def gen_neignbour(parent_pos, step_f):
    direction = normalise(random_vector(2))
    direction_scaled = [x * step_f for x in direction]
    return tuple(sum(x) for x in zip(direction_scaled, parent_pos))

def new_pos_from_pos(parent_pos, step_f):
    neighbours = []
    for i in range(10): 
        neighbours.append(gen_neignbour(parent_pos, step_f))

    neighbours_scores = [himmelblaup(x[0], x[1]) for x in neighbours]
    index_min = min(range(len(neighbours_scores)), key=neighbours_scores.__getitem__)
    return neighbours[index_min]


val0 = 0
val1 = 1

def monster(sym_1, sym_2, sym_3):
    sym_1 = -sym_1
    sym_2 = -sym_2
    sym_3 = -sym_3
    step = lim(mul(avg(sym_3, sym_3), sym_3), avg(lim(log(sym_1), add(val0, abs(log(sym_1))), val1), log(avg(neg(val1), val0))), lim(log(add(val0, abs(log(sym_1)))), mul(avg(sym_3, sym_3), sym_3), lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))))
    return step

# Get point sequence for constant



sequence = []
position = (0,0)
current_score = himmelblaup(*position)


gen = 0
sequence = []
while current_score > 0.1:
    gen = gen + 1
    sequence.append(position)
    step = 0.15
    position = new_pos_from_pos(position, step)
    new_score = himmelblaup(*position)
    p_p_best = p_best
    p_best = current_score
    current_score = new_score


sequence_monster = []
position = (0,0)
current_score = himmelblaup(*position)
p_best = 0.5
p_p_best = current_score
genx = 0
while current_score > 0.1:
    genx = genx + 1
    sequence_monster.append(position)
    step = monster(current_score, p_best, p_p_best)
    position = new_pos_from_pos(position, step)
    new_score = himmelblaup(*position)
    p_p_best = p_best
    p_best = current_score
    current_score = new_score


# %%
gen

# %%
sequence_monster

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {'size'   : 18}

matplotlib.rc('font', **font)
def himmelblaup(x:float, y:float)-> float:
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = himmelblaup(X, Y)

plt.figure(figsize=(8, 6))
cs = plt.contour(X, Y, Z, levels=100, cmap='Spectral',
                 norm=colors.Normalize(vmin=Z.min(), vmax=Z.max()), alpha=0.4)
plt.colorbar(cs)

plt.scatter(0,0, s=[300], marker="o", label="origin")


def print_lines(points):
    xs = [x[0] for x in points]
    ys = [x[1] for x in points]
    plt.plot(xs, ys)
        
print_lines(sequence)
print_lines(sequence_monster)

plt.legend()

plt.show()


# %%


import numpy as np
        
def normalise(pos):
    len = math.sqrt(pos[0]**2+pos[1]**2)
    return tuple(a / len for a in pos)

def random_vector(d):
    return tuple(np.random.normal() for i in range(d))

def gen_neignbour(parent_pos, step_f):
    direction = normalise(random_vector(2))
    direction_scaled = [x * step_f for x in direction]
    return tuple(sum(x) for x in zip(direction_scaled, parent_pos))

def new_pos_from_pos(parent_pos, step_f, use_function = himmelblaup):
    neighbours = []
    for i in range(10): 
        neighbours.append(gen_neignbour(parent_pos, step_f))

    neighbours_scores = [use_function(x[0], x[1]) for x in neighbours]
    index_min = min(range(len(neighbours_scores)), key=neighbours_scores.__getitem__)
    return neighbours[index_min]


val0 = 0
val1 = 1

def monster(sym_1, sym_2, sym_3):
    sym_1 = -sym_1
    sym_2 = -sym_2
    sym_3 = -sym_3
    step = lim(mul(avg(sym_3, sym_3), sym_3), avg(lim(log(sym_1), add(val0, abs(log(sym_1))), val1), log(avg(neg(val1), val0))), lim(log(add(val0, abs(log(sym_1)))), mul(avg(sym_3, sym_3), sym_3), lim(lim(sym_3, sym_3, sym_2), sym_2, lim(sym_2, val1, val1))))
    return step

# Get point sequence for constant

origin = (3,-3.2)

sequence = []
position = origin
current_score = himmelblaup(*position)


gen = 0
sequence = []
while current_score > 0.1:
    gen = gen + 1
    sequence.append(position)
    step = 0.15
    position = new_pos_from_pos(position, step)
    new_score = himmelblaup(*position)
    p_p_best = p_best
    p_best = current_score
    current_score = new_score


sequence_monster = []
position = origin
current_score = himmelblaup(*position)
p_best = 0.5
p_p_best = current_score
genx = 0
while current_score > 0.1:
    genx = genx + 1
    sequence_monster.append(position)
    step = monster(current_score, p_best, p_p_best)
    position = new_pos_from_pos(position, step)
    new_score = himmelblaup(*position)
    p_p_best = p_best
    p_best = current_score
    current_score = new_score


    import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {'size'   : 18}

matplotlib.rc('font', **font)
def himmelblaup(x:float, y:float)-> float:
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = himmelblaup(X, Y)

plt.figure(figsize=(8, 6))
cs = plt.contour(X, Y, Z, levels=100, cmap='Spectral',
                 norm=colors.Normalize(vmin=Z.min(), vmax=Z.max()), alpha=0.4)
plt.colorbar(cs)

plt.scatter(0,0, s=[300], marker="o", label="origin")


def print_lines(points):
    xs = [x[0] for x in points]
    ys = [x[1] for x in points]
    plt.plot(xs, ys)
        
print_lines(sequence)
print_lines(sequence_monster)

plt.legend()

plt.show()


# %% [markdown]
# Plot monster. X Y Z: z is the output, x is the delta of past , y is the 

# %%
origin = 10
for delta in range(-40, 40):
    print (monster(origin, origin+1, origin+delta))

# %%

def new_pos_from_pos(parent_pos, step_f, use_function = himmelblaup):
    neighbours = []
    for i in range(10): 
        neighbours.append(gen_neignbour(parent_pos, step_f))

    neighbours_scores = [use_function(x[0], x[1]) for x in neighbours]
    index_min = min(range(len(neighbours_scores)), key=neighbours_scores.__getitem__)
    return neighbours[index_min]

import random
class MovingHimmelblau:
    def __init__(self):
        self.x_shift = 0
        self.y_shift = 0

    def __call__(self, x, y):
        return MovingHimmelblau.himmelblau(x+self.x_shift, y+self.y_shift)
    
    def shift(self):
        self.x_shift += random.random()
        self.x_shift += random.random()

    @staticmethod    
    def himmelblau(x:float, y:float)-> float:
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    
    
moving_himmelblau = MovingHimmelblau()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {'size'   : 18}

matplotlib.rc('font', **font)

static_pos = (0,0)
dynamic_pos = (0,0)


current_position_c = (0,0)
current_position_f = (0,0)

current_score = moving_himmelblau(*current_position_f)


for i in range(10):
    previous_position_c = current_position_c
    previous_position_f = current_position_f

    # compute next constant position
    step = 0.15
    current_position_c = new_pos_from_pos(previous_position_c, step, moving_himmelblau)

    plt.plot([previous_position_c[0], current_position_c[0]], [previous_position_c[1], current_position_c[1]])

    # compute the functional position
    p_best = 0.5
    p_p_best = current_score
    step = monster(current_score, p_best, p_p_best)
    current_position_f = new_pos_from_pos(previous_position_f, step, moving_himmelblau)

    new_score = moving_himmelblau(*current_position_f)
    p_p_best = p_best
    p_best = current_score
    current_score = new_score
    plt.plot([previous_position_f[0], current_position_f[0]], [previous_position_f[1], current_position_f[1]])
    

    X = np.arange(-5, 5, 0.1)
    Y = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = moving_himmelblau(X, Y)

    plt.figure(figsize=(8, 6))
    cs = plt.contour(X, Y, Z, levels=100, cmap='Spectral',
                    norm=colors.Normalize(vmin=Z.min(), vmax=Z.max()), alpha=0.4)
    plt.colorbar(cs)

    plt.scatter(0,0, s=[300], marker="o", label="origin")
    moving_himmelblau.shift()

plt.show()








# %%
print ([str(x) for x in best_solutions])
print (str(best_scores))


