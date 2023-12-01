from __future__ import annotations
import abc
import typing

from typing import Optional
from typing import Tuple
from typing import Dict
from typing import Generic
from typing import Type
from dataclasses import dataclass

from typing import Union
from typing import List
from typing import Any
from typing import Self
from typing import Callable
from inspect import signature

from core.globals import LogLevel
from core.globals import report
from core.variator import Variator
from core.selector import ElitistSimpleSelector
from core.evaluator import Evaluator
from core.controller import Controller
from core.population import Population
from core.population import Genome

import gymnasium as gym

from random import choice


T = typing.TypeVar("T")

class ArityMismatch(Exception):

    def __init__(self, expr, expected:Optional[int], given: Optional[int]):
        super().__init__(f"Arity mismatch: while evaluating {str(expr)}, "
                         f"{expected} arguments are expected, while "
                         f"{given} are given")

def get_arity(fun: Any) -> int:
    if (callable(fun)):
        return len(signature(fun).parameters)                 
    else:
        return 0

class Expression(abc.ABC, typing.Generic[T]):
    def __init__(self, function: T | typing.Callable[..., T], *children: Expression[T]):
        self._function = function
        self.children : List[Expression[T]] = list(children)
    
    def evaluate(self) -> T:
        expected_arity = get_arity(self._function)
        children_arity = len(self.children)
        results = tuple(x.evaluate() for x in self.children)
        if callable(self._function):
            if (expected_arity == children_arity):
                return self._function(*results)
            else:
                raise ArityMismatch(self._function, expected_arity, children_arity)
        elif expected_arity == 0 and children_arity == 0:
            return self._function
        else:
            raise ArityMismatch(self._function, expected_arity, children_arity)
        
        return self._function(*results)
    
    def copy(self) -> Self:
        children_copies : Tuple[Expression[T], ...] = tuple(x.copy() for x in self.children)
        return self.__class__(self._function, *children_copies)
    
    # Beware this function leans to the left subtree
    def nodes(self) -> List[Expression[T]]:
        return [self] + list(self.children)
    
    def __str__(self) -> str:
        children_names : str = ''
        for c in self.children:
            children_names = children_names +" "+ str(c)
        my_name: str = ""
        if callable(self._function):
            my_name = self._function.__name__
        else:
            my_name = str(self._function)
        return (f"{str(my_name)} ({children_names})")

    __copy__ = copy
    __deepcopy__ = copy


class ExpressionFactory(typing.Generic[T]):
    def __init__(self, functions: Tuple[Callable[..., T], ...]):

        # Build a pool of functions, as a dictionary where each index corresponds to an arity, and each 
        #   value is a list of functions (or terminals!) with that arity.
        self.function_pool: Dict[int, List[Callable[..., T]]] = {}
        for fun in functions:
            arity: int = get_arity(fun)
            if arity in self.function_pool:
                self.function_pool[arity].append(fun)
            else:
                self.function_pool[arity] = [fun]

        self.depth_cap: int = 0
        self.budget_cap: int = 0
        self.depth = 0
        self.budget_used = 0

    def build(self, depth: int, budget: int) -> Expression:
        self.budget_cap = budget
        self.budget_used = 0

        target_function = self.poll_function()
        arity = get_arity(target_function)

        children : List [Expression]= []
        for i in range(0, arity):
            children.append(self._build_recurse(depth-1))

        root = Expression(target_function, *children)
        if (self.budget_used < self.budget_cap):
            report(LogLevel.TRC, f"Tree built below budget! used: {self.budget_used}, cap: {self.budget_cap}")
        else:
            report(LogLevel.TRC, f"Tree built at budget. used: {self.budget_used - 1}, cap: {self.budget_cap}")
        return root
        

    def _build_recurse(self, depth_left: int) -> Expression:
        if (depth_left < 1 or self.over_budget()):
            target_function = self.poll_arity(0)
            return Expression(target_function)
        else:
            target_function = self.poll_function()
            arity = get_arity(target_function)
            children : List [Expression]= []
            for i in range(0, arity):
                children.append(self._build_recurse(depth_left-1))
            base = Expression[T](target_function, *children)
            return base

    def over_budget(self) -> bool:
        return self.budget_used > self.budget_cap

    def cost_budget(self) -> None:
        self.budget_used = self.budget_used + 1
        # report(LogLevel.TRC, f"budget used: {self.budget_used - 1} -> {self.budget_used}")
    
    def poll_function(self) -> Callable[..., T]:
        self.cost_budget()
        return choice(choice(self.function_pool))

    def poll_arity(self, arity: int) -> Callable[..., T]:
        return choice(self.function_pool[arity])

import math



class BadSymbolError(Exception):
    def __init__(self, name: str):
        super().__init__(f"The symbol {name} is used but not assigned.")


class Symbol(typing.Generic[T]):
    def __init__(self, name: str = "default_symbol_name", value: Optional[T] = None):
        self.value : Optional[T] = value
        self.__name__ : str = name

    def __call__(self) -> T:
        if (self.value is None):
            raise BadSymbolError(self.__name__)
        return self.value
    
    def assign(self, val: T) -> None:
        self.value = val
    
    def __str__(self)-> str:
        return self.__name__

# a = Symbol[float](1, "var_a")
# b = Symbol[float](2, "var_b")
# c = Symbol[float](3, 'var_c')


# exprfactory = ExpressionFactory[float]((math.sin, math.sqrt, math.pow, add, sub, a, b, c, 4))

# e = exprfactory.build(10, 0)

# print (e.evaluate())

# a.value = 4

# print (e.evaluate())

class ProgramFactory(Generic[T]):
    def __init__(self, functions: Tuple[Callable[..., T], ...], arity):
        self.arity = arity
        self._symbol_count = 0 # only used to keep track of symbol names. Does not relate to arity
        self.symbol_deposit : List[Symbol]= []
        
        for i in range(0, arity):
            self.deposit_symbol(self.next_symbol())

        self.exprfactory = ExpressionFactory[T](functions + tuple(self.symbol_deposit))

    def next_symbol_name(self) -> str:
        self._symbol_count = self._symbol_count + 1
        return str(self._symbol_count)

    def deposit_symbol(self, s: Symbol[T]) -> None:
        self.symbol_deposit.append(s)
        
    def next_symbol(self) -> Symbol[T]:
        return Symbol("sym_" + self.next_symbol_name())
    
    def build(self, depth: int, budget: int) -> Program:
        return Program(self.exprfactory.build(depth, budget), self.symbol_deposit)
    
    
    

# Note that programs from the same factory share the same set of argument values.
# This should be desirable - and well hidden - if the object is only shared by one thread.
# I'm not prepared to deal with concurrency.

class ProgramArityMismatchError(Exception):

    def __init__(self, expected:Optional[int], given: Optional[int]):
        super().__init__(f"The program is expecting {expected} arguments, only {given} are given.")
        

class Program(Genome[T]):
    """
    
    """
    def __init__(self, expr: Expression, symbols: List[Symbol]):
        super().__init__()
        self.expr = expr
        self.symbols = symbols
    
    def evaluate(self, *args: T) -> T:
        if (len(args) != len(self.symbols)):
            raise ProgramArityMismatchError(len(self.symbols), len(args))
        self.__class__._assign_values(self.symbols, args)
        return self.expr.evaluate()

    @staticmethod
    def _assign_values(symbols: List[Symbol[T]], values: Tuple[T, ...]) -> None:
        # This is exceptionally unpythonic
        for i in range (0, len(symbols)):
            symbols[i].assign(values[i])

    def __str__(self) -> str:
        return str(self.expr)
    
    # Warning! Breaking design patterns.
    def nodes(self) -> List[Expression[T]]:
        return self.expr.nodes()
    
    def copy(self) -> Self:
        return self.__class__(self.expr.copy(), self.symbols)

def sin(x: float) -> float:
    return math.sin(x)

def cos(x: float) -> float:
    return math.cos(x)

def tan(x: float) -> float:
    return math.tan(x)

def add (x:float, y:float):
    return x + y

def sub (x:float, y:float):
    return x - y

def mul (x:float, y:float):
    return x * y

def div (x:float, y:float):
    if y == 0:
        return 1
    return x / y

def lim(x: float, max_val:float, min_val:float) -> float:
    return max(min(max_val, x), min_val)


import random

from typing import Iterable
class ProgramCrossoverVariator(Variator[Program[T]]):
    def vary(self, parents: Tuple[Program[T], ...]) -> Tuple[Program[T], ...]:
        root1: Program = parents[0].copy()
        root2: Program = parents[1].copy()
        root1.score = None
        root2.score = None

        expression_nodes_from_root_1 = root1.nodes()
        expression_nodes_from_root_2 = root2.nodes()

        # this relies on a very ad-hoc implementation that provides access to expressions though the program. 
        # This should not be possible. But I am tired and cannot think of better ways to do it.
        expression_internal_nodes_from_root_1 = [x for x in expression_nodes_from_root_1 if len(x.children) > 0]
        expression_internal_nodes_from_root_2 = [x for x in expression_nodes_from_root_2 if len(x.children) > 0]

        if (len(expression_internal_nodes_from_root_1) >= 1 and len(expression_internal_nodes_from_root_2) >= 1):
            expression_node_from_root_1_to_swap = random.choice(expression_internal_nodes_from_root_1)
            expression_node_from_root_2_to_swap = random.choice(expression_internal_nodes_from_root_2)
            self.__class__.swap_children(expression_node_from_root_1_to_swap, expression_node_from_root_2_to_swap)

        return (root1, root2, root1.copy(), root2.copy())

    @staticmethod
    def swap_children(expr1: Expression[T], expr2: Expression[T]) -> None:
        child_nodes = list(expr1.children + expr2.children)
        random.shuffle(child_nodes)

        for i in range(0,len(expr1.children)):
            expr1.children[i] = child_nodes[i].copy()

        for i in range(-1,-(len(expr2.children) + 1), -1):
            expr2.children[i] = child_nodes[i].copy()





class GymEvaluator(Evaluator[Program[float]]):
    def __init__(self, env, wrapper: Callable[[float], float], step_count: int, score_wrapper: Callable[[float], float] = lambda x : x):
        super().__init__()
        self.env = env
        self.wrapper = wrapper
        self.step_count = step_count
        self.score_wrapper = score_wrapper

    def evaluate(self, s1: Program[float]) -> float:
        score = GymEvaluator.evaluate_episode(s1, self.env, self.wrapper, self.step_count) 
        return self.score_wrapper(score)

    @staticmethod
    def evaluate_episode(s1: Program[float], env, wrapper: Callable[[float], float], step_count: int) -> float:
        score = 0.
        for i in range(0, step_count):
            score = score + GymEvaluator.evaluate_step(s1, env, wrapper)
        return score / step_count

    @staticmethod
    def evaluate_step(s1: Program[float], env, wrapper: Callable[[float], float]) -> float:
        step_result = env.reset()
        score = 0.
        # hard coded - an episode consists of 10 evaluations.
        for i in range(0, 10):
            step_result = env.step(wrapper(s1.evaluate(*step_result[0]))) #type: ignore
            if (step_result[2]):
                break
            score = score + step_result[1] #type: ignore
        return score

# ########## Begin setup :) ########## #

# Size of the population. Affects the size of the initial initial population, also enforced by selectors.
pop_size = 100

# Depth constraint of the expression tree
tree_depth = 5
# Node budget of the expression tree
node_budget = 8

# The number of episodes for each evaluation. The actual score should be the mean of these scores.
# The length of each episode is hard-coded to be 10 (see `evaluate_step`)
iter_bound = 100


# Build the population of ternary programs. The arity (4) should match the size of the observation space (4 for cartpole)
progf = ProgramFactory((add, sub, mul, div, sin, cos, mul, div, lim), 4)

pops: Population[Program[float]] = Population()
for i in range(0, pop_size):
    pops.append(progf.build(tree_depth, node_budget))

# Prepare the variator
variator = ProgramCrossoverVariator[Program[float]](arity = 2, coarity = 4, checked = True)

# The evaluaor is ready. Feed the custom wrapper and the environment to GymEvaluator,
#   which is ... or should able to handle all classical control problems.
def pendulum_wrapper(f: float):
    return [max(min(2, f), -2)]

def cartpole_wrapper(f: float) -> float:
    return int(max(min(1, f), 0))

eval = gym.make('CartPole-v1')
evaluator = GymEvaluator(eval, cartpole_wrapper, iter_bound, score_wrapper = lambda x : -x)

# Selector on standby
import gymnasium as gym
sel = ElitistSimpleSelector[Program[float]](coarity = 2, budget = pop_size)


ctrl = Controller[Program[float]](
    population = pops,
    evaluator = evaluator,
    parent_selector = sel,
    variator = variator,
    survivor_selector = sel
)

best_solutions: List[Program] = []
best_scores: List[Program] = []

def score_keeper(best_scores, best_solutions, c: Controller[Program[T]]):
    best_solutions = best_solutions.append(c.population[0])
    best_scores = best_scores.append(c.population[0].score)

from functools import partial
for i in range(0, 20):
    ctrl.step(partial(score_keeper, best_scores, best_solutions))


print ([str(x) for x in best_solutions])
print (str(best_scores))


 