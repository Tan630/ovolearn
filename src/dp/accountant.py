from __future__ import annotations
import typing
from typing import Generic, TypeVar, Sequence, Optional
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Self
from math import e, exp
from numbers import Number

I = TypeVar("I")
O = TypeVar("O")

class Budget(ABC):
    def __init__(self, name: str, parameters: Dict[str, float]):
        self.name: str = name
        self.parameters: Dict[str, float] = parameters

    # def __getattr__(self, param: str):
    #     if param in self.__dict__:
    #         return self.param
    #     elif param in self.parameters.keys():
    #         return self.parameters[param]
    #     else:
    #         raise Exception("The required privacy parameter is not implemented")
    
    def __str__(self)-> str:
        return f"Params: {self.parameters}"
    
    @abstractmethod
    def copy(self)-> Self:
        pass
    
    @abstractmethod
    def consume(self, other: Self)-> None:
        pass

    def clear(self)-> None:
        for key in self.parameters:
            self.parameters[key] = 0

    def exceeds(self, other: Self)-> bool:
        am_overbudget = False
        for key in self.parameters:
            if self.parameters[key] > other.parameters[key]:
                am_overbudget = True
        return am_overbudget
    
    def __getitem__(self, key):
        return self.parameters.__getitem__(key)
    
class PureBudget(Budget):
    def __init__(self, epsilon):
        super().__init__("pure", {"epsilon": epsilon})

    def copy(self)-> PureBudget:
        return PureBudget(self.parameters["epsilon"])
    
    def consume(self, other: Self)-> None:
        # This implementation does not consider adding delta
        if self.name != other.name:
            raise Exception("Budget names do not match!")
        elif set(self.parameters.keys()) != set(other.parameters.keys()):
            raise Exception("Budget parameters do not match!")
        else:
            for key in self.parameters.keys():
                self.parameters[key] += other.parameters[key]

    
from math import sqrt, log
class ApproximateBudget(Budget):
    def __init__(self, epsilon, delta, deltap):
        super().__init__("approximate", {"epsilon": epsilon, "delta": delta})
        self.n = 0
        self.epsilon = epsilon # this is the epsilon of a single algorithm
        self.delta = delta # this is the delta of a single algorithm
        self.deltap = deltap

    def copy(self)-> ApproximateBudget:
        new_bud = ApproximateBudget(self.parameters["epsilon"], self.parameters["delta"], self.deltap)
        new_bud.n = self.n
        new_bud.epsilon = self.epsilon
        new_bud.delta = self.delta
        return new_bud
    
    
    def consume(self, other: Self)-> None:
        # This implementation does not consider adding delta
        if (other.epsilon != self.epsilon
            or other.delta != self.delta
            or other.deltap != self.deltap):
            raise Exception("This is too advanced")
        

        
        n = self.n + 1
        new_delta = n * self.delta + self.deltap
        print (new_delta)
        new_epsilon = 2 * n * log(1 / self.deltap) * self.epsilon + n * self.epsilon * (e ** self.epsilon - 1)
        print(new_epsilon)
        self.n = n
        self.parameters["epsilon"] = new_epsilon
        self.parameters["delta"] = new_delta

        

        if self.name != other.name:
            raise Exception("Budget names do not match!")
        elif set(self.parameters.keys()) != set(other.parameters.keys()):
            raise Exception("Budget parameters do not match!")
        else:
            for key in self.parameters.keys():
                self.parameters[key] += other.parameters[key]

class BudgetCollection:
    def __init__(self, budgets: Sequence[Budget]):
        self.budgets: Dict[str, Budget]
        self.budgets = {}
        for bud in budgets:
            self.budgets[bud.name] = bud
    
    # def __getattr__(self, name: str):
    #     if hasattr(self, name):
    #         return self.name
    #     else:
    #         return self.budgets[name]
        
    def __iter__(self):
        return self.budgets.__iter__()
    
    def __getitem__(self, key):
        return self.budgets.__getitem__(key)
    
    def values(self)-> Sequence[Budget]:
        return list(self.budgets.values())
            

class Algorithm(ABC, Generic[I, O]):
    @abstractmethod
    def __init__(self):
        pass
        
    @abstractmethod
    def validate_input(self, input: I)-> bool:
        pass

    @abstractmethod
    def validate_output(self, output: O)-> bool:
        pass

    @abstractmethod
    def __call__(self, input: I, **kwargs)-> O:
        pass
        
    
    @property
    @abstractmethod
    def budgets(self)-> BudgetCollection:
        pass

    def guaded_call(self, input: I, **kwargs)-> O:
        if (not self.validate_input(input)):
            raise Exception("Input check fails")
        output = self(input, **kwargs)
        if (not self.validate_output(output)):
            raise Exception("Output check fails")
        else:
            return output
        
    def __str__(self)-> str:
        return str(self.budgets)



class Accountant:
    def __init__(self, algorithm: Algorithm[I,O], constraints: BudgetCollection):
        self.algorithm = algorithm
        self.constraints = constraints
        self.budgets = [bud.copy() for bud in self.algorithm.budgets.values()]
        self.truncate = False
        
        algorithm_budget_names = [bud.name for bud in self.algorithm.budgets.values()]
        constraint_budget_names = [bud.name for bud in self.budgets]

        if not (set(algorithm_budget_names) == set(constraint_budget_names)
                and len(algorithm_budget_names) == len(algorithm_budget_names)):
            raise Exception("Something is wrong with the budgets.")
        
    def __call__(self, input: I, **kwargs)-> Optional[O]:
        if not self.truncate:
            # Check if I have exceeded budgets
            
            for bud in self.budgets:
                



                if bud.exceeds(self.constraints[bud.name]):
                    self.truncate = True
                    
                else:
                    bud.consume(self.algorithm.budgets[bud.name])
            if not self.truncate:
                return self.algorithm.guaded_call(input, **kwargs)
        return None

    def reset(self):
        self.budgets = [bud.copy() for bud in self.algorithm.budgets.values()]
        self.truncate = False

def get_budget_from_list_by_name(budgets: Sequence[Budget], name: str)-> Budget:
    for bud in budgets:
        if bud.name == name:
            return bud
    else:
        raise Exception("get_budget_from_list_by_name fails")

from numpy.random import laplace
from numpy.random import randn
class PureLaplace(Algorithm[float, float]):
    def __init__(self, epsilon: float, sensitivity: float):
        self._budgets: BudgetCollection = BudgetCollection([PureBudget(epsilon)])
        self.sensitivity = sensitivity
        
    def validate_input(self, input: float)-> bool:
        return isinstance(input, Number)

    def validate_output(self, output: float)-> bool:
        return isinstance(output, Number)

    def __call__(self, input: float, **kwargs)-> float:
        return input + laplace_pdf(0, self.sensitivity / self.budgets["pure"]["epsilon"])
        
    @property
    def budgets(self)-> BudgetCollection:
        return self._budgets
    
import random
class BSC(Algorithm[int, int]):
    def __init__(self, epsilon: float, sensitivity = None):
        self._budgets: BudgetCollection = BudgetCollection([PureBudget(epsilon)])
        if (sensitivity is not None):
            raise Exception("BSC does not have sensitivity!")
        if (not self.validate_epsilon()):
            raise Exception("BSC epsilon out of range!")
        self.sensitivity = sensitivity
        
    def validate_input(self, input: int)-> bool:
        return (input == 0) or (input == 1)

    def validate_output(self, output: int)-> bool:
        return (output == 0) or (output == 1)
    
    def validate_epsilon(self)-> bool:
        return (self.budgets["pure"]["epsilon"] >= 0 and self.budgets["pure"]["epsilon"] <= 1)

    def __call__(self, input: int, **kwargs)-> int:
        flip_probability = random.random()
        return (input + 1) & 1 if flip_probability <= self.budgets["pure"]["epsilon"] else input
        
    @property
    def budgets(self)-> BudgetCollection:
        return self._budgets

from typing import Set, Callable, TypeVar

E = TypeVar("E")


from random import choices
class PureExponential(Algorithm[Sequence[E], E]):
    def __init__(self, epsilon: float, evaluator: Callable[[E], float], u_sensitivity: float):
        self._budgets: BudgetCollection = BudgetCollection([PureBudget(epsilon)])
        self.evaluator = evaluator
        self.u_sensitivity = u_sensitivity
        
        
    def validate_input(self, items: Sequence[E])-> bool:
        return True #what

    def validate_output(self, output: E)-> bool:
        return True # limitation of the framework, validate_output cannot access input

    def __call__(self, input: Sequence[E], **kwargs)-> E:
        weights = tuple (exp(self.budgets["pure"]["epsilon"] * self.evaluator(x)) / (2 * self.u_sensitivity)
                   for x in input)
    
        result: E = choices(input, weights, k=1)[0]
        return result
        
    @property
    def budgets(self)-> BudgetCollection:
        return self._budgets

    
from random import gauss
class ApproximateGaussian(Algorithm[float, float]):
    def __init__(self, epsilon: float, delta: float, deltap: float, sensitivity: float):
        self._budgets: BudgetCollection = BudgetCollection([ApproximateBudget(epsilon, delta, deltap)])
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
    def validate_input(self, input: float)-> bool:
        return isinstance(input, Number)

    def validate_output(self, output: float)-> bool:
        return isinstance(output, Number)

    def __call__(self, input: float, **kwargs)-> float:
        variance = 2 * (self.sensitivity)**2 * log(1 / self.delta) / self.epsilon**2
        print(f"------------ {variance}")
        return input + sqrt(variance) * randn()
        
    @property
    def budgets(self)-> BudgetCollection:
        return self._budgets
    

def laplace_pdf(mu, b)-> float:
        return laplace(mu, b)



# test lap
# lap1 = PureLaplace(0.5, 2)
# lap_bud_3 = Accountant(lap1, BudgetCollection([Budget("pure", {"epsilon": 3})]))
# lap_bud_3_2 = Accountant(lap1, BudgetCollection([Budget("pure", {"epsilon": 3})]))
# l1: List[float] = []
# l2: List[float] = []
# for i in range (10):
#     res1: Optional[float] = lap_bud_3(90)
#     res2: Optional[float] = lap_bud_3_2(91)
#     if (res1 is not None and res2 is not None):
#         l1.append(res1)
#         l2.append(res2)

# print(sum(l1) / len(l1))
# print(sum(l2) / len(l2))
# import matplotlib.pyplot as plt
# plt.plot(l1)
# plt.plot(l2)
# plt.show()

# bsc = BSC(0.5)
# bsc_1 = Accountant(bsc, BudgetCollection([PureBudget(3)]))
# bsc_2 = Accountant(bsc, BudgetCollection([PureBudget(3)]))
# l1: List[float] = []
# l2: List[float] = []
# for i in range (10):
#     res1: Optional[float] = bsc_1(1)
#     res2: Optional[float] = bsc_2(0)
#     if (res1 is not None and res2 is not None):
#         l1.append(res1)
#         l2.append(res2)

# print(sum(l1) / len(l1))
# print(sum(l2) / len(l2))
# import matplotlib.pyplot as plt
# plt.plot(l1)
# plt.plot(l2)
# plt.show()



# gaus = ApproximateGaussian(0.1, 0.001, 0.0001, 1)
# gaus_1 = Accountant(gaus, BudgetCollection([ApproximateBudget(3, 1, 1)]))
# gaus_2 = Accountant(gaus, BudgetCollection([ApproximateBudget(3, 1, 1)]))
# l1: List[float] = []
# l2: List[float] = []
# for i in range (10):
#     res1: Optional[float] = gaus_1(90)
#     res2: Optional[float] = gaus_1(91)
#     if (res1 is not None and res2 is not None):
#         l1.append(res1)
#         l2.append(res2)

# print(sum(l1) / len(l1))
# print(sum(l2) / len(l2))
# import matplotlib.pyplot as plt
# plt.plot(l1)
# plt.plot(l2)
# plt.show()


expn: PureExponential[List[float]] = PureExponential(0.4, lambda x : sum(x), 1)
expn_1 = Accountant(expn, BudgetCollection([PureBudget(99)]))
expn_2 = Accountant(expn, BudgetCollection([PureBudget(99)]))
l1: List[float] = []
l2: List[float] = []
for i in range (10):
    res1: Optional[List[float]] = expn_1([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]])
    res2: Optional[List[float]] = expn_1([[1.1], [2.2], [2.3], [4.4], [8.5], [1.6], [0.7]])
    if (res1 is not None and res2 is not None):
        l1.append(res1[0])
        l2.append(res2[0])


import matplotlib.pyplot as plt
plt.plot(l1)
plt.plot(l2)
plt.show()