from typing import Generic
from typing import TypeVar
from typing import Iterator
from typing import Iterable
from typing import Tuple
from typing import Optional
from typing import Self
from typing import List
from random import shuffle
from random import sample
import itertools

from abc import ABC
from abc import abstractmethod

R = TypeVar('R')

class Genome(ABC, Generic[R]):
    """!A solution
        A genotypical representation of a solution that specifies the capabilities a solution must have in order to work with evolutionary operators.
    """
    def __init__(self) -> None:
        # The genome
        self.score: Optional[float] = None
    
    @abstractmethod
    def copy(self) -> Self: ...
    """!Copy the solution.
        Copy the solution, so that changes made on the copy do not affect the original genome. The implementation decides if all components must be copied.
    """


T = TypeVar('T', bound=Genome)

class AbstractCollection(ABC, Generic[R]):
    """!An abstract collection of things. Provides the behaviour of other collections. Improving it will surely lead to improvement in overall performance of the model.
    """
    def __init__(self, *args: R):
        self._solutions = list(args)
        self._index = 0

    def __len__(self) -> int:
        return len(self._solutions)
    
    def __getitem__(self, key: int) -> R:
        return self._solutions[key]
    
    def __setitem__(self, key: int, value: R) -> None:
        self._solutions[key] = value

    def __delitem__(self, key: int) -> None:
        del self._solutions[key]

    def __str__(self) -> str:
        return str(list(map(str, self._solutions)))
    
    def __iter__(self) -> Iterator[R]:
        self._index = 0
        return self

    def __next__(self) -> R:
        if self._index < len(self._solutions):
            old_index = self._index
            self._index = self._index + 1
            return self._solutions[old_index]
        else:
            raise StopIteration
        
    def append(self, value: R) -> None:
        self._solutions.append(value)

    def extend(self, value: Iterable[R]) -> None:
        self._solutions = list(itertools.chain(self._solutions, value))

    def draw(self, key: int) -> R:
        a : R = self[key]
        del self[key]
        return a

# P = TypeVar('P', bound=Tuple[Genome, ...])
class GenomePool(AbstractCollection[Tuple[T, ...]]):
    """!A collection of tuple  of parents.
        A collection of tuples of solutions. Passed from the parent selector to the variator. Its arity is not enforced.
    """
    def __init__(self, arity: int, *args: Tuple[T, ...]):
        super().__init__(*args)
        self.arity = arity

from core.globals import report
from core.globals import LogLevel
class Population(AbstractCollection[T]):
    """!A collection of solutions.
        A population of many genomes.
    """
    def __init__(self, *args: T):
        super().__init__(*args)

    def copy(self) -> Self:
        return self.__class__(*[x.copy() for x in self._solutions])
    
    def sort(self: Self):
        self._solutions.sort(reverse=True, key=lambda x : x.score if x.score is not None else 0)
        
    

