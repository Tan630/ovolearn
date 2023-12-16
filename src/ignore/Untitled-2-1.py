# %%
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


# %%


# %%

from evolvables.planewalker import score_function
import math
import numpy as np

# %% [markdown]
# Import

# %%


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
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    


sub_episode_bound = 80
sub_step_bound = 60


d = {}
for i in np.arange(0, 1, 0.01):
    d[i] = score_function(himmelblau, lambda x,y,z: i, 1, (0,0), sub_episode_bound, sub_step_bound)

# %%
d

# %% [markdown]
# Instantiate controller

# %%
print (d)

# %%
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 15}

matplotlib.rc('font', **font)

plt.title("Convergence Time vs. Step Size")
plt.xlabel("Step Size")
plt.ylabel("Convergence Time (steps)")
plt.tight_layout()
plt.plot(d.keys(), [-x for x in d.values()])
plt.savefig("test.svg")
plt.show()


# %%
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 15}

matplotlib.rc('font', **font)

plt.title("Convergence Time vs. Step Size")
plt.xlabel("Step Size")
plt.ylabel("Convergence Time (steps)")
plt.tight_layout()
plt.plot(d.keys(), [-x for x in d.values()])
plt.scatter([0.25],[12.5] , label="Best Constant", s=[90])
plt.scatter([0.25],[7.475], s=[90], marker="v", color="black")
plt.scatter([0.25],[10.975], s=[90], marker="^", color="black")
plt.scatter([0.25],[5.975], label="Best Program", s=[90])
plt.arrow(0.25, 11.5, 0, -3.525)
plt.legend(loc='lower right')
plt.savefig("test.svg")
plt.show()


# %% [markdown]
# Plot everything

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# %%

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 18}

matplotlib.rc('font', **font)
def himmelblaup(x:float, y:float)-> float:
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = himmelblaup(X, Y)
#Z = rosenbrock(X, Y)

plt.figure(figsize=(8, 6))
cs = plt.contour(X, Y, Z, levels=100, cmap='Spectral',
                 norm=colors.Normalize(vmin=Z.min(), vmax=Z.max()), alpha=0.4)
plt.colorbar(cs)

plt.scatter(0,0, s=[400], marker="*", label="origin")
plt.scatter([-2.805118,-3.779310, 3.584428, 3],[3.131312,-3.283186, -1.848126, 2], marker="X", s=[90], label="targets")


plt.legend()

plt.show()


# %%


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
font = {
        'size'   : 18}

matplotlib.rc('font', **font)
def himmelblaup(x:float, y:float)-> float:
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
X = np.arange(-5, 5, 0.1)
Y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(X, Y)
Z = himmelblaup(X, Y)
#Z = rosenbrock(X, Y)

plt.figure(figsize=(8, 6))
cs = plt.contour(X, Y, Z, levels=100, cmap='Spectral',
                 norm=colors.Normalize(vmin=Z.min(), vmax=Z.max()), alpha=0.4)
plt.colorbar(cs)

import numpy
plt.scatter(0,0, s=[400], marker="*", label="origin")
import math
def v_len(pos):
    return math.sqrt(pos[0]**2+pos[1]**2)

def normalise(pos):
    len = v_len(pos)
    
    print(f"{pos} - {tuple(a / len for a in pos)}")
    return tuple(a / len for a in pos)

def random_vector(d: int):
    return tuple(numpy.random.normal() for i in range(d))


results = []
resultse = []
for i in range(10):
    direction = normalise(random_vector(2))
    new_point = tuple(sum(x) for x in zip(direction, (0,0)))
    resultse.append(new_point)
    direction_scaled = [x * 1 for x in direction]
    new_point = tuple(sum(x) for x in zip(direction_scaled, (0,0)))
    results.append(new_point)





plt.scatter([-2.805118,-3.779310, 3.584428, 3],[3.131312,-3.283186, -1.848126, 2], marker="X", s=[90], label="targets")

plt.scatter([x[0] for x in results],[x[1] for x in results], marker="^", s=[90], label="neighbours")



plt.legend()
plt.savefig("test.svg")
plt.show()





