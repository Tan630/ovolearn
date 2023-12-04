import types
class A:
    def printA(self):
        print ("----")
        print (self)

class B(A):
    pass

b = B()

b.__class__.__mro__[1].printA(1)