import random
from typing import Tuple


def getHalfRangeAndMidpoint(lower: float, upper: float) -> Tuple[float, float]:
    """
    basically gives you the 2nd the argument for twosComplementBinaryArrayGenerator if all you have
    is a range
    :param lower: lower bound of the range
    :param upper: upper bound of the range
    :return: range/2 and the midpoint of the range
    """
    if lower > upper:
        l2 = lower
        lower = upper
        upper = l2
    halfRan: float = (upper-lower)/2
    mid: float = upper - halfRan
    return halfRan, mid

print(getHalfRangeAndMidpoint(-0.5,2.5))

print(getHalfRangeAndMidpoint(-7,4))

if 1:
    print("1 yes")
else:
    print("1 no")

if 0:
    print("0 yes")
else:
    print("0 no")


bits = [1, 2, 3, 4, 5, 6]
bits2 =[7, 8, 9,10,11,12]

print(bits[5])


buffer = 1
crossoverPoint = random.randint(buffer, len(bits)-1-buffer)

for k in range(1,len(bits)):

    crossoverPoint = k

    result = []

    print(crossoverPoint)

    for i in range(0, crossoverPoint):
        print(i, ":", bits[i])
        result.append(bits[i])

    print(result)
    print("crossed over")

    for i in range(crossoverPoint, len(bits2)):
        print(i, ":", bits2[i])
        result.append(bits2[i])

    print(result)


print("random tests")
for i in range(0,10000):
    ran = random.random()
    if (ran <= 0.001):
        print (i, ": ", ran)


class Bob:
    def __init__(self):
        self.counter: int = -1

    def count(self):
        self.counter += 1


def bobCounter(itsBob):
    itsBob.count()


def bobChecker(itsBob):
    print(itsBob.counter)


bob = Bob()
for i in range(0,10):
    bobCounter(bob)
    bobChecker(bob)

bob2 = Bob()
while bob2.counter <= 5:
    bobCounter(bob2)
    bobChecker(bob2)


def anotherFunction(x,y):
    return (x > 0) and (y > 0) and (x > y)


print(anotherFunction(1,2))
print(anotherFunction(2,1))