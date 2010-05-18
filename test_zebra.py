

import numpy

from zebra import *

def test_derivatives_scaling():
    N = 8
    ics = [
        Scaling("scaling-1-2-3-5", [1,2,3,5], numpy.array([0.1, 1.3, -1.0])),
        Scaling("scaling-3-5-6", [3,5,6]),
        Scaling("scaling-1-5-7", [1,5,7]),
    ]
    for ic in ics:
        check_derivatives(ic, N)

def test_derivatives_translation():
    N = 8
    ic = Translation("translation-x-1-4-6", [1,4,6], numpy.array([1,0,0]))
    check_derivatives(ic, N)

def check_derivatives(ic, N):
    # generate an array with shape (8,3) where each element is a random number
    # uniformly distributed between 0 and 1.
    coordinates = numpy.random.uniform(0,1,(N,3))
    eps = 1e-4
    c0 = ic.transform(coordinates, -0.5*eps)
    c1 = ic.transform(coordinates, +0.5*eps)
    delta = (c1 - c0)/eps
    delta_approx = ic.derivatives(coordinates, 0)
    #print delta-delta_approx
    assert abs(delta-delta_approx).max() < 1e-3



