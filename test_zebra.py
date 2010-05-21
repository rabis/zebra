

import numpy, py.test

from zebra import *


def test_derivatives_scaling():
    N = 8
    ics = [
        Scaling("scaling-1-2-3-5", [1,2,3,5], numpy.array([0.1, 1.3, -1.0])),
        Scaling("scaling-3-5-6", [3,5,6]),
        Scaling("scaling-1-5-7", [1,5,7]),
    ]
    for ic in ics:
        check_derivatives(ic, N, q=0)
        check_derivatives(ic, N, q=1)


def test_derivatives_translation():
    N = 8
    ic = Translation("translation-x-1-4-6", [1,4,6], numpy.array([1,0,0]))
    check_derivatives(ic, N, q=0)
    check_derivatives(ic, N, q=1)


def test_derivatives_rotation():
    N = 8
    ic = Rotation("rotation-x-1-5-7", [1,5,7], numpy.array([1,0,0]))
    check_derivatives(ic, N, q=0)
    check_derivatives(ic, N, q=1)


@py.test.mark.xfail
def test_derivatives_composition():
    N = 8
    ic = Composition("sum-rotations",
        Rotation("rotation-x-1-4-6", [1,4,6], numpy.array([1,0,0])),
        Rotation("rotation-x-1-4-6", [1,2,6], numpy.array([1,0,0])),
    )
    check_derivatives(ic, N, q=0)
    check_derivatives(ic, N, q=1)


def test_translation_commuative():
    N = 8
    part1 = Translation("translation-x-1-4-6", [1,4,6], numpy.array([1,0,0]))
    part2 = Translation("translation-x-1-2-6", [1,2,6], numpy.array([1,0,0]))
    ic1 = Composition("sum12", part1, part2)
    ic2 = Composition("sum21", part2, part1)
    check_commutative(ic1, ic2, N)


def test_translation_rotation_commuative():
    N = 8
    part1 = Translation("translation-x-1-4-6", [1,4,6], numpy.array([1,0,0]))
    part2 = Rotation("rotation-x-1-2-6", [1,2,6], numpy.array([1,0,0]))
    ic1 = Composition("sum12", part1, part2)
    ic2 = Composition("sum21", part2, part1)
    check_commutative(ic1, ic2, N)


def check_derivatives(ic, N, q):
    # generate an array with shape (8,3) where each element is a random number
    # uniformly distributed between 0 and 1.
    coordinates = numpy.random.uniform(0,1,(N,3))
    eps = 1e-4
    c0 = ic.transform(coordinates, q-0.5*eps)
    c1 = ic.transform(coordinates, q+0.5*eps)
    delta = (c1 - c0)/eps
    delta_approx = ic.derivatives(coordinates, q)
    #print delta-delta_approx
    assert abs(delta-delta_approx).max() < 1e-3


def check_commutative(ic1, ic2, N):
    # generate an array with shape (8,3) where each element is a random number
    # uniformly distributed between 0 and 1.
    coordinates = numpy.random.uniform(0,1,(N,3))
    q = 1
    c0 = ic1.transform(coordinates, q)
    c1 = ic2.transform(coordinates, q)
    delta = (c1 - c0)
    #print delta
    assert abs(delta).max() < 1e-14



