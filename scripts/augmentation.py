import math
import numpy
import random


def random_rotation(xy):
    theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    # return numpy.einsum('ptc,ci->pti', xy, r)
    return numpy.einsum('tc,ci->ti', xy, r)


def random_rotation_n(nxy, r=None):
    if r==None:
        theta = random.random() * 2.0 * math.pi
    else:
        theta = r * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    # return numpy.einsum('ptc,ci->pti', xy, r)
    return numpy.einsum('ptc,ci->pti', nxy, r)


def rotation(xy, r=None):
    if r==None:
        theta = random.random() * 2.0 * math.pi
    else:
        theta = r * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = numpy.array([[ct, st], [-st, ct]])
    # return numpy.einsum('ptc,ci->pti', xy, r)
    return numpy.einsum('tc,ci->ti', xy, r)
