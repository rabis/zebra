#!/usr/bin/env python

# http://molmod.ugent.be/code/static/doc/sphinx/molmod/latest/
# http://www.scipy.org/Tentative_NumPy_Tutorial

from molmod import *
from molmod.io import FCHKFile

import os, numpy, sys

def log_finished(fn_log):
    f = open(fn_log)
    for line in f:
        #print line
        if line.startswith(" Normal termination"):
            f.close()
            return True
    f.close()
    return False


def run_gaussian(mol, coordinates, prefix):
    # 1) create the input file
    f = file("{0}.com".format(prefix), "w")
    print >> f, "%Nproc=1"
    print >> f, "%chk={0}.chk".format(prefix)

    print >> f, "# HF/3-21G* sp force nosymm "
    print >> f
    print >> f, "comment"
    print >> f
    print >> f, "0 1"
    for i in xrange(mol.size):
        print >> f, mol.numbers[i], coordinates[i,0]/angstrom, coordinates[i,1]/angstrom, coordinates[i,2]/angstrom
    print >> f
    print >> f
    f.close()
    # 2) run gaussian
    fn_log = "{0}.log".format(prefix)
    if not os.path.isfile(fn_log):
        os.system("sqsub -q gaussian -n 1 -r 3.0d -o {0}.out g03 {0}.com".format(prefix))
    elif log_finished(fn_log):
        if not os.path.isfile("{0}.fchk".format(prefix)):
            os.system("formchk {0}.chk {0}.fchk".format(prefix))
        # 3) read the formatted checkpoint file
        fchk = FCHKFile("{0}.fchk".format(prefix))
        return fchk.fields["Total Energy"], fchk.fields["Cartesian Gradient"]


def main():
    prefix = sys.argv[1]
    fn_xyz = "{0}.xyz".format(prefix) # First argument from command line
    mol = Molecule.from_file(fn_xyz)

    q0 = 1.1
    c0 = q0*mol.coordinates
    result0 = run_gaussian(mol, c0, prefix+"0")

    eps = 1e-4
    q1 = q0 + eps
    c1 = q1*mol.coordinates
    result1 = run_gaussian(mol, c1, prefix+"1")

    if not (result0 is None or result1 is None):
        energy0, gradient0 = result0
        energy1, gradient1 = result1
        gradq0 = numpy.dot(gradient0, mol.coordinates.ravel())
        gradq1 = numpy.dot(gradient1, mol.coordinates.ravel())
        secondgrad = (gradq1 - gradq0)/eps
        print "original gradient is:", gradq0
        print "new gradient is:", gradq1
        print "second order gradient is:", secondgrad


if __name__ == "__main__":
    main()

