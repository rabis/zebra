#!/usr/bin/env python

# http://molmod.ugent.be/code/static/doc/sphinx/molmod/latest/
# http://www.scipy.org/Tentative_NumPy_Tutorial

from molmod import *
from molmod.io import FCHKFile

import os, numpy, sys

def log_finished(fn_log):
    """Return True if the Gaussian computation file has properly terminated.

       Arguments:
        | fn_log  --  The filename of the Gaussian log file.
    """
    f = open(fn_log)
    for line in f:
        #print line
        if line.startswith(" Normal termination"):
            f.close()
            return True
    f.close()
    return False


def run_gaussian(mol, coordinates, prefix):
    """this function creates the input file and then submit it to bull. Then if the job is done, it will convert the checkpoint file to a formatted one"""
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
        print "Job %s successful" % prefix
        return fchk.fields["Total Energy"], fchk.fields["Cartesian Gradient"]
    print "Job %s not ready (yet)" % prefix
    return None


class InternalCoordinate(object):
    def transform(self, coordinates_in, q):
        raise NotImplementedError

    def derivatives(self, coordinates_in, q):
        raise NotImplementedError

    def compute_norm(self, coordinates_in):
        return numpy.linalg.norm(self.derivatives(coordinates_in, 0, normalized=False))


class Scaling(InternalCoordinate):
    def __init__(self, name, indexes, center=None):
        """
           Arguments:
            | name  --  name of the internal coordinate
            | indexes -- list of atom number (starting from zero, so it is one less than gaussview)
            | center -- is the origin of the scaling
            """
        self.name = name
        self.indexes = indexes
        self.center = center
        
    def get_center(self, coordinates_in):
        """this method return center: there are two cases:
            1) when you have not specified center, then it will choose the average of the chosen indexes
             2) you have specified center, then it is the center.
       """
        if self.center is None:
            center = coordinates_in[self.indexes].mean(axis=0)
        else:
            center = self.center
        return center
            
    def transform(self, coordinates_in, q):
        """this method return new transformed coordinates. First it copies the old coordinates, and then it transforms to a new coordinates
        Arguments:
        |coordinates_out: new transformed coordinates
        """
        coordinates_out = coordinates_in.copy()
        center = self.get_center(coordinates_in)
        qp = q/self.compute_norm(coordinates_in)
        for i in self.indexes:
            coordinates_out[i] = (1+qp)*(coordinates_in[i]-center) + center
        return coordinates_out

    def derivatives(self, coordinates_in, q, normalized=True):
        """ this method return first derivative(called result) it is defined as initial coordinates minus the center
        Arguments:
        |result : derivative 
        """
        result = numpy.zeros(coordinates_in.shape, float)
        center = self.get_center(coordinates_in)
        for i in self.indexes:
            result[i] = (coordinates_in[i]-center)
        if normalized:
            norm = self.compute_norm(coordinates_in)
        else:
            norm = 1
        return result/norm


class Translation(InternalCoordinate):
    def __init__(self, name, indexes, direction):
        """
           Arguments:
            | name  --  name of the internal coordinate
            | indexes -- list of atom number (starting from zero, so it is one less than gaussview)
            | center -- is the origin of the scaling
            """
        self.name = name
        self.indexes = indexes
        self.direction = direction
            
    def transform(self, coordinates_in, q):
        """this method return new transformed coordinates. First it copies the old coordinates, and then it transforms to a new coordinates
        Arguments:
        |coordinates_out: new transformed coordinates
        """
        coordinates_out = coordinates_in.copy()
        qp = q/self.compute_norm(coordinates_in)
        for i in self.indexes:
            coordinates_out[i] = qp*self.direction + (coordinates_in[i]) 
        return coordinates_out

    def derivatives(self, coordinates_in, q, normalized=True):
        """ this method return first derivative(called result) it is defined as initial coordinates minus the center
        Arguments:
        |result : derivative 
        """
        result = numpy.zeros(coordinates_in.shape, float)
        for i in self.indexes:
            result[i] = self.direction
        if normalized:
            norm = self.compute_norm(coordinates_in)
        else:
            norm = 1
        return result/norm
        
        
class Rotation(InternalCoordinate):
    def __init__(self, name, indexes, axis, center=None):
        """
           Arguments:
            | name  --  name of the internal coordinate
            | indexes -- list of atom number (starting from zero, so it is one less than gaussview)
            | center -- is the origin of the scaling
            """
        self.name = name
        self.indexes = indexes
        self.center = center
        self.axis = axis
        
    def get_center(self, coordinates_in):
        """this method return center: there are two cases:
            1) when you have not specified center, then it will choose the average of the chosen indexes
             2) you have specified center, then it is the center.
       """
        if self.center is None:
            center = coordinates_in[self.indexes].mean(axis=0)
        else:
            center = self.center
        return center
            
    def transform(self, coordinates_in, q):
        """this method return new transformed coordinates. First it copies the old coordinates, and then it transforms to a new coordinates
        Arguments:
        |coordinates_out: new transformed coordinates
        """
        coordinates_out = coordinates_in.copy()
        center = self.get_center(coordinates_in)
        qp = q/self.compute_norm(coordinates_in)
        for i in self.indexes:
            v1 = coordinates_in[i] - center
            v2 = v1*numpy.cos(qp) + numpy.cross(self.axis, v1)*numpy.sin(qp) + self.axis*(numpy.dot(self.axis, v1))*(1-numpy.cos(qp))
            coordinates_out[i] = v2 + center
        return coordinates_out

    def derivatives(self, coordinates_in, q, normalized=True):
        """ this method return first derivative(called result) it is defined as initial coordinates minus the center
        Arguments:
        |result : derivative 
        """
        result = numpy.zeros(coordinates_in.shape, float)
        center = self.get_center(coordinates_in)
        if normalized:
            norm = self.compute_norm(coordinates_in)
            qp = q/norm
        else:
            qp = 1
            norm = 1
        for i in self.indexes:
            v1 = coordinates_in[i] - center
            result[i] = -v1*numpy.sin(qp) + (numpy.cross(self.axis, v1)*numpy.cos(qp)) + self.axis*(numpy.dot(self.axis, v1))*numpy.sin(qp)
        return result/norm


def main():
    """ here we choose the different scaling that we want to do scaling class on it
    Arguments : ics -- it will keep all the different scalings
    """
                
    ics = [
        Scaling("scaling-1-2-3-5", [1,2,3,5], numpy.array([0.1, 1.3, -1.0])), 
        Scaling("scaling-3-5-6", [3,5,6]), 
        Scaling("scaling-1-5-7", [1,5,7]),
        Translation("translation-x-1-4-6", [1,4,6], numpy.array([1,0,0])),
        Rotation("rotation-x-1-5-7", [1,5,7], numpy.array([1,0,0]))
    ]

    prefix = sys.argv[1]
    fn_xyz = "{0}.xyz".format(prefix) # First argument from command line
    mol = Molecule.from_file(fn_xyz)
    eps = 1e-5
    results = [] 
    ready = True
    for ic in ics:
        result_m = run_gaussian(mol, ic.transform(mol.coordinates,-0.5*eps), prefix + "_" + ic.name + "_m")
        result_p = run_gaussian(mol, ic.transform(mol.coordinates,+0.5*eps), prefix + "_" + ic.name + "_p")
        results.append((result_m, result_p))
        if (result_m is None or result_p is None):
            ready = False
   
    if ready:
        hessian = numpy.zeros((len(ics), len(ics)), float)
        for i in xrange(len(ics)): # finite difference
            for j in xrange(len(ics)): # analytical derivation
                energy0, gradient0 = results[i][0] # minus
                energy1, gradient1 = results[i][1] # plus
                ic = ics[j]
                gradq0 = numpy.dot(gradient0, ic.derivatives(mol.coordinates,-0.5*eps).ravel())
                gradq1 = numpy.dot(gradient1, ic.derivatives(mol.coordinates,+0.5*eps).ravel())
                
                hessian[i,j] = (gradq1 - gradq0)/eps
                print "original gradient is:", gradq0
                print "new gradient is:", gradq1
                print "second order gradient is:", hessian[i,j]
        
        print hessian
        hessian = 0.5*(hessian + hessian.transpose())
        print hessian


if __name__ == "__main__":
    main()

