#! /usr/bin/env python3
# H+
#	Title   : orbits.py
#	Author  : Matt Muszynski
#	Date    : 09/04/16
#	Synopsis: Functions for orbit stuff
# 
#
#	$Date$
#	$Source$
#  @(#) $Revision$
#	$Locker$
#
#	Revisions:
#
# H-
# U+
#	Usage   :
#	Example	:
#	Output  :
# U-
# D+
#
# D-
###############################################################################

def tilde(threevec):
	import numpy as np
	x1, x2, x3 = threevec
	return np.array([
		[  0,-x3, x2],
		[ x3,  0,-x1],
		[-x2, x1,  0]
		])

def symbolic_tilde(vector):
	import sympy as sym
	x1 = vector[0]
	x2 = vector[1]
	x3 = vector[2]
	return sym.Matrix([
		[  0,-x3, x2],
		[ x3,  0,-x1],
		[-x2, x1,  0]
		])

def cayley_transform(matrix):
	import numpy as np
	import numpy.linalg as la
	#define identity in a generalized way so function can be used
	#in n dimensions
	ident = np.identity(matrix.shape[0])
	return (ident-matrix)*la.inv(ident+matrix)

def euler_params2dcm(beta_vec):
	import numpy as np
	b0 = beta_vec.item(0)
	b1 = beta_vec.item(1)
	b2 = beta_vec.item(2)
	b3 = beta_vec.item(3)
	return np.matrix([
		[b0**2+b1**2-b2**2-b3**2,         2*(b1*b2+b0*b3),         2*(b1*b3-b0*b2)],
		[        2*(b1*b2-b0*b3), b0**2-b1**2+b2**2-b3**2,         2*(b2*b3+b0*b1)],
		[        2*(b1*b3+b0*b2),         2*(b2*b3-b0*b1), b0**2-b1**2-b2**2+b3**2]
		])


def crp2dcp(crp_vec):
	import numpy as np
	q1 = crp_vec.item(0)
	q2 = crp_vec.item(1)
	q3 = crp_vec.item(2)

	q_vec = np.matrix([q1,q2,q3]).T

	return 1/(1+q_vec.T*q_vec).item(0)*\
	(
		(1-q_vec.T*q_vec).item(0)*np.identity(3) + \
		2*q_vec*q_vec.T - 
		2*tilde(q_vec)
	)


def Euler321_2DCM(psi,theta,phi):
	from util import r1, r2, r3
	return r1(phi).dot(r2(theta).dot(r3(psi)))









