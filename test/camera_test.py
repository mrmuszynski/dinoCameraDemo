#! /usr/bin/env python3
# H+
#	Title   : camera_test.py
#	Author  : Matt Muszynski
#	Date    : 10/10/17
#	Synopsis: 
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

import sys
sys.path.insert(0, '../dependencies/')
from em import planck, stefanBoltzmann
from constants import T_sun, r_sun, au
from numpy import arange, pi
import bodies as bod
import camera
import numpy as np
import matplotlib.pyplot as plt
import pdb
#create a spacecraft object to be shared across tests
#a lot of the details of the spacecraft don't matter because
#they aren't used in the camera model. The really important one is 
#the state vector.
sc = bod.sc(
	"SC", #name of this body
	"Earth", #central body
	2451545.0, #Epoch of coe presented here, in JD
	122164 , #a in km
	0, # e
	0, #inclination in deg
	0, #omega in deg
	0, #OMEGA in deg
	0, #Mean Anomaly at epoch in deg
	np.array([]), #state vector in HCI frame. [x,y,z,v_x,v_y,v_z] 
	np.nan, #true anomaly measured at same time as state vector
	np.nan
	)

qe = {}
qe['lambda'] = np.arange(420,721,2.)
#the throughput definition here is a cheat to make something
#somewhat realistic while not spending time on researching that
#realism just yet.
qe['throughput'] = (100000-(qe['lambda']-570)**2) - \
	min(100000-(qe['lambda']-570)**2)
qe['throughput'] = 0.8*qe['throughput']/max(qe['throughput'])

tc = {}
tc['lambda'] = np.arange(390,721,1.7)
tc['throughput'] = (1000*4-(tc['lambda']-545)**4) - \
	min(1000*4-(tc['lambda']-545)**4)
tc['throughput'] = 0.6*tc['throughput']/max(tc['throughput'])

bod.earth.state = np.array([au,0,0,0,0,0])
bod.luna.state = bod.earth.state + 250000*np.array([0,1,0,0,0,0])
sc.state = bod.earth.state - 250000*np.array([1,0,0,0,0,0])

msg = { 'bodies': [
	bod.earth,
	bod.luna,
	sc
	], 
	'addStars': 0,'rmOcc': 0, 'addBod': 0, 'psf': 1, 
	'raster': 1, 'photon': 0, 'dark': 0, 'read': 0, 'dt': 0.01}

#create camera with no stars in it for tests that don't need them
#They will run significantly faster without them.
noStarCam = camera.camera(
	2, 					#detectorHeight
	2, 					#detectorWidth
	5.0, 				#focalLength
	512, 				#resolutionHeight
	512,				#resolutionWidth
	np.identity(3), 	#body2cameraDCM
	1000,		   	 	#maximum magnitude
	-1000,				#minimum magnitude (for debugging)
	qe,					#quantum efficiency dictionary
	tc,					#transmission curve dictionary
	1,					#wavelength bin size in nm
	0.01**2, 			#effective area in m^2
	100, 				#dark current in electrons per second
	100, 				#std for read noise in electrons
	100, 				#bin size
	2**16, 				#max bin depth
	sc,					#spacecraft the camera is attached to
	msg,				#debug message
	db='../db/tycho.db'	#stellar database
	)

#now create a camera with stars in it for use in the tests that
#actually need them.
msg['addStars'] = 1
starCam = camera.camera(
	2, 					#detectorHeight
	2, 					#detectorWidth
	5.0, 				#focalLength
	512, 				#resolutionHeight
	512,				#resolutionWidth
	np.identity(3), 	#body2cameraDCM
	1000,		   	 	#maximum magnitude
	-1000,				#minimum magnitude (for debugging)
	qe,					#quantum efficiency dictionary
	tc,					#transmission curve dictionary
	1,					#wavelength bin size in nm
	0.01**2, 			#effective area in m^2
	100, 				#dark current in electrons per second
	100, 				#std for read noise in electrons
	100, 				#bin size
	2**16, 				#max bin depth
	sc,					#spacecraft the camera is attached to
	msg,				#debug message
	db='../db/tycho.db'	#stellar database
	)
sc.attitudeDCM = np.identity(3)

def test_4_1_loadAllStars():
	#load support dict that was calculated offline
	test_4_1_support_dict = np.load('camera_test_support_files/4.1.test_support.npy')[0]
	
	#check that all sums of arrays loaded into the camera at run time
	#match what they were when the support dict was made.
	assert(sum(starCam.T) == test_4_1_support_dict['Tsum'])
	assert(sum(starCam.n1) == test_4_1_support_dict['n1sum'])
	assert(sum(starCam.n2) == test_4_1_support_dict['n2sum'])
	assert(sum(starCam.n3) == test_4_1_support_dict['n3sum'])
	assert(sum(starCam.RA) == test_4_1_support_dict['RAsum'])
	assert(sum(starCam.DE) == test_4_1_support_dict['DEsum'])
	assert(sum(starCam.VT) == test_4_1_support_dict['VTsum'])
	assert(sum(starCam.BVT) == test_4_1_support_dict['BVTsum'])

# def test_4_2_calculate_FOV():
# 	cam1 = camera.camera(
# 		1,2,2)
# 	cam2= camera.camera(
# 		3,4,5)
# 	cam3 = camera.camera(
# 		6,4,2)

# 	assert(cam1.angularHeight == 15)
# 	assert(cam1.angularWidth == 16)
# 	assert(cam1.angularDiagonal == 18)

# 	assert(cam2.angularHeight == 54)
# 	assert(cam2.angularWidth == 45)
# 	assert(cam2.angularDiagonal == 25)

# 	assert(cam2.angularHeight == 13)
# 	assert(cam2.angularWidth == 17)
# 	assert(cam2.angularDiagonal == 18)

def test_4_7_cameraUpdateState():
	assert(len(noStarCam.images) == 0)
	msg['takeImage'] = 1
	noStarCam.updateState()
	noStarCam.updateState()
	noStarCam.updateState()
	assert(len(noStarCam.images) == 1)
	assert(len(noStarCam.images[0].scenes) == 0)
	msg['takeImage'] = 0
	noStarCam.updateState()
	assert(len(noStarCam.images) == 1)
	assert(len(noStarCam.images[0].scenes) == 3)
	msg['takeImage'] = 1
	noStarCam.updateState()
	assert(len(noStarCam.images) == 2)
	assert(len(noStarCam.images[0].scenes) == 3)
	assert(len(noStarCam.images[1].scenes) == 0)
	noStarCam.updateState()
	msg['takeImage'] = 0
	noStarCam.updateState()
	assert(len(noStarCam.images) == 2)
	assert(len(noStarCam.images[0].scenes) == 3)
	assert(len(noStarCam.images[1].scenes) == 2)


def test_4_9_imageRemoveOccultations():
	#enforce position of earth and location of sc.
	#this way, earth is in the exact center of the FOV
	bod.earth.state = np.array([au,0,0,0,0,0])
	sc.state = bod.earth.state - 250000*np.array([1,0,0,0,0,0])
	sc.attitudeDCM = np.identity(3)

	#take an image pointed at the earth
	#remove the stars occulted by the earth
	#but don't add the earth back in
	msg['rmOcc'] = 1
	msg['addBod'] = 0
	msg['takeImage'] = 1
	starCam.updateState()
	msg['takeImage'] = 0
	starCam.updateState()
	
	#take another image, pointed in the same place, but don't
	#remove occulted stars
	msg['rmOcc'] = 0
	msg['takeImage'] = 1
	starCam.updateState()
	msg['takeImage'] = 0
	starCam.updateState()

	#find distance between center of FOV and each star in p/l coords.
	pixPos = starCam.images[0].scenes[0].pixel - starCam.resolutionWidth/2
	linePos = starCam.images[0].scenes[0].line - starCam.resolutionHeight/2

	#physical size of each pixel in the p/l directions
	pixSize = float(starCam.detectorWidth)/\
		float(starCam.resolutionWidth)
	lineSize = float(starCam.detectorHeight)/\
		float(starCam.resolutionHeight)

	#physical distance between center of FOV and each star in p/l coords
	pixDist = pixPos*pixSize
	lineDist = linePos*lineSize

	#physical distance between center of FOV and each star
	diagDist = np.sqrt(pixDist**2 + lineDist**2)

	#angular distance between each star and FOV center
	angDist = np.arctan2(diagDist,starCam.focalLength)

	#angular distance between center of Earth and limb
	sc2earthDist = np.linalg.norm((sc.state - bod.earth.state)[0:3])
	center2limbAng = np.arctan2(bod.earth.r_eq,sc2earthDist)

	#assert that star closest to the center of the FOV in the image
	#with stars removed is farther from the center than the limb of
	#the earth
	assert( min(angDist) > center2limbAng )


	#now find use the second image to find all the stars that were
	#removed from the first
	removedStars = starCam.images[1].scenes[0].starIDs
	ind = removedStars > -1
	for starID in starCam.images[0].scenes[0].starIDs:
		ind = np.logical_and(ind,removedStars != starID)

	rmPix = starCam.images[1].scenes[0].pixel[ind] - starCam.resolutionWidth/2
	rmLine = starCam.images[1].scenes[0].line[ind] - starCam.resolutionHeight/2
	rmPixDist = rmPix*pixSize
	rmLineDist = rmLine*lineSize
	rmDiagDist = np.sqrt(rmPixDist**2 + rmLineDist**2)
	rmAngDist = np.arctan2(rmDiagDist,starCam.focalLength)

	#assert that the number of stars in the image with stars removed
	#plus the number of stars removed from it equal the number of
	#stars in the image where none were removed
	assert(len((starCam.images[0].scenes[0].starIDs)) + sum(ind) \
		== len((starCam.images[1].scenes[0].starIDs)))
	#assert that the removed star farthest from the center of the FOV
	#is closer than the limb of the earth.
	assert( max(rmAngDist) < center2limbAng )
	plot = 0
	if plot:

		plt.figure()
		plt.plot(starCam.images[0].scenes[0].pixel, starCam.images[0].scenes[0].line, '.', markersize='2' )
		plt.plot(256,256,marker='x',color='red',markersize='5')
		plt.xlim([0,512])
		plt.ylim([0,512])
		plt.xlabel('Pixel')
		plt.ylabel('Line')
		plt.title('Test 4.9 Full Star Field with Occulted Stars Removed')
		plt.axes().set_aspect('equal')

		plt.figure()
		plt.plot(starCam.images[0].scenes[0].pixel, starCam.images[0].scenes[0].line, '.', markersize='5' )
		plt.plot(256,256,marker='x',color='red',markersize='5')
		plt.xlim([156,356])
		plt.ylim([156,356])
		plt.xlabel('Pixel')
		plt.ylabel('Line')
		plt.title('Test 4.9 Partial Star Field with Occulted Stars Removed')
		plt.axes().set_aspect('equal')

		plt.figure()
		plt.plot(rmPix+256, rmLine+256, '.', markersize='5' )
		plt.plot(256,256,marker='x',color='red',markersize='5')
		plt.xlim([156,356])
		plt.ylim([156,356])
		plt.xlabel('Pixel')
		plt.ylabel('Line')
		plt.title('Test 4.9 Partial Star Field with Non-Occulted Stars Removed')
		plt.axes().set_aspect('equal')

		plt.figure()
		plt.plot(starCam.images[1].scenes[0].pixel, starCam.images[1].scenes[0].line, '.', markersize='2' )
		plt.plot(256,256,marker='x',color='red',markersize='5')
		plt.xlim([0,512])
		plt.ylim([0,512])
		plt.xlabel('Pixel')
		plt.ylabel('Line')
		plt.title('Test 4.9 Full Star Field')
		plt.axes().set_aspect('equal')

		plt.figure()
		plt.plot(rmPix+256, rmLine+256, '.', markersize=2)
		plt.plot(256,256,marker='x',color='red',markersize='5')
		plt.xlim([0,512])
		plt.ylim([0,512])
		plt.xlabel('Pixel')
		plt.ylabel('Line')
		plt.title('Test 4.9 Full Star Field with Non-Occulted Stars Removed')
		plt.axes().set_aspect('equal')
		pdb.set_trace()

def test_4_16_pixelLineConversion():
	#find distance between center of FOV and each star in p/l coords.
	pixPos = starCam.images[1].scenes[0].pixel - starCam.resolutionWidth/2
	linePos = starCam.images[1].scenes[0].line - starCam.resolutionHeight/2

	#physical size of each pixel in the p/l directions
	pixSize = float(starCam.detectorWidth)/\
		float(starCam.resolutionWidth)
	lineSize = float(starCam.detectorHeight)/\
		float(starCam.resolutionHeight)

	pixDist = pixPos*pixSize
	lineDist = linePos*lineSize

	diagDist = np.sqrt(pixDist**2 + lineDist**2)

	#angualr distance computed by projecting physical distance
	#on focal plane through lens
	angDist = np.arctan2(diagDist,starCam.focalLength)

	#angular distance computed by dotting the camera boresight
	#vector with the unit vector to the star
	trueAngDist = np.arccos(starCam.images[1].scenes[0].c1)

	plot = 0

	if plot:
		plt.figure()
		plt.plot(starCam.images[1].scenes[0].pixel, starCam.images[1].scenes[0].line, '.', markersize='2' )
		plt.plot(256,256,marker='x',color='red',markersize='5')
		plt.xlim([0,512])
		plt.ylim([0,512])
		plt.xlabel('Pixel')
		plt.ylabel('Line')
		plt.title('Test 4.10 Star Field')
		plt.axes().set_aspect('equal')

	#assert that the worst error between the two angular distances
	#is still within machine precision	
	pdb.set_trace()
	assert( max(abs(angDist - trueAngDist)) < 1e-13 )


def test_4_17_planckEqStefanBoltzmann():
	sb = stefanBoltzmann(T_sun)*r_sun**2/au**2
	lambdaSet = arange(1,10001,1) #in nm
	lambdaSet = lambdaSet*1e-9 #convert to m
	bbCurve = planck(T_sun,lambdaSet)
	TSI = sum(pi*r_sun**2/au**2*bbCurve)
	assert( abs((TSI - sb)/sb) <0.001 )


def test_4_18_PlanckEqTSI():
	lambdaSet = arange(1,10001,1) #in nm
	lambdaSet = lambdaSet*1e-9 #convert to m
	bbCurve = planck(T_sun,lambdaSet)
	TSI = sum(pi*r_sun**2/au**2*bbCurve)
	assert( abs((TSI - 1367)/1367) <0.001 )


