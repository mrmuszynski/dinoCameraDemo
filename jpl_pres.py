#! /usr/bin/env python3
# H+
#	Title   : jpl_pres.py
#	Author  : Matt Muszynski
#	Date    : 10/20/17
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
sys.path.insert(0, 'dependencies/')
from adcs import Euler321_2DCM
from constants import au
from numpy import arange, pi
import bodies as bod
import camera
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import pdb
import util


print('Initializing')
###############################################################################
#
#	Create a spacecraft object to attach the camera to.
#	A lot of the details of the spacecraft don't matter because
#	they aren't used in the camera model. The really important one is 
#	the state vector.
#
###############################################################################

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

###############################################################################
#
#	Pull in canned QE and Transmission curves from DINO C-REx files
#
###############################################################################

#load tranmission curve for Canon 10D
_10D = np.load('tc/10D.npz')
tc = {}
tc['lambda'] = _10D['x']
tc['throughput'] = _10D['y']

#load QE curve for Hubble Space Telecope Advanced Camera for Surveys SITe CCD
ACS = np.load('qe/ACS.npz')
qe = {}
qe['lambda'] = ACS['x']
qe['throughput'] = ACS['y']


###############################################################################
#
#	Set initial conditions for speacraft, Earth, and moon.
#
###############################################################################

i = np.deg2rad(-30)
j = np.deg2rad(0)
bod.earth.state = np.array([au/1000,0,0,0,0,0])
bod.luna.state = bod.earth.state + \
	250000*np.array([np.cos(i),np.sin(i),0,0,0,0])
sc.state = bod.earth.state - \
	42164*np.array([np.cos(j),np.sin(j),0,0,0,0])

msg = { 'bodies': [
	bod.earth,
	bod.luna,
	sc
	], 
	'addStars': 1,'rmOcc': 0, 'addBod': 0, 'psf': 1, 
	'raster': 1, 'photon': 1, 'dark': 1, 'read': 1, 'dt': 0.001}

print('Creating small camera images')

###############################################################################
#
#	Initialize camera for making small plots. Use these to show conversion
#	from stars in FOV in RA/DE space to pixel/line space, psf, and
#	rasterization
#
###############################################################################

smallCam = camera.camera(
	0.06, 				#detectorHeight
	0.06, 				#detectorWidth
	5.0, 				#focalLength
	25, 				#resolutionHeight
	25,					#resolutionWidth
	np.identity(3), 	#body2cameraDCM
	8,		    		#maximum magnitude
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
	)

#point the s/c at a good starfield for the demo I want to do
sc.attitudeDCM = util.ry(np.deg2rad(-2.4)).dot(util.rz(np.deg2rad(2.2)))

#take image with small cam
msg['takeImage'] = 1
smallCam.updateState()
msg['takeImage'] = 0
smallCam.updateState()

#plot full starfield
fig, ax = plt.subplots()
ax.plot(smallCam.RA,smallCam.DE,'w.',markersize=1)
ax.set_facecolor('black')
ax.set_ylim([-90,90])
ax.set_xlim([0,360])
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')
ax.set_title('Full DINO C-REx Star Field')

#plot only stars in image (in RA/DE space)
fig, ax = plt.subplots()
ax.plot(smallCam.images[0].scenes[0].RA,smallCam.images[0].scenes[0].DE,'w.')
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.set_ylim([-2.4-smallCam.angularHeight/2,-2.4+smallCam.angularHeight/2])
ax.set_xlim([2.2-smallCam.angularWidth/2,2.2+smallCam.angularWidth/2])
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')
ax.set_title('Star Field Reduced to Camera Field of View')

#plot only stars in image (in pixel/line space)
fig, ax = plt.subplots()
ax.plot(smallCam.images[0].scenes[0].pixel,smallCam.images[0].scenes[0].line,'w.')
ax.set_facecolor('black')
ax.set_ylim([0,25])
ax.set_xlim([0,25])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect('equal')
ax.set_xlabel('Pixel')
ax.set_ylabel('Line')
ax.set_title('Camera Field of View Converted to Pixel/Line Coordinates')

#plot only stars in image with psf applied (in pixel/line space)
fig, ax = plt.subplots()
ax.plot(smallCam.images[0].scenes[0].psfPixel,smallCam.images[0].scenes[0].psfLine,'w.',markersize=2)
ax.set_facecolor('black')
ax.set_ylim([0,25])
ax.set_xlim([0,25])
ax.set_ylim(ax.get_ylim()[::-1])
ax.set_aspect('equal')
ax.set_xlabel('Pixel')
ax.set_ylabel('Line')
ax.set_title('Camera Field of View with PSF Applied')

#plot rasterized image (including noise)
plt.figure()
plt.imshow(smallCam.images[0].scenes[0].detectorArray.reshape(25,25),cmap='Greys_r')
plt.xlabel('Pixel')
plt.ylabel('Line')
plt.title('Rasterized Star Image')

print('Creating large camera images')

###############################################################################
#
#	Initialize camera for larger image. 
#
###############################################################################

cam = camera.camera(
	3.0, 				#detectorHeight
	3.0, 				#detectorWidth
	5.0, 				#focalLength
	256, 				#resolutionHeight
	256,				#resolutionWidth
	np.identity(3), 	#body2cameraDCM
	7.9,	    		#maximum magnitude
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
	)

#point the s/c at a good starfield for the demo I want to do
sc.attitudeDCM = util.rz(np.deg2rad(190-10))

#take image with larger camera
msg['takeImage'] = 1
cam.updateState()
msg['takeImage'] = 0
cam.updateState()

#plot stars as simple points
fig, ax = plt.subplots()
ax.plot(cam.images[0].scenes[0].RA,cam.images[0].scenes[0].DE,'w.',markersize=2)
ax.set_facecolor('black')
ax.set_xlim(ax.get_xlim()[::-1])
ax.set_aspect('equal')
ax.set_ylim([-cam.angularHeight/2,cam.angularHeight/2])
ax.set_xlim([190-10-cam.angularWidth/2,190-10+cam.angularWidth/2])
ax.set_xlabel('Right Ascension')
ax.set_ylabel('Declination')
ax.set_title('Star Field Reduced to Camera Field of View')

#plot rasterized image with noise added
plt.figure()
plt.imshow(cam.images[0].scenes[0].detectorArray.reshape(256,256),cmap='Greys_r')
plt.xlabel('Pixel')
plt.ylabel('Line')
plt.title('Rasterized Star Image')

###############################################################################
#
#	Initialize camera for slew images. 
#
###############################################################################
print('Creating slew images')

msg = { 'bodies': [
	bod.earth,
	bod.luna,
	sc
	], 
	'addStars': 1,'rmOcc': 0, 'addBod': 0, 'psf': 1, 
	'raster': 1, 'photon': 1, 'dark': 1, 'read': 1, 'dt': 0.001}

cam = camera.camera(
	1, 					#detector_height
	1, 					#detector_width
	5.0, 				#focal_length
	512, 				#resolutionHeight
	512,				#resolutionWidth
	np.identity(3), 	#body2cameraDCM
	8,		    		#maximum magnitude
	-1000,				#minimum magnitude (for debugging)
	qe,					#quantum efficiency dictionary
	tc,					#transmission curve dictionary
	1,					#wavelength bin size in nm
	0.01**2, 			#effective area in m^2
	1000, 				#dark current in electrons per second
	1000, 				#std for read noise in electrons
    100, 				#bin size
    2**16, 				#max bin depth
	sc,					#spacecraft the camera is attached to
	msg,				#debug message
	)

takeImageArr = np.concatenate([np.zeros(1),np.ones(20),np.zeros(1)])

psi = 0
theta = 0
phi = 0

sc.attitudeDCM = np.identity(3)
msg['takeImage'] = 1
cam.updateState()
msg['takeImage'] = 0
cam.updateState()

for takeImage in takeImageArr:
	sc.attitudeDCM = Euler321_2DCM(psi,theta,phi)
	psi += np.deg2rad(0.05)
	msg['takeImage'] = takeImage
	cam.updateState()

for takeImage in takeImageArr:
	sc.attitudeDCM = Euler321_2DCM(psi,theta,phi)
	theta += np.deg2rad(0.05)
	msg['takeImage'] = takeImage
	cam.updateState()

for takeImage in takeImageArr:
	sc.attitudeDCM = Euler321_2DCM(psi,theta,phi)
	phi += np.deg2rad(0.15)
	msg['takeImage'] = takeImage
	cam.updateState()

takeImageArr = np.concatenate([np.zeros(1),np.ones(40),np.zeros(1)])

for takeImage in takeImageArr:
	sc.attitudeDCM = Euler321_2DCM(psi,theta,phi)
	psi += np.deg2rad(0.05)/2
	theta += np.deg2rad(0.05)/2
	phi += np.deg2rad(0.15)/2

	msg['takeImage'] = takeImage
	cam.updateState()


psi = np.deg2rad(90)
theta = 0
phi = 0
sc.attitudeDCM = Euler321_2DCM(psi,theta,phi)
msg['takeImage'] = 1
cam.updateState()
msg['takeImage'] = 0
cam.updateState()

plt.figure()
plt.imshow(cam.images[0].detectorArray.reshape(
	cam.resolutionHeight,cam.resolutionWidth),cmap='Greys_r')
plt.title('Still Image')
plt.xlabel('Pixel')
plt.ylabel('Line')

plt.figure()
plt.imshow(cam.images[1].detectorArray.reshape(
	cam.resolutionHeight,cam.resolutionWidth),cmap='Greys_r')
plt.title('0.1 Degree Yaw')
plt.xlabel('Pixel')
plt.ylabel('Line')

plt.figure()
plt.imshow(cam.images[2].detectorArray.reshape(
	cam.resolutionHeight,cam.resolutionWidth),cmap='Greys_r')
plt.title('0.1 Degree Pitch')
plt.xlabel('Pixel')
plt.ylabel('Line')

plt.figure()
plt.imshow(cam.images[3].detectorArray.reshape(
	cam.resolutionHeight,cam.resolutionWidth),cmap='Greys_r')
plt.title('0.1 Degree Roll')
plt.xlabel('Pixel')
plt.ylabel('Line')

plt.figure()
plt.imshow(cam.images[4].detectorArray.reshape(
	cam.resolutionHeight,cam.resolutionWidth),cmap='Greys_r')
plt.title('0.1 Degree Yaw, Pitch, and Roll')
plt.xlabel('Pixel')
plt.ylabel('Line')


plt.show()

###############################################################################
#
#	Initialize camera for animation. 
#
###############################################################################
print('Creating extended body animation (this one takes a a few minutes)')

#edit debug msg to ensure we include bodies
msg['rmOcc'] = 1
msg['addBod'] = 1

cam = camera.camera(
	4, 					#detectorHeight
	4, 					#detectorWidth
	5.0, 				#focalLength
	256, 				#resolutionHeight
	256,				#resolutionWidth
	np.identity(3), 	#body2cameraDCM
	1000,		    	#maximum magnitude
	-1000,				#minimum magnitude (for debugging)
	qe,					#quantum efficiency dictionary
	tc,					#transmission curve dictionary
	1,					#wavelength bin size in nm
	0.01**2, 			#effective area in m^2
	100, 			#dark current in electrons per second
	100, 			#std for read noise in electrons
    100, 			#bin size
    2**16, 			#max bin depth
	sc,					#spacecraft the camera is attached to
	msg,				#debug message
	)

#make the moon 10x bigger so we can easily see it in the final product
bod.luna.r_eq *= 10
bod.luna.r_pole *= 10
#make earth and moon much much dimmer than nominal so we can still
#see stars in final image
bod.earth.albedo = 1e-6
bod.luna.albedo = 3e-7

#container for animation frames
ims = []

#initialize figure for animation
fig = plt.figure()

#create frames for gif
j = np.deg2rad(-60)
for t in range(0,46):
	j += np.deg2rad(120/90)
	i += np.deg2rad(4/90)
	sc.attitudeDCM = util.rz(j)
	#this moon isn't really in orbit, but it gets the point across
	bod.luna.state = bod.earth.state + \
		250000*np.array([np.cos(i),np.sin(i),0,0,0,0]) + \
		np.array([0,0,40000,0,0,0])
	sc.state = bod.earth.state - \
		42164*np.array([np.cos(j),np.sin(j),0,0,0,0])

	msg['takeImage'] = 1
	cam.updateState()
	msg['takeImage'] = 0
	cam.updateState()
	im = plt.imshow(cam.images[t].detectorArray.reshape(256,256),cmap='Greys_r',animated=True)
	ims.append([im])



#create gif
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True,repeat_delay=1000)
plt.title('6 Hours in Geo')
plt.xlabel('Pixel')
plt.ylabel('Line')
plt.show()
pdb.set_trace()
