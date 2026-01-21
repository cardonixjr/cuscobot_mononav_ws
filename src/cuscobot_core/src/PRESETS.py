#!/usr/bin/env python

# MG49 commands
SET_SPEED_RIGHT = chr(100)
SET_SPEED_LEFT = chr(101)
SET_SPEED = chr(102)
GET_LEFT_ENCODER = chr(105)
GET_RIGHT_ENCODER = chr(106)
RESET_ENCODER = chr(114)

# Frequency in Hz
UPDATE_FREQUENCY = 10

# Robot info
WHEEL_RADIUS = 0.06
WHEEL_BASE = 0.37
TICKS_PER_REVOLUTION = 980
MAX_PWM = 40
MAX_PWM_STEP = 30           # Biggest PWM step made each loop.
MAX_SPEED_DISTANCE = 1      # Distance (meters) from goal before the robot start reducing its speed.
ZERO_SPEED = 128

# Plotting
PLOTTING = True
