#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Authors: Stephanie C. Leach & Kai Hwang.
University of Iowa, Iowa City, IA
Hwang Lab, Dpt. of Psychological and Brain Sciences
As of Feb. 1, 2026:
    Office:355N PBSB
    Office Phone:319-467-0610
    Fax Number:319-335-0191
    Lab:355W PBSB
Lab Contact Info: 
    Web - https://kaihwang.github.io/
    Email - kai-hwang@uiowa.edu

This version of the Psychopy script is designed to mimic and model functions
written from the psychopy builder

However, this script has been heavily edited with python code non-specific to the builder.
Take this into account when reading through this as script design here does not mirror builder
script layout. 
"""
# # # ----------- Import functions/packages ------------- # # #
from psychopy import gui, visual, core, data, event, monitors, sound, hardware
from psychopy.hardware import keyboard
try:
    from psychopy.hardware.parallel import ParallelPort
except:
    print("unable to load MEG port packages!!!!")
import numpy as np  # whole numpy lib is available, prepend 'np.'
from random import choice as randomchoice
import random
import os  # handy system and path functions
import sys  # to get file system encoding
import shutil
import glob
import math
import glob # this pulls files in directories to create Dictionaries/Str
import pandas as pd #Facilitates data structure and analysis tools. prepend 'np.'
from datetime import datetime
from PIL import Image
import serial
import matplotlib.pyplot as plt
try:
    import pylink
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
except:
    print("unable to load eyetracking packages!!!")

######################################################################################################################## 
##################################################   Initialization  ##################################################
# # # ----------- Change Directory ------------- # # #
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))#.decode(sys.getfilesystemencoding())
os.chdir(_thisDir)
img_dir = os.path.join(_thisDir, "cue_objects") # sub folders are "faces" or "scenes"
output_dir = os.getcwd() #"/Users/scleach/OneDrive - University of Iowa/Research/Research_Projects/human task/"
print(output_dir)


# -- load audio device manager
# IF THE SCRIPT CRASHES b/c of this, it means you need to install the package 
try:
    deviceManager = hardware.DeviceManager()
    #print(deviceManager.getAvailableDevices())
    # initialize 'ab'
    deviceManager.addDevice(
        deviceName='ab',
        index=6.0,
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        resample=True,
        latencyClass=1,
    )
    sound.Sound.backend = 'ptb'
    modality_opts = ['visual','auditory']
except:
    print("current system unable to load speaker device... visual presentation only")
    modality_opts = ['visual']

######################################################################################################################
##################################################   Get Subj Info  ##################################################
# -- set up for GUI
subInfo = {'Subject_ID': 99999, 
            'Consented': ['no','yes'],
            'Block': 1, 
            'Stay_R':'0.33',
            'IDS_R':'0.5',
            'context_switch_prob':['equal','unequal'],
            'context_modality':modality_opts,
            'probe_subset':[False,True],
            'method': ['Behavioral','Behavioral+pupil','EEG','EEG+pupil','MRI','MEG','MEG+pupil']} 
# -- use gui to get info before starting the experiment
dlg = gui.DlgFromDict(dictionary=subInfo, title="Subject Info Box", sortKeys=False)
if dlg.OK == False:
    core.quit()  # user pressed cancel
cur_subid = subInfo['Subject_ID'] # pull out subid 
context_modality = subInfo['context_modality']
probe_subset = subInfo['probe_subset']
start_blk = int(subInfo['Block'] - 1)

if subInfo['context_switch_prob']=='unequal':
    low_prob_context = np.random.randint(1,4) # randomly pick 1, 2, or 3
else:
    low_prob_context = np.nan

# -- load subject info sheet to get and save gui info
subj_info_df = pd.read_csv("HCC_Subject_Info.csv")
new_row = pd.DataFrame({'Subject_ID':subInfo['Subject_ID'], 
                        'Method':subInfo['method'], 
                        'Consent_Confirmed':subInfo['Consented'], 
                        'SessionDate':datetime.today().strftime('%m/%d/%y'), 
                        'StartTime':datetime.today().strftime('%I:%M:%S %p'), 
                        'Version':5, 
                        'Probe_Subset':probe_subset,
                        'Context_Switch_Probability':subInfo['context_switch_prob'],
                        'Low_Probability_Context':low_prob_context,
                        'Context_Modality':context_modality}, index=[1])
subj_info_df = pd.concat([subj_info_df, new_row], ignore_index=True) # Append new row to existing dataframe using concat
# -- save out current subject info
#print(subj_info_df)
if subInfo['Subject_ID'] < 90000:
    # use IDs starting with 9 to indicate not a real subject
    subj_info_df.to_csv("HCC_Subject_Info.csv", index=False)


max_time = float('inf') # float(inf) would set as inf (or until button press), any other value is okay for testing code
core.wait(1) # pause 1 second so we can read output printed to terminal



#######################################################################################################################
##################################################   Set up Monitor  ##################################################
brightness = .8 # 0.0 - 1.0
##### Define monitor window
if subInfo['method'] == 'Behavioral':
    expInfo = {'win':[1920,1080], 'frameRate':None, 'colorSpace':'hsv', 'backgroundColor':[0,0,brightness]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('Behavioral/EEGRoomMonitor', distance = 70, width = 53.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=False , monitor=mon, checkTiming=True, colorSpace = 'hsv', color=[0,0,brightness])
    #fullscr = True once you have everything tested

if subInfo['method'] == 'EEG':
    expInfo = {'win':[1920,1080], 'frameRate':None, 'colorSpace':'hsv', 'backgroundColor':[0,0,brightness]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('Behavioral/EEGRoomMonitor', distance = 70, width = 53.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=True, monitor=mon, checkTiming=True, colorSpace = 'hsv', color=[0,0,brightness])
    #fullscreen = True once you have everything tested 

if 'pupil' in subInfo['method']:
    expInfo = {'win':[1920,1080], 'frameRate':None, 'colorSpace':'hsv', 'backgroundColor':[0,0,brightness]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('Behavioral/EEGRoomMonitor', distance = 60, width = 53.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], winType='pyglet', units = "pix", fullscr=True, monitor=mon, checkTiming=True, colorSpace = 'hsv', color=[0,0,brightness])
    
    # ---- code from Zexuan Niu
    scn_width, scn_height = win.size
    # Set this variable to True to run the script in "Dummy Mode"
    dummy_mode = False

if subInfo['method'] == 'MRI':
    expInfo = {'win':[3840,2160], 'frameRate':None, 'colorSpace':'rgb', 'backgroundColor':[0,0,0]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('testMonitor', distance = 50, width = 28.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=True, monitor=mon, checkTiming=True, colorSpace = 'rgb', color=[0,0,0])
    #fullscreen = True once you have everything tested 
    
if 'MEG' in subInfo['method']:
    expInfo = {'win':[1280, 800], 'frameRate':None, 'colorSpace':'rgb', 'backgroundColor':[0,0,0]} # set up to save exp info (task parameters, trial info, etc)
    visual_unit = 'deg' #'pix' #
    mon = monitors.Monitor('testMonitor', distance = 70, width = 53.5)
    mon.setSizePix(expInfo['win'])
    win = visual.Window(expInfo['win'], units=visual_unit, fullscr=True, monitor=mon, checkTiming=True, colorSpace = 'rgb', color=[0,0,0])
    #fullscreen = True once you have everything tested 

##### Temporal parameters.
expInfo['frameRate'] = win.getActualFrameRate() # store frame rate of monitor if we can measure it
if expInfo['frameRate'] != None:
    frame_rate = round(expInfo['frameRate'])  # frame_rate_def = 1.0 / round(expInfo['frameRate'])
else:
    frame_rate = 60.0  # could not measure, so default 60 Hz   #frame_rate_def = 1.0 / 60.0 
# Add multiple shutdown keys "at once".
# for key in ['q', 'escape']:
#     event.globalKeys.add(key, func=core.quit)
refresh_rate=expInfo['frameRate']

win.mouseVisible = False # remove mouse from screen

# # ----- minimize window until task is ready
# win.winHandle.minimize() 
# win.winHandle.set_fullscreen(False)

######################################################################################################################
########################################  DEFINE EXP PARAMS  ##########################################
#### Timing and trial keeping variables
# -----  BEHAVIORAL  ----- | -----     EEG     ----- | -----     MRI     -----|
# -- Cue Obj   1500 ms     | -- Cue Obj   1000 ms    | -- Cue Obj   1000 ms   |   
# -- Delay     jittered    | -- Delay     jittered   | -- Delay     jittered  |     
# -- Stim       NA ms      | -- Stim       NA ms     | -- Stim       NA ms    |       
# -- ISI       jittered    | -- ISI       jittered   | -- ISI       jittered  |     
# -- Probe     1500 ms     | -- Probe     1500 ms    | -- Probe     1500 ms   |   
# -- ITI       jittered    | -- ITI       jittered   | -- ITI       jittered  |       
# -----------------------------------------------------------------------------
if subInfo['method']=='Behavioral':
    Task_Parameters = {'n_blocks': 20, 'n_trials': 74, 'n_prac_trials': 18}
    Trl_Durs = {'cue': 1.0, 
                'delay': [0.200,0.300], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1,2]}
    Number_of_initial_stays = 2
    real_Total_trials = 72 + Number_of_initial_stays #  number of first stay trials 
    trigDict = {}

elif subInfo['method']=='Behavioral+pupil':
    Task_Parameters = {'n_blocks': 20, 'n_trials': 74, 'n_prac_trials': 18}
    Trl_Durs = {'cue': 1.5, 
                'delay': [0.350,0.650], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1,2]}
    Number_of_initial_stays = 2
    real_Total_trials = 72 + Number_of_initial_stays #  number of first stay trials 
    trigDict = {}
    
elif subInfo['method']=='EEG':
    Task_Parameters = {'n_blocks': 20, 'n_trials': 74, 'n_prac_trials': 18}
    Trl_Durs = {'cue': 1.0, 
                'delay': [0.350,0.650], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1.5,3]}
    Number_of_initial_stays = 2
    real_Total_trials = 72 + Number_of_initial_stays #  number of first stay trials 
    #trigDict = {}

elif subInfo['method']=='EEG+pupil':
    Task_Parameters = {'n_blocks': 20, 'n_trials': 74, 'n_prac_trials': 18}
    Trl_Durs = {'cue': 1.0, 
                'delay': [0.350,0.650], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1.5,3]}
    Number_of_initial_stays = 2
    real_Total_trials = 72 + Number_of_initial_stays #  number of first stay trials 
    #trigDict = {}

elif subInfo['method']=='MRI':
    Task_Parameters = {'n_blocks': 12, 'n_trials': 38, 'n_prac_trials': 18} # number of blocks HAS TO BE an even number
    Trl_Durs = {'cue': 1.0, 
                'delay': [0.25,0.35], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1.25,1.75]}
    Number_of_initial_stays = 2
    real_Total_trials = 36 + Number_of_initial_stays #  number of first stay trials 
    trigDict = {}
    if Task_Parameters['n_blocks'] % 2 != 0: # making sure the IDS/EDS ratio make sense
        print('REQUESTED BLOCK NUMBER NOT DIVISIBLE BY 2 !!!! CODE WILL NOT WORK !!! CRASHING SCRIPT')
        core.quit()

elif subInfo['method']=='MEG':
    Task_Parameters = {'n_blocks': 20, 'n_trials': 74, 'n_prac_trials': 18}
    Trl_Durs = {'cue': 1.0, 
                'delay': [0.350,0.650], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1,2]}
    Number_of_initial_stays = 2
    real_Total_trials = 72 + Number_of_initial_stays #  number of first stay trials 
    #trigDict = {}
    
elif subInfo['method']=='MEG+pupil':
    Task_Parameters = {'n_blocks': 20, 'n_trials': 74, 'n_prac_trials': 18}
    Trl_Durs = {'cue': 1.0, 
                'delay': [0.350,0.650], 
                #'stim': .25, 
                #'ISI': [0.350,0.650], 
                'probe': 1.5, 
                'feedback': 0.25,
                'ITI': [1,2]}
    Number_of_initial_stays = 2
    real_Total_trials = 72 + Number_of_initial_stays #  number of first stay trials 
    #trigDict = {}

if Task_Parameters['n_blocks'] % 2 != 0: # same w switch ratio
    print('REQUESTED BLOCK NUMBER NOT DIVISIBLE BY 2 !!!! CODE WILL NOT WORK !!! CRASHING SCRIPT')
    core.quit()

# ---- ITI jitter
# Note, add 2 trials to start that will be tossed, timing is estimated below
#           pre-trial    cue    iti_1   probe   iti_2
# trial_1:     0.5       1.0    1.0     1.5     2.0
# trial_2:     N/A       1.0    4.5     1.5     N/A  * note iti_2 on this trial is the 6 seconds start time est by AFNI
pre_first_trial_delay = 0.50 # that way 2nd trial starts at 6 seconds
if subInfo['method']=='MRI':
    iti_dict = {'delay1':{}, 'ITI':{}}
    ITIpath = os.path.join(_thisDir,"ITIs/") #'Z:/Generate_ITIs/HCC_ITIs/'
    iti_list = os.listdir(ITIpath)
    iti_inds = np.arange(0,len(iti_list))
    for blk in range(Task_Parameters['n_blocks']):
        # grab random file from list
        np.random.shuffle(iti_inds) # shuffle order to randomize
        ITI_rand_file=iti_list[ iti_inds[0] ] # grab value at randomly selected index
        ITI_rand_file=open(ITIpath+ITI_rand_file,'r').readlines()
        # remove file index from list so we don't pick again
        np.delete(iti_inds, 0)
        # now loop to grab delay and iti values
        jitter_delay1=[1.0, 4.5] # since 1st trial is to be tossed, set first jitter to 1.5s; set 2nd to 4.5s
        jitter_ITI=[2.0, 4.5] # since 1st trial is to be tossed, set first jitter to 1.5s; set 2nd to 6s
        for odd_even, i in enumerate(ITI_rand_file):
            t=i.split('\n')
            t=t[0]
            if not(t==''):
                if int(odd_even+1)%2 == 1:
                    # odd rows are delay rows
                    jitter_delay1.append(float(t))
                else:
                    # even rows are ITI rows
                    jitter_ITI.append(float(t))
        iti_dict['delay1'][blk] = jitter_delay1
        iti_dict['ITI'][blk] = jitter_ITI


############################################
#### Set up for EEG or MEG
if 'EEG' in subInfo['method']:
    trigDict = {'startSaveflag':bytes([201]), 'stopSaveflag':bytes([255]), 'blockEnd':203, 'pracEnd':205,
                'cue':{'shape':{'bvs':11, 'bgm':12, 'grm':13, 'gvl':14, 'hrs':15, 'hgl':16}, 
                        'color':{'bvs':21, 'bgm':22, 'grm':23, 'gvl':24, 'hrs':25, 'hgl':26},
                        'outline':{'bvs':31, 'bgm':32, 'grm':33, 'gvl':34, 'hrs':35, 'hgl':36}}, # grab diff codes depending on state 1 or -1 and cue 1(B) or -1(R)
                'delay':99,  
                'probe':{1:41, 2:42, 3:43, 4:44}, 
                'resp':{4:1, 5:2, 1:3, 2:4, 'yes':1, 'no':2},  
                'feedback':{1:7,
                            0:8},
                'ITI':98}
    port=serial.Serial('COM6',baudrate=115200)
    port.close()
# cue [1|2|3] [1|2|3|4|5|6] ... 1# = shape context, 2# = color context, 3# = outline context
# ... #1 = 'bvs', #2 = 'bgm', #3 = 'grm', #4 = 'gvl', #5 = 'hrs', #6 = 'hgl'
if 'MEG' in subInfo['method']:
    trigDict = {'cue':11, # grab diff codes depending on state 1 or -1 and cue 1(B) or -1(R)
                'probe':22, 
                'resp':99,  
                'feedback':33}
    port=ParallelPort(address = 0x0378)
    # send trigger with following call
    # # Send a trigger (e.g., value 1)
    # port.setData(1)
    # # Reset trigger to 0 to avoid continuous signals
    # port.setData(0)


##############################################################################################################################
##################################################   Initialize Variables   ##################################################
##### Response keys and keyboard
# if int(subInfo['Yes_Key'])==1:
#     resp_keys = {'yes':'num_1','no':'num_2'}
#     resp_dict = {1:'yes', 2:'no'}
#     resp_instructions = "YES is farthest left, and NO is farthest right"
# else:
#     resp_keys = {'yes':'num_2','no':'num_1'}
#     resp_dict = {2:'yes', 1:'no'}
#     resp_instructions = "YES is farthest right, and NO is farthest left"
if subInfo['method']=='MRI':
    # using button box
    #  L-I 2 | R-I 6
    # -------|-------
    #  L-M 3 | R-M 7
    # using 4 key pad
    #  4 | 1                 2 | 3 
    # ---|---   or flipped  ---|---
    #  3 | 2                 1 | 4 
    resp_keys = {1:'4', 2:'1', 3:'3', 4:'2'}
    resp_dict = {1:'upper_left', 2:'upper_right', 3:'lower_left', 4:'lower_right'}
    resp_instructions = "[left index] is upper left, [right index] is upper right, [left middle] is lower left, [right middle] is lower right"
    bottom_quads = {'left':'3', 'right':'2'}
else:
    #    keyboard        how we coded quad
    #  num_4 | num_5   |   1 | 2
    # -------|-------  |  ---|---
    #  num_1 | num_2   |   3 | 4
    resp_keys = {1:'num_4', 2:'num_5', 3:'num_1', 4:'num_2'}
    resp_dict = {1:'upper_left', 2:'upper_right', 3:'lower_left', 4:'lower_right'}
    resp_instructions = "[4] is upper left, [5] is upper right, [1] is lower left, [2] is lower right"
    bottom_quads = {'left':'lower_left',  'right':'lower_right'}
kb = keyboard.Keyboard() # keyboard has better timing that other keypress functions
# 1 = upper left   2 = upper right   3 = lower left   4 = lower right
resp_keys_list = list(resp_keys.values())

##### Task Parameters
print(resp_keys)


##### cue object coordinates by shape
# unit length
unit = 3 # example: one unit is 3 visual degrees
unit_mult = 2 
hypot = np.sqrt( int( ((2*unit)**2) + (unit**2) ) ) # sqrt( (2*unit)^2 + unit^2 )
#1:[(x,y), ...]
cue_coord_dict = {1:[[-1*unit_mult*unit,unit_mult*unit],[0,unit],[unit_mult*unit,unit_mult*unit],[unit_mult*unit,-1*unit_mult*unit],[0,-1*unit],[-1*unit_mult*unit,-1*unit_mult*unit]], 2:[[-1*unit_mult*unit,unit_mult*unit],[0,unit],[unit_mult*unit,unit_mult*unit],[unit_mult*unit,-1*unit_mult*unit],[0,-1*unit],[-1*unit_mult*unit,-1*unit_mult*unit]],
                  3:[[-1*unit_mult*unit,unit_mult*unit],[unit_mult*unit,unit_mult*unit],[unit,0],[unit_mult*unit,-1*unit_mult*unit],[-1*unit_mult*unit,-1*unit_mult*unit],[-1*unit,0]], 4:[[-1*unit_mult*unit,unit_mult*unit],[unit_mult*unit,unit_mult*unit],[unit,0],[unit_mult*unit,-1*unit_mult*unit],[-1*unit_mult*unit,-1*unit_mult*unit],[-1*unit,0]],
                  5:[[-1*unit,unit_mult*unit],[unit,unit_mult*unit],[unit_mult*unit,0],[unit,-1*unit_mult*unit],[-1*unit,-1*unit_mult*unit],[-1*unit_mult*unit,0]], 6:[[-1*unit,unit_mult*unit],[unit,unit_mult*unit],[unit_mult*unit,0],[unit,-1*unit_mult*unit],[-1*unit,-1*unit_mult*unit],[-1*unit_mult*unit,0]]}
cue_seg_len_dict = {1:[hypot,hypot,int(2*2*unit),hypot,hypot,int(2*2*unit)], 2:[hypot,hypot,int(2*2*unit),hypot,hypot,int(2*2*unit)],
                    3:[int(2*2*unit),hypot,hypot,int(2*2*unit),hypot,hypot], 4:[int(2*2*unit),hypot,hypot,int(2*2*unit),hypot,hypot],
                    5:[int(2*unit),hypot,hypot,int(2*unit),hypot,hypot], 6:[int(2*unit),hypot,hypot,int(2*unit),hypot,hypot]}
#1:[num_segments_per_unit, dash_length, gap_length] ... 1-s, 2-m,  3-m, 4-l,  5-s, 6-l
cue_dash_dict = {1:[5, 0.16, 0.04], 2:[1, 0.8, 0.2], 
                 3:[1, 0.8, 0.2],   4:[2, 0.4, 0.1], 
                 5:[5, 0.16, 0.04], 6:[2, 0.4, 0.1]}
saturation = .7 # 0.0 - 1.0
brightness = .8 # 0.0 - 1.0
# blue=260, green=140, orange=20
cue_color_dict = {1:(265, saturation, brightness), 
                  2:(145, saturation, brightness), 
                  3:(25,  saturation, brightness), 
                  4:(265, saturation, brightness), 
                  5:(25,  saturation, brightness), 
                  6:(145, saturation, brightness)} 
cue_list = ['bvs', 'bgm', 'grm', 'gvl', 'hrs', 'hgl']
cue_dict = {}

for c_obj in range(1,7):
    c_pos_list = cue_coord_dict[c_obj] # will be 6 positions for each vertex
    c_len_list = cue_seg_len_dict[c_obj] # will be lengths of 6 line segments
    c_dash_list = cue_dash_dict[c_obj] # will be [num_segments_per_unit, dash_length, gap_length]
    obj_lines=[]
    # Create and add the line segments
    for c_idx, couplet in enumerate([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]):
        c_start = np.asarray(c_pos_list[couplet[0]], dtype=float)
        c_stop = np.asarray(c_pos_list[couplet[1]], dtype=float)
        num_units_in_side = int( c_len_list[c_idx] / unit ) # round down to integer, total number of units for this side
        num_segments = num_units_in_side * c_dash_list[0] # total number of segments for this side
        dash_length = c_dash_list[1] / num_units_in_side # dash length for each segment
        gap_length = c_dash_list[2] / num_units_in_side # gap length for each segment
        dx_dy_seg = (c_stop - c_start)
        if ((abs(dx_dy_seg[0]) > 0) & (abs(dx_dy_seg[1]) == 0)):
            rot_angle = 90 # horizontal is 90 or 270
        elif ((abs(dx_dy_seg[0]) == 0) & (abs(dx_dy_seg[1]) > 0)):
            rot_angle = 0 # vertical is 0 or 180
        else:
            rot_angle = (np.arctan(dx_dy_seg[1]/dx_dy_seg[0]) * 180/np.pi)*-1 + 90 # add  -1(angle)+90  b/c psychopy rotates CC instead of C
        if c_dash_list[0] == 1:
            # one straight line gets calculated differently
            # add half the gap to the start and end points AND if the segment is longer than a multiple of the unit, add half the difference to start and end
            c_start_xy = c_start + (dx_dy_seg * (c_dash_list[2]/2)) # example: (-3,6) + (6,0)*0.1 = (-2.4, 6)
            c_stop_xy = c_start + (dx_dy_seg * (c_dash_list[2]/2)) + (dx_dy_seg * c_dash_list[1]) # example: (-3,6) + (6,0)*0.1 + (6,0)*0.8 = (-3, 6) + (0.6, 0) + (4.8,0) = (2.4,6)
            c_pos = c_start + (dx_dy_seg * (c_dash_list[2]/2)) + (dx_dy_seg * (c_dash_list[1])/2) # add half of dash length to get center point
            size_h = c_len_list[c_idx] * c_dash_list[1]
            #print("pos: ", c_pos)
            #print("size y: ", size_h)
            line_segment = visual.rect.Rect(
                                        win=win,
                                        size=(.5, size_h),
                                        pos=c_pos,
                                        ori=rot_angle,
                                        units='deg',
                                        colorSpace='hsv',
                                        lineColor=cue_color_dict[c_obj],
                                        fillColor=cue_color_dict[c_obj],
                                        lineWidth=1
                                    )
            obj_lines.append(line_segment)
        else:
            dx_dy_seg_dash_len = dx_dy_seg * dash_length
            dx_dy_seg_gap_len = dx_dy_seg * gap_length
            # if segment is not a multiple of the unit, adjust start and end points
            for si in range(num_segments):
                c_start_xy = c_start + (si * dx_dy_seg_dash_len) + (si * dx_dy_seg_gap_len)
                c_stop_xy = c_start + ((1 + si) * dx_dy_seg_dash_len) + (si * dx_dy_seg_gap_len)
                c_pos = c_start + ((1 + si) * dx_dy_seg_dash_len) + (si * dx_dy_seg_gap_len) - (dx_dy_seg_dash_len/2) # subtract half of dash length to get center point
                size_h = (num_units_in_side*unit) * dash_length
                #print("pos: ", c_pos)
                #print("size y: ", size_h)
                # add new segment to list
                line_segment = visual.rect.Rect(
                                            win=win,
                                            size=(.5, size_h),
                                            pos=c_pos,
                                            ori=rot_angle,
                                            units='deg',
                                            colorSpace='hsv',
                                            lineColor=cue_color_dict[c_obj],
                                            fillColor=cue_color_dict[c_obj],
                                            lineWidth=1
                                        )
                obj_lines.append(line_segment)
    # add lines for current object to cue dict
    cue_dict[cue_list[int(c_obj-1)]] = obj_lines


##### Setup Cue stim objects
size=5
size_c=2
if subInfo['method'] != "MRI":
    size_c=5 # don't have to worry about cue obj size since I define it above

# set up context tone if audio was possible on this system
if len(modality_opts) > 1:
    context_tone_1 = sound.Sound(value=350, secs=1.5, octave=4, stereo=True, hamming=True, speaker='ab', volume=1.0, name='c1')
    context_tone_2 = sound.Sound(value=440, secs=1.5, octave=4, stereo=True, hamming=True, speaker='ab', volume=1.0, name='c2')
    context_tone_3 = sound.Sound(value=277, secs=1.5, octave=4, stereo=True, hamming=True, speaker='ab', volume=1.0, name='c3')
else:
    context_tone_1 = -1
    context_tone_2 = -1
    context_tone_3 = -1

# 220 Hz, 440 Hz, 660 Hz context_dict['shape'].play()
context_dict={ 'auditory': {'shape':context_tone_1, 
                            'color':context_tone_2, 
                            'outline':context_tone_3},
               'visual': {'shape':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-shape.png',name='shapeContext', units=visual_unit, pos=(0,0), size=size_c), 
                          'color':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-color.png',name='colorContext', units=visual_unit, pos=(0,0), size=size_c), 
                          'outline':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-outline.png',name='outlineContext', units=visual_unit, pos=(0,0), size=size_c)}}

if subInfo['method']=='MRI':
    cue_dict = {'bvs': [visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-1.png',name='dbb', units=visual_unit, pos=(0,0), size=size)],
                'bgm': [visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-2.png',name='dyi', units=visual_unit, pos=(0,0), size=size)],
                'grm': [visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-3.png',name='pri', units=visual_unit, pos=(0,0), size=size)],
                'gvl': [visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-4.png',name='pbo', units=visual_unit, pos=(0,0), size=size)],
                'hrs': [visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-5.png',name='hrb', units=visual_unit, pos=(0,0), size=size)], 
                'hgl': [visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-6.png',name='hyo', units=visual_unit, pos=(0,0), size=size)]}

shape_context_groups = [['bvs', 'bgm'],['grm', 'gvl'],['hrs', 'hgl']]
shape_context_cues = {'bvs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-1.png',name='dbb', units='norm', pos=(-0.6,0.2), size=(0.35,0.60)),
                    'bgm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-2.png',name='dyi', units='norm', pos=(-0.6,-0.4), size=(0.35,0.60)),
                    'grm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-3.png',name='pri', units='norm', pos=(0,0.2), size=(0.35,0.60)),
                    'gvl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-4.png',name='pbo', units='norm', pos=(0,-0.4), size=(0.35,0.60)),
                    'hrs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-5.png',name='hrb', units='norm', pos=(0.6,-0.4), size=(0.35,0.60)), 
                    'hgl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-6.png',name='hyo', units='norm', pos=(0.6,0.2), size=(0.35,0.60)),
                    'context':{'visual':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-shape.png',name='shapeContext', units='norm', pos=(0,0.2), size=(0.22,0.33)),
                                'auditory':context_tone_1}}

color_context_groups = [['grm', 'hrs'],['bgm', 'hgl'],['bvs', 'gvl']]
color_context_cues = {'bvs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-1.png',name='dbb', units='norm', pos=(0.6,0.2), size=(0.35,0.60)),
                    'bgm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-2.png',name='dyi', units='norm', pos=(0,-0.4), size=(0.35,0.60)),
                    'grm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-3.png',name='pri', units='norm', pos=(-0.6,0.2), size=(0.35,0.60)),
                    'gvl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-4.png',name='pbo', units='norm', pos=(0.6,-0.4), size=(0.35,0.60)),
                    'hrs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-5.png',name='hrb', units='norm', pos=(-0.6,-0.4), size=(0.35,0.60)), 
                    'hgl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-6.png',name='hyo', units='norm', pos=(0,0.2), size=(0.35,0.60)),
                    'context':{'visual':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-color.png',name='colorContext', units='norm', pos=(0,0.2), size=(0.22,0.33)),
                                'auditory':context_tone_2}}

outline_context_groups = [['gvl', 'hgl'],['bvs', 'hrs'],['bgm', 'grm']]
outline_context_cues = {'bvs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-1.png',name='dbb', units='norm', pos=(0,0.2), size=(0.35,0.60)),
                        'bgm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-2.png',name='dyi', units='norm', pos=(0.6,0.2), size=(0.35,0.60)),
                        'grm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-3.png',name='pri', units='norm', pos=(0.6,-0.4), size=(0.35,0.60)),
                        'gvl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-4.png',name='pbo', units='norm', pos=(-0.6,0.2), size=(0.35,0.60)),
                        'hrs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-5.png',name='hrb', units='norm', pos=(0,-0.4), size=(0.35,0.60)), 
                        'hgl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-6.png',name='hyo', units='norm', pos=(-0.6,-0.4), size=(0.35,0.60)),
                        'context':{'visual':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-outline.png',name='outlineContext', units='norm', pos=(0,0.2), size=(0.22,0.33)),
                                   'auditory':context_tone_3}}

#----------setup trial ordrs (switch and stay trials)---------
####  hiearchical cue trees
#                [ shape ]                       [ color ]                     [outline]
#               /    |    \                     /    |    \                   /    |    \ 
#              /     |     \                   /     |     \                 /     |     \ 
#       [ bow ] [hourglass] [hexagon]    [orance] [green] [purple]        [long] [short] [solid]
#          |         |        |              |      |       |              |       |       |
#        Task1     Task2     Task3         Task1  Task2   Task3          Task1   Task2   Task3
#
# Total 3 tasks
# Task 1 :  horizontal
# Task 2 :  vertical
# Task 3 :  diagonal
# 
# # 6 unique cues associated with 3 contexts
# 1 = [b]ow, hour[g]lass, [h]exagon
# 2 = [v]ilot, [g]reen, [r]ed/orange
# 3 = [m]ixed, [s]hort, [l]ong
context_list = ['shape', 'color', 'outline']
cue_list = ['bvs', 'bgm', 'grm', 'gvl', 'hrs', 'hgl']
task_list = ['T1', 'T2', 'T3']
#resp_list = ['yes','no']
resp_list = [1,2,3,4] # quadrant: # 1 = upper left   2 = upper right   3 = lower left   4 = lower right
#  num_4 | num_5
# -------|-------
#  num_1 | num_2

context_code_dict = {1:'shape', 2:'color', 3:'outline'}

cue_details_dict = {'bvs':{'shape':'bow', 'color':'violet', 'outline':'short'},
                    'bgm':{'shape':'bow', 'color':'green', 'outline':'mixed'},
                    'grm':{'shape':'hourglass', 'color':'red', 'outline':'mixed'},
                    'gvl':{'shape':'hourglass', 'color':'violet', 'outline':'long'},
                    'hrs':{'shape':'hexagon', 'color':'red', 'outline':'short'},
                    'hgl':{'shape':'hexagon', 'color':'green', 'outline':'long'}}

task_dict = {'shape':{'bvs': 'T1',
                    'bgm': 'T1',
                    'grm': 'T2',
                    'gvl': 'T2',
                    'hrs': 'T3', 
                    'hgl': 'T3'},
            'color':{'bvs': 'T3',
                    'bgm': 'T2',
                    'grm': 'T1',
                    'gvl': 'T3',
                    'hrs': 'T1', 
                    'hgl':'T2' },
            'outline':{'bvs': 'T2',
                    'bgm': 'T3',
                    'grm': 'T3',
                    'gvl': 'T1',
                    'hrs': 'T2', 
                    'hgl': 'T1'}}
task_inv_dict = {'T1':{'shape':['bvs','bgm'],
                       'color':['grm','hrs'],
                       'outline':['gvl','hgl']},
                'T2':{'shape':['grm','gvl'],
                       'color':['bgm','hgl'],
                       'outline':['bvs','hrs']},
                'T3':{'shape':['hrs','hgl'],
                       'color':['bvs','gvl'],
                       'outline':['bgm','grm']}}

# Task 3 changes for MRI vs EEG/Behavioral
if probe_subset:
    T3_dict = {'MRI':{1:'1', 2:'2', 3:'4', 4:'3'},
               'Other':{1:'num_5', 2:'num_2', 3:'num_4', 4:'num_1'}}
else:
    T3_dict = {'Other':{1:'num_2', 2:'num_1', 3:'num_5', 4:'num_4'},
                'MRI':{1:'2', 2:'3', 3:'1', 4:'4'}}

# T1=horiz  T2=vert  T3=diag
# 1 = upper left   2 = upper right   3 = lower left   4 = lower right
# 
#  num_4 | num_5     1 | 2
# -------|-------   ---|---
#  num_1 | num_2     3 | 4
# 
# stim_resp_dict[quad][task] ... correct key press
# # based on horiz, vertical, and clockwise
# stim_resp_dict = {1:{'T1':'num_5', 'T2':'num_1', 'T3':T3_dict['Other'][1]},   2:{'T1':'num_4', 'T2':'num_2', 'T3':T3_dict['Other'][2]},
#                   3:{'T1':'num_2', 'T2':'num_4', 'T3':T3_dict['Other'][3]},   4:{'T1':'num_1', 'T2':'num_5', 'T3':T3_dict['Other'][4]}}
#old version with diagonal as 3rd task
stim_resp_dict = {1:{'T1':'num_5', 'T2':'num_1', 'T3':'num_2'},
                  2:{'T1':'num_4', 'T2':'num_2', 'T3':'num_1'},
                  3:{'T1':'num_2', 'T2':'num_4', 'T3':'num_5'},
                  4:{'T1':'num_1', 'T2':'num_5', 'T3':'num_4'}}
if subInfo['method']=='MRI':
    # using 4 key pad
    #  4 | 1                 2 | 3 
    # ---|---   or flipped  ---|---
    #  3 | 2                 1 | 4 
    stim_resp_dict = {1:{'T1':'1', 'T2':'3', 'T3':T3_dict['MRI'][1]},
                      2:{'T1':'4', 'T2':'2', 'T3':T3_dict['MRI'][2]},
                      3:{'T1':'2', 'T2':'4', 'T3':T3_dict['MRI'][3]},
                      4:{'T1':'3', 'T2':'1', 'T3':T3_dict['MRI'][4]}}


laptop_resp_keys_list = ['ralt', 'comma', 'period', 'rctrl']
def laptop_key_press(key_name):
    key_dict = {'ralt':3, 'comma':4, 'period':1, 'rctrl':2}
    mod_key_name = key_dict[key_name]
    return mod_key_name

#########################
#### Timing and trial keeping variables
Total_trials = real_Total_trials - Number_of_initial_stays # set at 3 above
Stay_probability = float(subInfo['Stay_R'])
Switch_probability = 1-float(subInfo['Stay_R'])
Extra_dimension_switch_probability = 1-float(subInfo['IDS_R'])
Intra_dimension_switch_probability = float(subInfo['IDS_R'])
if Extra_dimension_switch_probability + Intra_dimension_switch_probability > 1: # making sure the IDS/EDS ratio make sense
    print('IDS and EDS ratio add up to more than 1')
    core.quit()
if Switch_probability >= .9: # same w switch ratio
    print('Switch Prob is No Bueno (equal to or above 90%)')
    core.quit()

#### setup number of trials for each trial type (EDS, IDS, stay)
if subInfo['method']=='MRI':
    step = 2 # MRI blocks have to be half as long
else:
    step = 1
    
# ---- since it is predefined now I can just load existing
trial_order_dict = {} # empty dict that block num will become key
trial_type_order_dict = {}
cue_order_dict = {}
task_order_dict = {}
probe_order_dict = {}

trial_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "trial_order.csv"))
trial_type_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "trial_type_order.csv"))
cue_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "trial_cue_order.csv"))
task_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "trial_task_order.csv"))
probe_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "probe_order.csv"))

#usable_blk_sequences_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "usable_sequences_v3.csv"))
#usable_blk_sequences = usable_blk_sequences_df[str(np.random.randint(3))] # only two usable sequences for now

blk_selections = np.arange(1, 101) #np.asarray(usable_blk_sequences) # np.arange(1, 101)
print(blk_selections)
np.random.shuffle(blk_selections)

if subInfo['method']!='MRI':
    step=1
else:
    step=2
    usable_trls = int(real_Total_trials - Number_of_initial_stays)
    midpt = usable_trls + Number_of_initial_stays # repetative I know, but want it to make sense to new user

for blk in range(0, int(Task_Parameters['n_blocks']), step):
    print("for block ", blk+1, ": grabbing column ", blk_selections[blk], " from pre-generated order")
    trial_order = trial_order_df[str(blk_selections[blk])]
    #print(trial_order)
    trial_type_order = trial_type_order_df[str(blk_selections[blk])]
    #print(trial_type_order)
    cue_order = cue_order_df[str(blk_selections[blk])]
    #print(cue_order)
    task_order = task_order_df[str(blk_selections[blk])]
    #print(task_order)
    probe_order = probe_order_df[str(blk_selections[blk])]
    #print(probe_order)
    
    if subInfo['method']!='MRI':
        trial_order_dict[int(blk+1)] = trial_order
        trial_type_order_dict[int(blk+1)] = trial_type_order
        cue_order_dict[int(blk+1)] = cue_order
        task_order_dict[int(blk+1)] = task_order
        probe_order_dict[int(blk+1)] = probe_order
    else:
        # odd first block
        trial_order_dict[int(blk+1)] = trial_order[:midpt]
        trial_type_order_dict[int(blk+1)] = trial_type_order[:midpt]
        cue_order_dict[int(blk+1)] = cue_order[:midpt]
        task_order_dict[int(blk+1)] = task_order[:midpt]
        probe_order_dict[int(blk+1)] = probe_order[:midpt]
        
        # even second block
        trial_order_dict[int(blk+2)] = np.concatenate( [ np.repeat(np.abs(trial_order[midpt]),Number_of_initial_stays), trial_order[midpt:] ] )
        trial_type_order_dict[int(blk+2)] = np.concatenate( [np.repeat('Stay',Number_of_initial_stays), trial_type_order[midpt:] ] )
        cue_order_dict[int(blk+2)] = np.concatenate( [np.repeat(cue_order[midpt],Number_of_initial_stays), cue_order[midpt:] ] )
        task_order_dict[int(blk+2)] = np.concatenate( [np.repeat(task_order[midpt],Number_of_initial_stays), task_order[midpt:] ] )
        probe_order_dict[int(blk+2)] = np.concatenate( [np.repeat(probe_order[midpt],Number_of_initial_stays), probe_order[midpt:] ] )
        

# trial_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "trial_order.csv"))
# trial_type_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "trial_type_order.csv"))
# if probe_subset:
#     probe_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "probe_order_subset.csv"))
# else:
#     probe_order_df = pd.read_csv(os.path.join(_thisDir, "trial_order", "probe_order_v1.csv"))
# even_blk_selections = np.arange(2, 101, 2) # since I know I generated 100 options
# odd_blk_selections = np.arange(1, 101, 2) # since I know I generated 100 options
# np.random.shuffle(even_blk_selections) # randomize order so I can select first X number in block loop below
# np.random.shuffle(odd_blk_selections) # randomize order so I can select first X number in block loop below
# for blk in range(int(Task_Parameters['n_blocks'])):
#     if subInfo['method']!='MRI':
#         if (blk+1)%2 == 0:
#             print("for block ", blk+1, ": grabbing column ", even_blk_selections[blk], " from pre-generated order")
#             trial_order = trial_order_df[str(even_blk_selections[blk])]
#             trial_type_order = trial_type_order_df[str(even_blk_selections[blk])]
#             probe_order = probe_order_df[str(even_blk_selections[blk])]
#         else:
#             print("for block ", blk+1, ": grabbing column ", odd_blk_selections[blk], " from pre-generated order")
#             trial_order = trial_order_df[str(odd_blk_selections[blk])]
#             trial_type_order = trial_type_order_df[str(odd_blk_selections[blk])]
#             probe_order = probe_order_df[str(odd_blk_selections[blk])]
#         trial_order_dict[int(blk+1)] = trial_order
#         trial_type_order_dict[int(blk+1)] = trial_type_order
#         probe_order_dict[int(blk+1)] = probe_order
#     else:
#         if (blk)%4 == 0:
#             for add_blk in [int(blk), int(blk+2)]:
#                 if add_blk == blk:
#                     print("for block ", add_blk+1, ": grabbing column ", even_blk_selections[blk], " from pre-generated order")
#                     trial_order = trial_order_df[str(even_blk_selections[blk])]
#                     trial_type_order = trial_type_order_df[str(even_blk_selections[blk])]
#                     probe_order = probe_order_df[str(even_blk_selections[blk])]
#                 else:
#                     print("for block ", add_blk+1, ": grabbing column ", odd_blk_selections[blk], " from pre-generated order")
#                     trial_order = trial_order_df[str(odd_blk_selections[blk])]
#                     trial_type_order = trial_type_order_df[str(odd_blk_selections[blk])]
#                     probe_order = probe_order_df[str(odd_blk_selections[blk])]
                
#                 print("trial order generate for blocks ", int(add_blk+1), " and ", int(add_blk+2))
#                 trial_order_dict[int(add_blk+1)] = trial_order[:int(Task_Parameters['n_trials'])]
#                 trial_order_dict[int(add_blk+2)] = np.concatenate( [ np.repeat(abs(trial_order[int(Task_Parameters['n_trials'])]),int(Number_of_initial_stays)), 
#                                                                 trial_order[int(Task_Parameters['n_trials']):] ] )
#                 # save out Stay | IDS | EDS as well
#                 trial_type_order_dict[int(add_blk+1)] = trial_type_order[:int(Task_Parameters['n_trials'])]
#                 trial_type_order_dict[int(add_blk+2)] = np.concatenate( [np.repeat(['Stay'],int(Number_of_initial_stays)),
#                                                                     trial_type_order[int(Task_Parameters['n_trials']):]] )
#                 # -- extra bit to calculate what probe combo we want
#                 probe_order_dict[int(add_blk+1)] = probe_order[:int(Task_Parameters['n_trials'])]
#                 probe_order_dict[int(add_blk+2)] = np.concatenate( [ [probe_order[int(Task_Parameters['n_trials'])], probe_order[int(Task_Parameters['n_trials'])]],
#                                                                 probe_order[int(Task_Parameters['n_trials']):] ] )

################################################################################################################################
##################################################   Set up EyeLink Tracker   ##################################################
if 'pupil' in subInfo['method']:
    # ---- code from Zexuan Niu
    # make initial connection with eye tracker
    if dummy_mode:
        el_tracker = pylink.EyeLink(None)
    else:
        try:
            el_tracker = pylink.EyeLink("100.1.1.2")
        except RuntimeError as error:
            print('ERROR:', error)
            core.quit()
            sys.exit()

    preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)    
    el_tracker.setOfflineMode()
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode

    if not dummy_mode:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        # print out some version info in the shell
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))
    # File and Link data control
    # what eye events to save in the EDF file, include everything by default
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    # what eye events to make available over the link, include everything by default
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
    # what sample data to save in the EDF data file and to make available
    # over the link, include the 'HTARGET' flag to save head target sticker
    # data for supported eye trackers
    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)    
    el_tracker.sendCommand("calibration_type = HV3")
    # Set a gamepad button to accept calibration/drift check target
    # You need a supported gamepad/button box that is connected to the Host PC
    #el_tracker.sendCommand("a 'accept_target_fixation'")
    #el_tracker.sendCommand("calibration_type = HV9")
    # Set a gamepad button to accept calibration/drift check target
    # You need a supported gamepad/button box that is connected to the Host PC
    #el_tracker.sendCommand("a 'accept_target_fixation'")


def set_up_calibration():
    global win
    global el_tracker
    global scn_width
    global scn_height
    
    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)
    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    print(genv)  # print out the version number of the CoreGraphics library
    foreground_color = (-1, -1, -1)
    background_color = win.color
    genv.setCalibrationColors(foreground_color, background_color)
    genv.setTargetType('circle')
    genv.setTargetSize(10)
    genv.setCalibrationSounds('off', 'off', 'off')
    pylink.openGraphicsEx(genv)
    

def abort_trial():
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    #clear_screen(win)
    win.flip()
    # Send a message to clear the Data Viewer screen
    bgcolor_RGB = (116, 116, 116)
    el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR

def Close():
    global win
    el_tracker = pylink.getEYELINK()
    if el_tracker.isConnected():
    # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()
        
        # Put tracker in Offline mode
        el_tracker.setOfflineMode()
        
        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)
        
        # Close the edf data file on the Host
        el_tracker.closeDataFile()
        
        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = edf_file
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)
            
    win.close()
    core.quit()
    sys.exit()



# # ----- maximize window now that task is mostly ready
# win.winHandle.set_fullscreen(True)
# win.winHandle.maximize()

################################################################################################################################
##################################################   Define Task functions   ##################################################
#This will be for CSV initiation
def makeCSV(filename, thistrialDict, trial_num):
    pd.DataFrame(thistrialDict).to_csv(filename, index=False)

# prepare practice Task_Parameters = {'n_blocks': 8, 'n_trials': 51, 'n_prac_trials': 15}
def prepare_prac_trials(i_block, Task_Parameters, resp_keys):
    # set up output dict
    rand_context_order = [1,2,3] # 1=shape, 2=color, 3=outline
    np.random.shuffle(rand_context_order) # shuffle to make random
    out_dict = {'block': np.zeros(Task_Parameters['n_prac_trials'])+i_block,
                'trial': np.linspace(1,Task_Parameters['n_prac_trials'],num=Task_Parameters['n_prac_trials'],endpoint=True),
                'context_code': np.concatenate( [np.repeat(rand_context_order[0], int(Task_Parameters['n_prac_trials']/3)), np.repeat(rand_context_order[1], int(Task_Parameters['n_prac_trials']/3)), np.repeat(rand_context_order[2], int(Task_Parameters['n_prac_trials']/3))] ),
                'context': [],  'cue': [],  'shape': [],  'color': [],  'outline': [],
                'task': ['T1','T1','T3','T3','T2','T2', 'T3','T3','T2','T2','T1','T1', 'T2','T2','T1','T1','T3','T3'],  
                'probe_loc': [],  
                'hierarhical_switch_type': ['Stay','Stay','IDS','Stay','IDS','Stay', 'EDS','Stay','IDS','Stay','IDS','Stay', 'EDS','Stay','IDS','Stay','IDS','Stay'],
                'correct_response': np.zeros(Task_Parameters['n_prac_trials']),   'correct_response_str': [],
                'subject_response': np.ones(Task_Parameters['n_prac_trials'])*np.nan,  'subject_correct': np.zeros(Task_Parameters['n_prac_trials']),
                'response_time': np.ones(Task_Parameters['n_prac_trials'])*np.nan,
                'delay':np.zeros(Task_Parameters['n_prac_trials']), 'ITI':np.zeros(Task_Parameters['n_prac_trials'])}
    
    for i in range(Task_Parameters['n_prac_trials']):
        # these will always be the same, but just want them saved for ease
        out_dict['context'].append( context_code_dict[out_dict['context_code'][i]] ) # 1=shape, 2=color, 3=outline
        
        # determine cue object to present based on trial order predefined above ... task_inv_dict [T1] [shape] = [bvs, dyi]
        out_dict['cue'].append(task_inv_dict[out_dict['task'][i]][out_dict['context'][i]][np.random.choice([0,1])])
        if i > 0:
            while out_dict['cue'][i] == out_dict['cue'][int(i-1)]:
                out_dict['cue'][i] = task_inv_dict[out_dict['task'][i]][out_dict['context'][i]][np.random.choice([0,1])] # make sure cue is different from prev trial
        
        # record what correct resp would be  stim_resp_dict [1] [T1] = 'num_1' ... resp_keys 'num_1' | 'num_2' | 'num_4' | 'num_5'
        out_dict['probe_loc'].append( np.random.randint(1,5) )
        out_dict['correct_response_str'].append(stim_resp_dict[out_dict['probe_loc'][i]][out_dict['task'][i]]) # puts in term of quad 
        out_dict['correct_response'][i] = int(stim_resp_dict[out_dict['probe_loc'][i]][out_dict['task'][i]][-1:]) # puts in term of key to press 1 or 2 or 4 or 5
        
        # add cue  shape, color, and outline
        out_dict['shape'].append( cue_details_dict[out_dict['cue'][i]]['shape'] )
        out_dict['color'].append( cue_details_dict[out_dict['cue'][i]]['color'] )
        out_dict['outline'].append( cue_details_dict[out_dict['cue'][i]]['outline'] )

    #print(out_dict)
    return out_dict


# prepare task trials
# def prepare_block_trials(i_block, trial_order_dict, trial_type_order_dict, Task_Parameters, cue_list, task_list, resp_keys):
#     # set up output dict
#     out_dict = {'block': np.zeros(Task_Parameters['n_trials'])+i_block,
#                 'trial': np.linspace(1,Task_Parameters['n_trials'],num=Task_Parameters['n_trials'],endpoint=True),
#                 'cue': [], 
#                 'context': [],   'shape': [],   'color': [],   'outline': [],
#                 'probe_loc': [],  
#                 'task': [],  'hierarhical_switch_type': trial_type_order_dict[i_block],
#                 'correct_response': np.zeros(Task_Parameters['n_trials']),   'correct_response_str': [],
#                 'subject_response': np.ones(Task_Parameters['n_trials'])*np.nan,  'subject_correct': np.zeros(Task_Parameters['n_trials']),
#                 'response_time': np.ones(Task_Parameters['n_trials'])*np.nan,
#                 'delay':np.zeros(Task_Parameters['n_trials']), 'ITI':np.zeros(Task_Parameters['n_trials'])}
#     c_trial_order = trial_order_dict[i_block] # more detailed trial order information, which context on each trial
    
#     # T1: 1-4 ... T2: 5-8 ... T3: 9-12
#     # PL1: 1,5,9 ... PL2: 2,6,10 ... PL3: 3,7,11 ... PL4: 4,8, 12     probe_order_dict[block]
#     for i in range(Task_Parameters['n_trials']):
#         # these will always be the same, but just want them saved for ease
#         out_dict['context'].append( context_code_dict[abs(c_trial_order[i])] ) # 1=shape, 2=color, 3=outline
        
#         # grab task we pre-defined
#         # T1: 1-4 ... T2: 5-8 ... T3: 9-12
#         # PL1: 1,5,9 ... PL2: 2,6,10 ... PL3: 3,7,11 ... PL4: 4,8,12
#         # R1: 2,7,12 ...  R2: 1,8,11 ...  R3: 4,5,10 ...  R4: 3,6,9
#         if probe_order_dict[i_block][i] < 5:
#             out_dict['task'].append( 'T1' )
#         elif probe_order_dict[i_block][i] < 9:
#             out_dict['task'].append( 'T2' )
#         else:
#             out_dict['task'].append( 'T3' )
            
#         # generate cue based on task ... task_inv_dict[task][context] = [cue1, cue2]
#         new_cue = task_inv_dict[out_dict['task'][i]][out_dict['context'][i]][np.random.choice([0,1])]
#         if i > 0:
#             # make sure exact cue doesn't repeat
#             while new_cue==out_dict['cue'][int(i-1)]:
#                 new_cue = task_inv_dict[out_dict['task'][i]][out_dict['context'][i]][np.random.choice([0,1])] # make sure new cue is not the same as previous cue
#         out_dict['cue'].append( new_cue )
                
#         # grab probe location we pre-defined
#         # T1: 1-4 ... T2: 5-8 ... T3: 9-12
#         # PL1: 1,5,9 ... PL2: 2,6,10 ... PL3: 3,7,11 ... PL4: 4,8,12
#         # R1: 2,7,12 ...  R2: 1,8,11 ...  R3: 4,5,10 ...  R4: 3,6,9
#         if probe_order_dict[i_block][i] in [1,5,9]:
#             out_dict['probe_loc'].append( 1 )
#         elif probe_order_dict[i_block][i] in [2,6,10]:
#             out_dict['probe_loc'].append( 2 )
#         elif probe_order_dict[i_block][i] in [3,7,11]:
#             out_dict['probe_loc'].append( 3 )
#         else:
#             out_dict['probe_loc'].append( 4 )
        
        
#         # record what image type (face|scene) and filename
#         #out_dict['probe_loc'].append( np.random.randint(1,5) )
#         out_dict['correct_response_str'].append(stim_resp_dict[out_dict['probe_loc'][i]][out_dict['task'][i]]) # puts in term of quad 
#         out_dict['correct_response'][i] = int(stim_resp_dict[out_dict['probe_loc'][i]][out_dict['task'][i]][-1:]) # puts in term of key to press 1 or 2 or 4 or 5
        
#         # add cue  shape, color, and outline
#         out_dict['shape'].append( cue_details_dict[out_dict['cue'][i]]['shape'] )
#         out_dict['color'].append( cue_details_dict[out_dict['cue'][i]]['color'] )
#         out_dict['outline'].append( cue_details_dict[out_dict['cue'][i]]['outline'] )
    
#         # --- new way to define task and probe_lco
#         # out_dict['task'][i] = [0][probe_order_dict[i_block][i]]
#         # out_dict['probe_loc'][i] = [1][probe_order_dict[i_block][i]]
#     #print(out_dict)
#     return out_dict

def prepare_block_trials(i_block, trial_order_dict, trial_type_order_dict, Task_Parameters, cue_list, task_list, resp_keys):
    # set up output dict
    out_dict = {'block': np.zeros(Task_Parameters['n_trials'])+i_block,
                'trial': np.linspace(1,Task_Parameters['n_trials'],num=Task_Parameters['n_trials'],endpoint=True),
                'cue': cue_order_dict[i_block], 
                'context': [],   'shape': [],   'color': [],   'outline': [],
                'probe_loc': probe_order_dict[i_block],  
                'task': task_order_dict[i_block],  'hierarhical_switch_type': trial_type_order_dict[i_block],
                'correct_response': np.zeros(Task_Parameters['n_trials']),   'correct_response_str': [],
                'subject_response': np.ones(Task_Parameters['n_trials'])*np.nan,  'subject_correct': np.zeros(Task_Parameters['n_trials']),
                'response_time': np.ones(Task_Parameters['n_trials'])*np.nan,
                'delay':np.zeros(Task_Parameters['n_trials']), 'ITI':np.zeros(Task_Parameters['n_trials'])}
    c_trial_order = trial_order_dict[i_block] # more detailed trial order information, which context on each trial
    
    # T1: 1-4 ... T2: 5-8 ... T3: 9-12
    # PL1: 1,5,9 ... PL2: 2,6,10 ... PL3: 3,7,11 ... PL4: 4,8, 12     probe_order_dict[block]
    for i in range(Task_Parameters['n_trials']):
        # these will always be the same, but just want them saved for ease
        out_dict['context'].append( context_code_dict[np.abs(c_trial_order[i])] ) # 1=shape, 2=color, 3=outline
        
        # record what image type (face|scene) and filename
        #out_dict['probe_loc'].append( np.random.randint(1,5) )
        out_dict['correct_response_str'].append(stim_resp_dict[out_dict['probe_loc'][i]][out_dict['task'][i]]) # puts in term of quad 
        out_dict['correct_response'][i] = int(stim_resp_dict[out_dict['probe_loc'][i]][out_dict['task'][i]][-1:]) # puts in term of key to press 1 or 2 or 4 or 5
        
        # add cue  shape, color, and outline
        out_dict['shape'].append( cue_details_dict[out_dict['cue'][i]]['shape'] )
        out_dict['color'].append( cue_details_dict[out_dict['cue'][i]]['color'] )
        out_dict['outline'].append( cue_details_dict[out_dict['cue'][i]]['outline'] )
    
        # --- new way to define task and probe_lco
        # out_dict['task'][i] = [0][probe_order_dict[i_block][i]]
        # out_dict['probe_loc'][i] = [1][probe_order_dict[i_block][i]]
    #print(out_dict)
    return out_dict


def disp_block_text(EndofBlock_Text, cur_block_acc, lastblock, max_time):
    """
    Inputs: 
      EndofBlock_Text: object ... psychopy text object that we modify and present to display text in between blocks
      cur_block_acc: numeric ... indicates the accuracy on the last block
      lastblock: true or false ... indicates if we want to display text associated with the last block (aka no accuracy)
      max_time: will be infinity when actually running the experiment, 1 if testing code (auto moves on without waiting long)
    
    Displays text inbetween each block and changes text based on if the block is a task or practice block AND if it is the last block in the experiment
    """
    if lastblock:
        EndofBlock_Text.text = "Task complete: please press any key to end."
    else: 
        if cur_block_acc > 0:
            EndofBlock_Text.text = "End of trial block: please take a short break or press any key to continue.\n\n\nAccuracy for last block was " + str(cur_block_acc) + "%"
        else:
            EndofBlock_Text.text = "End of trial block: please take a short break or press any key to continue."
    EndofBlock_Text.draw()
    win.flip()
    core.wait(1)
    event.waitKeys(maxWait=max_time)
    #nothing to return

def check_prac_task_choice(textobj,repeat_instructions, do_task):
    """
    Inputs: 
      textobj: object ... psychopy text object that we modify and present to display text after instructions and before either practice or task
      repeat_instructions: true or false ... if true, starts instruction screens over, if false, move on to either practice or task
      do_task: numeric, 0 or 1 ... 0 means do practice, 1 means do task
      
    Displays text inbetween each block and changes text based on if the block is a task or practice block AND if it is the last block in the experiment
    """
    # check if they want to repeat instructions or move onto task
    textobj.draw()
    win.flip()
    keys = event.waitKeys(keyList=['p','y','n','escape'])
    for key in keys:
        if key=="y":
            repeat_instructions = False
            do_task = 1
        elif key == "p":
            repeat_instructions = False
            do_task = 0
        elif key == 'escape':
            win.close()
            core.quit()
                
    return repeat_instructions, do_task



def make_ITI(ITI_list, trial_n):
        #ITI=np.random.choice([1,2,3,4,5,6,7,8,9,10],1,p=[(.7/9),(.7/9),(.7/9),.3,(.7/9),(.7/9),(.7/9),(.7/9),(.7/9),(.7/9)])[0] # averages to around 4 seconds?
        ITI=ITI_list[trial_n]
        return ITI

def draw_ITI(CP_Trl_Durs, refresh_rate, fixation, fixation_gray):
    for frame in range(int( (refresh_rate * (CP_Trl_Durs['ITI'][np.random.randint(0,len(CP_Trl_Durs['ITI']))] - 0.75) ))):
        fixation_gray.draw()
        win.flip()
    for frame in range(int(refresh_rate*0.75)):
        # change back to the black fixation cue for the last 0.75 seconds so they know not to blink
        fixation.draw()
        win.flip()


#### ----------------------------------------------------------------------------
#### FUNCTIONS FOR TASK INSTRUCTIONS
def disp_instruct_choice(win, instruct_dict, associated_img_dict, do_task):
    """
    Inputs: 
        win: window object ... to draw instructions to
        instruct_dict: dictionary ... defines instruction TEXT for each screen
        associated_img_dict: dictionary ... defines potential stimuli to present on each instruct screen
        do_task: int ... 0=practice, 1=task, 2=end code
        
    You can move forward with [right arrow] and backwards with [left arrow]
    Display stimuli examples by hitting space bar
    """
    repeat_instructions=False
    # -- set up text object for displaying instructions
    Instruction_Text = visual.TextStim(win=win, name='InstructionsText', text=u'', units='norm', pos=(0,0), height=0.06, ori=0, color=u'black', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0, wrapWidth=1.95)
    # set up a couple screen num varialbes for loops
    max_screen = np.max(np.asarray(list(instruct_dict.keys())))
    iscreen_num = 0
    sound_played = False
    # ---- loop through instruct screens
    while iscreen_num <= int(max_screen):
        # -- display instruction text for current screen
        Instruction_Text.text = instruct_dict[iscreen_num] # set up to display current screen text
        Instruction_Text.draw()
        # -- display image (if assinged)
        if associated_img_dict[iscreen_num] != []:
            cur_cue_dict = associated_img_dict[iscreen_num]
            for cur_cue in cur_cue_dict.keys():
                if cur_cue=="context":
                    if context_modality=='visual':
                        cur_cue_dict['context'][context_modality].draw()
                    else:
                        if not(sound_played):
                            cur_cue_dict['context'][context_modality].play(when=win.getFutureFlipTime(clock='ptb'))
                            sound_played=True
                else:
                    cur_cue_dict[cur_cue].draw()
        win.flip()
        if sound_played:
            core.wait(1.5)
            cur_cue_dict['context'][context_modality].stop()
        # -- check key press to move forward or backward in screen list
        keys = event.waitKeys(keyList=['space','right','left','q','escape','p','y','i',bottom_quads['right'],bottom_quads['left']])
        for key in keys:
            print(key)
            # -- move forward
            if key=="right":
                iscreen_num += 1 
                sound_played = False
            # -- move backward
            elif key == "left":
                if iscreen_num > 0:
                    iscreen_num -= 1
                    sound_played = False
            elif key==bottom_quads['right']:
                iscreen_num += 1 
                sound_played = False
            elif key==bottom_quads['left']:
                if iscreen_num > 0:
                    iscreen_num -= 1
                    sound_played = False
            elif ((key == 'q')|(key == 'escape')):
                win.close()
                core.quit() # make sure we can force quit the code if needed
            # -- start task
            elif (key=="y") & (iscreen_num == int(max_screen)):
                do_task = 1
                iscreen_num += 1
            # -- start practice
            elif (key=="p") & (iscreen_num == int(max_screen)):
                do_task = 0
                iscreen_num += 1
            # -- repeat instructions
            elif (key=="i") & (iscreen_num == int(max_screen)):
                if do_task != -1:
                    repeat_instructions = True
                    iscreen_num += 1
                    sound_played = False
        
        # don't allow for moving past last screen unless you hit [p] for practice or [y] for task
        if repeat_instructions & (do_task==0):
            print("repeat instructions")
            # special case where we let them go back through instructions after practice
            do_task = -1
        elif (do_task < 0) & (iscreen_num>max_screen):
            print("need to pick between task and practice")
            iscreen_num = max_screen # force the participant/experimentor to pick between practice and task before moving on from instructions
        elif (do_task == 0) & (iscreen_num>max_screen):
            print("practice finished, need to pick between task and practice")
            # if they finished practice, force p or y
            if (key!="y") & (key!="p"):
                iscreen_num = max_screen
    
    return do_task


#################################################################################################################################
####################################################### Give Instruction ########################################################
if context_modality=="visual":
    c_text0 = "see 4 pacman objects presented inside a complex object"
    c_text3 = "4 pacman objects you see"
    context_text = "the fixation point is surrounded by pacman objects oriented this way"
else:
    c_text0 = "hear a tone in addition to seeing an object"
    c_text3 = "tone you hear"
    context_text = "you hear a tone in this octave"

if subInfo['method']=="MRI":
    box_size = (2,2)
    ball_size = 0.2
    cue_size = 1.5
    pos_shrink = 0.4
else:
    box_size = (5,5)
    ball_size = 0.5
    cue_size = 3.5
    pos_shrink = 1
    
instruct_trial = {'cue_box' : visual.rect.Rect(win=win, name='cueB', size=box_size, pos=(-8*pos_shrink,2*pos_shrink), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
                'ex_cue' : visual.ImageStim(win=win, image=os.getcwd()+f"/cue_objects/cue-{np.random.randint(1,7)}.png", name='ex_cue', units=visual_unit, pos=(-8*pos_shrink,2*pos_shrink), size=cue_size),
                'context':{'visual':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-shape.png',name='shapeContext', units=visual_unit, pos=(-8*pos_shrink,2*pos_shrink), size=4*(cue_size/7)),
                            'auditory':context_tone_1},
                'delay1_box' : visual.rect.Rect(win=win, name='d1B', size=box_size, pos=(-4*pos_shrink,1*pos_shrink), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
                'probe_box' : visual.rect.Rect(win=win, name='pB', size=box_size, pos=(0,0), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
                'ex_probe' : visual.Circle(win, name='ex_probe', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(np.random.choice([-1.5*pos_shrink,1.5*pos_shrink]), np.random.choice([-1.5*pos_shrink,1.5*pos_shrink])), size=1.0, opacity=0.5),
                'delay2_box' : visual.rect.Rect(win=win, name='d2B', size=box_size, pos=(4*pos_shrink,-1*pos_shrink), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
                'feed_box' : visual.rect.Rect(win=win, name='feedB', size=box_size, pos=(8*pos_shrink,-2*pos_shrink), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
                'ex_feed' : visual.TextStim(win=win, name='corr', text='+', bold=True, units='deg', pos=(8*pos_shrink, -1.75*pos_shrink), font=u'Times', height=2.5*pos_shrink, color='black')}

tmp = {'shape':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-shape.png',name='shapeContext', units='norm', pos=(-0.6,0.25), size=(0.22,0.33)), 
        'color':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-color.png',name='colorContext', units='norm', pos=(0,0.25), size=(0.22,0.33)), 
        'outline':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-outline.png',name='outlineContext', units='norm', pos=(0.6,0.25), size=(0.22,0.33)),
        'bvs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-1.png',name='dbb', units='norm', pos=(-0.75,-0.5), size=(0.28,0.50)),
        'bgm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-2.png',name='dyi', units='norm', pos=(-0.45,-0.5), size=(0.28,0.50)),
        'grm': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-3.png',name='pri', units='norm', pos=(-0.15,-0.5), size=(0.28,0.50)),
        'gvl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-4.png',name='pbo', units='norm', pos=(0.15,-0.5), size=(0.28,0.50)),
        'hrs': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-5.png',name='hrb', units='norm', pos=(0.45,-0.5), size=(0.28,0.50)), 
        'hgl': visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/cue-6.png',name='hyo', units='norm', pos=(0.75,-0.5), size=(0.28,0.50))}
 
# instruct_trial = {'cue_box' : visual.rect.Rect(win=win, name='cueB', size=(5, 5), pos=(-9,2), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
#                   'ex_cue' : visual.ImageStim(win=win, image=os.getcwd()+f"/cue_objects/cue-{np.random.randint(1,7)}.png", name='ex_cue', units=visual_unit, pos=(-9,2), size=3.5),
#                   'context':{'visual':visual.ImageStim(win=win, image=os.getcwd()+'/cue_objects/context-shape.png',name='shapeContext', units=visual_unit, pos=(-9,2), size=2),
#                                 'auditory':context_tone_1},
#                   'delay1_box' : visual.rect.Rect(win=win, name='d1B', size=(5, 5), pos=(-4.5,1), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
#                   'probe_box' : visual.rect.Rect(win=win, name='pB', size=(5, 5), pos=(0,0), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
#                   'ex_probe' : visual.Circle(win, name='ex_probe', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(np.random.choice([-1.5,1.5]), np.random.choice([-1.5,1.5])), size=1.0, opacity=0.5),
#                   'delay2_box' : visual.rect.Rect(win=win, name='d2B', size=(5, 5), pos=(4.5,-1), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
#                   'feed_box' : visual.rect.Rect(win=win, name='feedB', size=(5, 5), pos=(9,-2), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5),
#                   'ex_feed' : visual.TextStim(win=win, name='corr', text='+', bold=True, units='deg', pos=(9, -1.85), font=u'Times', height=1.5, color='black')}
if probe_subset:
    instruct_trial2 = {'horz_box' : visual.rect.Rect(win=win, name='horz_box', size=(5, 5), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(-8,4)),
                    'horz_ball' : visual.Circle(win, name='horz_ball', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-9.5,5.5), size=1.0, opacity=0.5),
                    'horz_ball_R' : visual.Circle(win, name='horz_ball_R', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-6.5,5.5), size=1.0, opacity=0.1),
                    'vert_box' : visual.rect.Rect(win=win, name='vert_box', size=(5, 5), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(0,4)),
                    'vert_ball' : visual.Circle(win, name='vert_ball', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-1.5,5.5), size=1.0, opacity=0.5),
                    'vert_ball_R' : visual.Circle(win, name='vert_ball_R', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-1.5,2.5), size=1.0, opacity=0.1),
                    'clock_box' : visual.rect.Rect(win=win, name='clock_box', size=(5, 5), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(8,4)),
                    'clock_ball' : visual.Circle(win, name='clock_ball', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(9.5,5.5), size=1.0, opacity=0.5),
                    'clock_ball_R' : visual.Circle(win, name='clock_ball_R', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(9.5,2.5), size=1.0, opacity=0.1),
                    'horz_box2' : visual.rect.Rect(win=win, name='horz_box2', size=(5, 5), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(-8,-2)),
                    'horz_ball2' : visual.Circle(win, name='horz_ball2', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-6.5,-3.5), size=1.0, opacity=0.5),
                    'horz_ball_R2' : visual.Circle(win, name='horz_ball_R2', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-9.5,-3.5), size=1.0, opacity=0.1),
                    'vert_box2' : visual.rect.Rect(win=win, name='vert_box2', size=(5, 5), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(0,-2)),
                    'vert_ball2' : visual.Circle(win, name='vert_ball2', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(1.5,-3.5), size=1.0, opacity=0.5),
                    'vert_ball_R2' : visual.Circle(win, name='vert_ball_R2', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(1.5,-0.5), size=1.0, opacity=0.1),
                    'clock_box2' : visual.rect.Rect(win=win, name='clock_box2', size=(5, 5), units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(8,-2)),
                    'clock_ball2' : visual.Circle(win, name='clock_ball2', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(6.5,-0.5), size=1.0, opacity=0.5),
                    'clock_ball_R2' : visual.Circle(win, name='clock_ball_R2', radius=0.5, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(9.5,-0.5), size=1.0, opacity=0.1)}
else:
    instruct_trial2 = {'horz_box' : visual.rect.Rect(win=win, name='horz_box', size=box_size, units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(-8*pos_shrink,4*pos_shrink)),
                    'horz_ball' : visual.Circle(win, name='horz_ball', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-9.5*pos_shrink,5.5*pos_shrink), size=1.0, opacity=0.5),
                    'horz_ball_R' : visual.Circle(win, name='horz_ball_R', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-6.5*pos_shrink,5.5*pos_shrink), size=1.0, opacity=0.1),
                    'vert_box' : visual.rect.Rect(win=win, name='vert_box', size=box_size, units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(0*pos_shrink,4*pos_shrink)),
                    'vert_ball' : visual.Circle(win, name='vert_ball', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-1.5*pos_shrink,5.5*pos_shrink), size=1.0, opacity=0.5),
                    'vert_ball_R' : visual.Circle(win, name='vert_ball_R', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-1.5*pos_shrink,2.5*pos_shrink), size=1.0, opacity=0.1),
                    'clock_box' : visual.rect.Rect(win=win, name='clock_box', size=box_size, units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(8*pos_shrink,4*pos_shrink)),
                    'clock_ball' : visual.Circle(win, name='clock_ball', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(6.5*pos_shrink,5.5*pos_shrink), size=1.0, opacity=0.5),
                    'clock_ball_R' : visual.Circle(win, name='clock_ball_R', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(9.5*pos_shrink,2.5*pos_shrink), size=1.0, opacity=0.1),
                    'horz_box2' : visual.rect.Rect(win=win, name='horz_box2', size=box_size, units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(-8*pos_shrink,-2*pos_shrink)),
                    'horz_ball2' : visual.Circle(win, name='horz_ball2', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-6.5*pos_shrink,-3.5*pos_shrink), size=1.0, opacity=0.5),
                    'horz_ball_R2' : visual.Circle(win, name='horz_ball_R2', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-9.5*pos_shrink,-3.5*pos_shrink), size=1.0, opacity=0.1),
                    'vert_box2' : visual.rect.Rect(win=win, name='vert_box2', size=box_size, units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(0*pos_shrink,-2*pos_shrink)),
                    'vert_ball2' : visual.Circle(win, name='vert_ball2', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(1.5*pos_shrink,-3.5*pos_shrink), size=1.0, opacity=0.5),
                    'vert_ball_R2' : visual.Circle(win, name='vert_ball_R2', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(1.5*pos_shrink,-0.5*pos_shrink), size=1.0, opacity=0.1),
                    'clock_box2' : visual.rect.Rect(win=win, name='clock_box2', size=box_size, units=visual_unit, colorSpace='rgb', lineColor='black', fillColor='grey', lineWidth=5, pos=(8*pos_shrink,-2*pos_shrink)),
                    'clock_ball2' : visual.Circle(win, name='clock_ball2', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(9.5*pos_shrink,-3.5*pos_shrink), size=1.0, opacity=0.5),
                    'clock_ball_R2' : visual.Circle(win, name='clock_ball_R2', radius=ball_size, units=visual_unit, lineWidth=1, lineColor=None, fillColor='black', colorSpace='rgb', pos=(6.5*pos_shrink,-0.5*pos_shrink), size=1.0, opacity=0.1)}
if probe_subset:
    instruct_dict = {0: f"\nDuring this task you will {c_text0}, then, on a following screen, you will see a ball in one of four quadrants, at which point you will be expected to perform one of three tasks:\nHorizontal task,   Vertical Task,   Clockwise Task\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nWhat task to perform depends on the first screen (we will cover that after discussing that 3 tasks in more detail).\n\n\nPress the right arrow key to move on to the next screen",
                    1: f"For the Horizontal Task (far left), you should report what quadrant the ball would move to if it was moved horizontally\nFor the Vertical Task (middle), you should report what quadrant the ball would move to if it was moved vertically \nFor the Clockwise Task (far right), you should report what quadrant the ball would move to if it was moved clockwise \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nIn order for the correct response to be recorded, note that {resp_instructions}. \n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    2: f"\n\nMoving on to how to figure out what task to perform. This depends on the {c_text3} at the start of the trial in addition to the complex object also presented at the start of the trial.\n\nThe {c_text3} determines what feature you should focus on in the complex object that is co-presented. This complex object is composed of 3 features: Shape, Color, and Outline pattern. We will go through these one by one. \nShape\t\t\tColor\t\t\tOutline\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    3: f"\n\n\nIf {context_text}, then SHAPE matters! \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    4: f'\n\n\n    BOW = TASK 1 ( horizontal )         HOURGLASS = TASK 2 ( vertical )      HEXAGON = Task 3 ( clockwise ) \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen',
                    5: f"\n\n\nIf {context_text}, then COLOR matters! \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    6: f'\n\n\n ORANGE = TASK 1 ( horizontal )      GREEN = TASK 2 ( vertical )      PURPLE = Task 3 ( clockwise ) \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen',
                    7: f"\n\n\nIf {context_text}, then OUTLINE matters! \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    8: f'\n\n\n LONG DASH = TASK 1 ( horizontal )      SHORT DASH = TASK 2 ( vertical )      NO DASH = Task 3 ( clockwise ) \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen',
                    9: 'Please wait for the experimentor to start either the practice block or the full task'}
else:
    instruct_dict = {0: f"\nDuring this task you will {c_text0}, then, on a following screen, you will see a ball in one of four quadrants, at which point you will be expected to perform one of three tasks:\nHorizontal task,   Vertical Task,   Diagonal Task\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nWhat task to perform depends on the first screen (we will cover that after discussing that 3 tasks in more detail).\n\n\nPress the right arrow key to move on to the next screen",
                    1: f"For the Horizontal Task (far left), you should report what quadrant the ball would move to if it was moved horizontally\nFor the Vertical Task (middle), you should report what quadrant the ball would move to if it was moved vertically \nFor the Diagonal Task (far right), you should report what quadrant the ball would move to if it was moved diagonally \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nIn order for the correct response to be recorded, note that {resp_instructions}. \n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    2: f"Moving on to how to figure out what task to perform. This depends on the {c_text3} at the start of the trial in addition to the complex object also presented at the start of the trial.\n\nThe {c_text3} determines what feature you should focus on in the complex object that is co-presented. This complex object is composed of 3 features: Shape, Color, and Outline pattern. We will go through these one by one. \nShape\t\t\tColor\t\t\tOutline\n\n\n\n\n\n\n\n\nThere are 6 unique complex objects composed of the 3 possibly relevant features. The pairing between the {c_text3} and currenly relevant feature in the complex cue object determines what task to perform\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    3: f"\n\n\nIf {context_text}, then SHAPE matters! \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    4: f'\n\n\n    BOW = TASK 1 ( horizontal )         HOURGLASS = TASK 2 ( vertical )      HEXAGON = Task 3 ( diagonal ) \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen',
                    5: f"\n\n\nIf {context_text}, then COLOR matters! \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    6: f'\n\n\n ORANGE = TASK 1 ( horizontal )      GREEN = TASK 2 ( vertical )      PURPLE = Task 3 ( diagonal ) \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen',
                    7: f"\n\n\nIf {context_text}, then OUTLINE matters! \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen",
                    8: f'\n\n\n LONG DASH = TASK 1 ( horizontal )      SHORT DASH = TASK 2 ( vertical )      NO DASH = Task 3 ( diagonal ) \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nPress the right arrow key to move on to the next screen\nPress the left arrow key to go back a screen',
                    9: 'Please wait for the experimentor to start either the practice block or the full task'}
associated_img_dict = {0: instruct_trial, 1: instruct_trial2, 2: tmp,
                        3: shape_context_cues,
                        4: shape_context_cues,
                        5: color_context_cues,
                        6: color_context_cues,
                        7: outline_context_cues,
                        8: outline_context_cues,
                        9: []}



do_task = -1
# ---- display instruction screen(s)
do_task = disp_instruct_choice(win, instruct_dict, associated_img_dict, do_task)
win.flip()
core.wait(1)


####################################################################################
#### Set up text objects and basic shapes like fixation and feedback signs
####################################################################################
text_obj = visual.TextStim(win=win, name='text', text='', units='deg', pos=(0, 0), font=u'Times', height=1.5, color='white', wrapWidth=35)
fixation = visual.GratingStim(win=win, units=visual_unit, size=0.25, pos=[0,0], sf=0, color='black', colorSpace='rgb')
fixation_center = visual.GratingStim(win=win, units=visual_unit, size=0.05, pos=[0,0], sf=0, color=expInfo['backgroundColor'], colorSpace='hsv')
plus = visual.TextStim(win=win, name='plus', text='+', bold=True, units='deg', pos=(0, 0.25), font=u'Times', height=3.5, color='black')
minus = visual.TextStim(win=win, name='minus', text='-', bold=True, units='deg', pos=(0, 0.55), font=u'Times', height=3.5, color='black')
flash_obj = visual.Circle(win, name='flash', radius=1, units='deg', lineWidth=1, lineColor='white', fillColor=None, colorSpace='rgb', pos=(0, 0), size=1.0) #, opacity=0.1)
#probe_circ = visual.Circle(win, name='probe_circ', radius=5, edges='circ', units='deg', lineWidth=5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(0, 0), size=1.0) # long (T2)

EndofBlock_Text = visual.TextStim(win=win, text=u'End of trial block: please take a short break or press any key to continue.', font=u'Arial', units='norm', pos=(0, 0), height=0.08, ori=0, color=u'white', colorSpace='rgb', opacity=1)

# 1='top left' ,  2='top right' ,  3='bottom left' ,  4='bottom right'
quad_dict={1:visual.Circle(win, name='UL', radius=0.5, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-3.5, 3.5), size=1.0, opacity=0.75),
          2:visual.Circle(win, name='UR', radius=0.5, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(3.5, 3.5), size=1.0, opacity=0.75),
          3:visual.Circle(win, name='LL', radius=0.5, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-3.5, -3.5), size=1.0, opacity=0.75),
          4:visual.Circle(win, name='LR', radius=0.5, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(3.5, -3.5), size=1.0, opacity=0.75)}

# MRI objects
Directions = visual.TextStim(win=win, text=u'You are now about to begin the task. \n\nGet Ready \n\nPress Any Key to Continue',
                            font=u'Arial', units='norm', pos=(0, 0), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)
Wait_for_Scanner = visual.TextStim(win=win, text=u'Waiting for MRI to initiate task', units='norm', pos=(0,0), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)
Pre_Calibration = visual.TextStim(win=win, text=u'Press [C] to start calibration', units='norm', pos=(0,0), height=0.09, ori=0, color=u'white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0)
if subInfo['method']=="MRI":
    fixation = visual.GratingStim(win=win, units=visual_unit, size=0.15, pos=[0,0], sf=0, color='black', colorSpace='rgb') # modify fixation size for MRI
    fixation_center = visual.GratingStim(win=win, units=visual_unit, size=0.05, pos=[0,0], sf=0, color=expInfo['backgroundColor'], colorSpace='hsv')
    quad_dict={1:visual.Circle(win, name='UL', radius=0.25, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-1.5, 1.5), size=1.0, opacity=0.75),
                2:visual.Circle(win, name='UR', radius=0.25, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(1.5, 1.5), size=1.0, opacity=0.75),
                3:visual.Circle(win, name='LL', radius=0.25, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(-1.5, -1.5), size=1.0, opacity=0.75),
                4:visual.Circle(win, name='LR', radius=0.25, units='deg', lineWidth=2.5, lineColor=None, fillColor='black', colorSpace='rgb', pos=(1.5, -1.5), size=1.0, opacity=0.75)}
###########################################################################################################################
###################################################### Run Task #########################################################

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# - - - - - - - - - - - - -     Run Task Code     - - - - - - - - - - - - #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
if subInfo['method']=="MRI":
    resp_keys['start_scan'] = '1' # add key for subject to press when they are ready to start the scan
    startlist = list(resp_keys.values())
    accepted_keys = resp_keys_list + laptop_resp_keys_list + ['escape']
else:
    accepted_keys = resp_keys_list + ['escape']

if 'EEG' in subInfo['method']:
    port.open() # TTL Pulse trigger
    core.wait(1)
    win.flip()
    port.write(trigDict['startSaveflag']) # TTL Pulse trigger
    # Add 15 second cushion to the start of the file (for filter purposes)
    if 'pupil' not in subInfo['method']:
        for countdown in np.asarray([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]):
            text_obj.text = str(countdown) # drawing countdown for participant so they don't get antsy
            text_obj.draw()
            fixation.draw()
            win.flip()
            core.wait(1)

if 'pupil' in subInfo['method']:
    set_up_calibration()

# - - - - - Task Prep Code
prac_count = 0
#do_task=0
# - - - - - Block Code
while do_task < 2:
    
    if do_task == -1:
        # ---- display instruction screen(s) again if requested
        do_task = disp_instruct_choice(win, instruct_dict, associated_img_dict, do_task)
        win.flip()
        core.wait(1)
        
    if do_task == 1:
        n_blocks = Task_Parameters['n_blocks']
        n_trials = Task_Parameters['n_trials']
    else:
        n_blocks = 1 # only 1 practice block
        n_trials = Task_Parameters['n_prac_trials']
    
    abort_task = False # initialize here so we can undo exit if needed
    # --------------------------------------------------------------------------------------
    # ------------------------------------  BLOCK loop  ------------------------------------
    # --------------------------------------------------------------------------------------
    for i_block in range(start_blk, n_blocks):
        # ---- Prep block trials
        cur_datetime = datetime.today().strftime('%Y%m%d_%I%M')
        if do_task == 0:
            # PRACTICE BLOCK
            extra_time = 2 # lengthen the cue duration and stim duration and probe duration a little bit to make it easier
            prac_count += 1 # in case we have to do practice more than once
            Block_Output = prepare_prac_trials(prac_count, Task_Parameters, resp_keys)
            blk_filename = os.path.join(output_dir, ("Data_"+subInfo['method'].split("+")[0]), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-HCC_prac-00" + str(int(prac_count)) + "_date-" + cur_datetime + ".csv"))
            local_blk_filename = os.path.join("C:","PsychoPyData","HCC_LC", ("Data_"+subInfo['method'].split("+")[0]), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-HCC_prac-00" + str(int(prac_count)) + "_date-" + cur_datetime + ".csv"))
            edf_file = str(int(subInfo['Subject_ID'])) + '_p' + str(int(prac_count)) + ".EDF" # os.path.join(output_dir, ("Data_"+subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-HCC_prac-00" + str(int(i_block)) + "_date-" + cur_datetime + ".EDF"))
        else:
            # ACTUAL TASK
            extra_time = 1 # no extra time
            Block_Output = prepare_block_trials(int(i_block+1), trial_order_dict, trial_type_order_dict, Task_Parameters, cue_list, task_list, resp_keys)
            blk_filename = os.path.join(output_dir, ("Data_"+subInfo['method'].split("+")[0]), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-HCC_block-00" + str(int(i_block)) + "_date-" + cur_datetime + ".csv"))
            local_blk_filename = os.path.join("C:","PsychoPyData","HCC_LC", ("Data_"+subInfo['method'].split("+")[0]), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-HCC_block-00" + str(int(i_block)) + "_date-" + cur_datetime + ".csv"))
            edf_file = str(int(subInfo['Subject_ID'])) + '_' + str(int(i_block)) + ".EDF" #os.path.join(output_dir, ("Data_"+subInfo['method']), ("sub-" + str(int(subInfo['Subject_ID'])) + "_task-HCC_block-00" + str(int(i_block)) + "_date-" + cur_datetime + ".EDF"))
            
        # ---- Start Block
        if 'pupil' in subInfo['method']:
            Pre_Calibration.draw() # have them press when they're ready
            win.flip()
            #event.waitKeys() 
            # -- 
            if dummy_mode:
                print('ERROR: This task requires real-time gaze data.\n' +
                    'It cannot run in Dummy mode (with no tracker connection).')
                Close()
            else:  # calibrate the tracker
                try:
                    el_tracker.doTrackerSetup()
                except RuntimeError as err:
                    print('ERROR:', err)
                    el_tracker.exitCalibration()
            # -- open edf file for block
            try:
                el_tracker.openDataFile(edf_file)
            except RuntimeError as err:
                print('ERROR:', err)
                # close the link if we have one open
                if el_tracker.isConnected():
                    el_tracker.close()
                core.quit()
                sys.exit()
            
        if 'EEG' in subInfo['method']:
            # do a 4 second initial fixation before starting the first trial (lets EEG settle down a bit)
            for frame in range(int(4*refresh_rate)):
                fixation.draw()
                win.flip()
        elif subInfo['method']=='MRI':
            # add time since run columns for tracking onsets
            if do_task == 1:
                Block_Output['Time_Since_Run_Cue_Prez'] = np.zeros(Task_Parameters['n_trials'])
                Block_Output['Time_Since_Run_Probe_Prez'] = np.zeros(Task_Parameters['n_trials'])
                Block_Output['Time_Since_Run_subRespo'] = np.zeros(Task_Parameters['n_trials'])
                #Block_Output['Time_Since_Run_Feedback_Prez'] = np.zeros(Task_Parameters['n_trials'])
            else:
                Block_Output['Time_Since_Run_Cue_Prez'] = np.zeros(Task_Parameters['n_prac_trials'])
                Block_Output['Time_Since_Run_Probe_Prez'] = np.zeros(Task_Parameters['n_prac_trials'])
                Block_Output['Time_Since_Run_subRespo'] = np.zeros(Task_Parameters['n_prac_trials'])
                #Block_Output['Time_Since_Run_Feedback_Prez'] = np.zeros(Task_Parameters['n_prac_trials'])
            
            # grab out current block iti and delay list for jittering
            if do_task == 1:
                jitter_delay1 = iti_dict['delay1'][i_block]
                jitter_ITI = iti_dict['ITI'][i_block]
            else:
                jitter_delay1 = np.random.uniform(Trl_Durs['delay'][0], Trl_Durs['delay'][1], Task_Parameters['n_prac_trials'])
                jitter_ITI = np.random.uniform(Trl_Durs['delay'][0], Trl_Durs['delay'][1], Task_Parameters['n_prac_trials'])
                
            # draw directions text to have them press before starting block
            Directions.draw() # have them press when they're ready
            win.flip()
            event.waitKeys(keyList=startlist) # only use subj resp keys so scanner can prep in background
            
            ##### TTL Pulse trigger
            Wait_for_Scanner.draw()
            win.flip()
            key = event.waitKeys(keyList=['lshift','z','equal','5', 'escape','q']) # all key options for 3T or 7T at Uiowa
            if ((key=='q')|(key=='escape')):
                print("experimentor requested to exit the script")
                
            #### Setting up a global clock to track initiation of experiment to end
            Time_Since_Run = core.MonotonicClock()  # to track the time since experiment started, this way it is very flexible compared to psychopy.clock
            ##### 6 seconds Intial fixation
            fixation.draw()
            win.flip()
            core.wait(pre_first_trial_delay) # set to ~1 instead of 6 seconds b/c first trial will be tossed (aka it is okay for first 3 TRs to overlap with this trial)
        else:
            fixation.draw()
            #flash_obj.draw()
            win.flip()
            core.wait(1)
            #n_trials = 10 # DELETE BEFORE ACTUALLY RUNNING!!!!!!!
        
        # --------------------------------------------------------------------------------------
        # ------------------------------------  TRIAL loop  ------------------------------------
        # --------------------------------------------------------------------------------------
        win.mouseVisible = False # remove mouse from screen
        for i_trial in range(0, n_trials):
            
            ### - - - -  (0) Pre-trial Fixation  - - - - - - - - - - -
            if 'pupil' in subInfo['method']:
                # put the tracker in the offline mode first
                el_tracker = pylink.getEYELINK()
                el_tracker.setOfflineMode()    
                el_tracker.sendCommand('clear_screen 0')  # clear the host Display
                el_tracker.sendMessage('TRIALID %d' % i_trial)
                status_msg = 'TRIAL number %d, block %d' % (i_trial, i_block)
                el_tracker.sendCommand("record_status_message '%s'" % status_msg)
                # only do drift check if just collecting pupil and not EEG or MEG
                if subInfo['method'].split("+")[0]=='Behavioral':
                    while not dummy_mode:
                        # terminate the task if no longer connected to the tracker or
                        # user pressed Ctrl-C to terminate the task
                        if (not el_tracker.isConnected()) or el_tracker.breakPressed():
                            Close()
                        # drift-check and re-do camera setup if ESCAPE is pressed
                        try:
                            error = el_tracker.doDriftCorrect(int(scn_width/2.0),
                                                            int(scn_height/2.0), 1, 1)
                            # break following a success drift-check
                            if error is not pylink.ESC_KEY:
                                break
                        except:
                            pass
                el_tracker.setOfflineMode()
                try:
                    el_tracker.startRecording(1, 1, 1, 1)
                except RuntimeError as error:
                    print("ERROR:", error)
                    abort_trial()
                    #return pylink.TRIAL_ERROR
                pylink.pumpDelay(100)
                eye_used = el_tracker.eyeAvailable()
                if eye_used == 1:
                    el_tracker.sendMessage("EYE_USED 1 RIGHT")
                elif eye_used == 0 or eye_used == 2:
                    el_tracker.sendMessage("EYE_USED 0 LEFT")
                    eye_used = 0
                else:
                    print("Error in getting the eye information!")
                    #return pylink.TRIAL_ERROR    
                # send flag for fixation setup
                el_tracker.sendMessage('fix_onset')
                fixation.draw()
                fixation_center.draw()
                win.flip()
                core.wait(0.995)
                cx, cy = scn_width/2, scn_height/2
                TOL_PX = 50
                sp = None
                fix_ok = 1
                while sp is None:
                    sp = el_tracker.getNewestSample()
                    if sp and sp.getEye() != eye_used:
                        sp = None
                fx,fy = (sp.getLeftEye().getGaze() if eye_used == 0
                    else sp.getRightEye().getGaze())
                if (fx-cx)**2 + (fy-cy)**2 <= TOL_PX**2:
                    el_tracker.sendMessage('fix_ok')    # accept trial
                else:
                    el_tracker.sendMessage('fix_fail')  # log a miss
                    fix_ok = 0
            
            ### - - - -  (1) Cue Screen  - - - - - - - - - - -
            # Will present a flash before or after cue presentation
            if 'pupil' in subInfo['method']:
                el_tracker.sendMessage('cue_onset')
            if 'EEG' in subInfo['method']:
                win.callOnFlip(port.write,bytes( [ trigDict['cue'][Block_Output['context'][i_trial]][Block_Output['cue'][i_trial]] ] ))
            if 'MEG' in subInfo['method']:
                win.callOnFlip(port.setData, int(trigDict['cue']))
            if subInfo['method']=='MRI':
                Block_Output['Time_Since_Run_Cue_Prez'][i_trial]=Time_Since_Run.getTime()
            # ---- draw specific cue object
            if context_modality == "visual":
                # ---- context can be visual or auditory
                for frame in range(int(Trl_Durs['cue']*refresh_rate*extra_time)):
                    context_dict['visual'][Block_Output['context'][i_trial]].draw()
                    fixation.draw()
                    fixation_center.draw()
                    for cue_line in cue_dict[Block_Output['cue'][i_trial]]:
                        cue_line.draw() # grab correct cue
                    win.flip()
                    # -- quit check if request to quit
                    keys = kb.getKeys(keyList=accepted_keys, waitRelease=False, clear=True) #keyList=resp_keys
                    if (keys!=None) and (keys!=[]):
                        if ((keys[0].name == 'q')|(keys[0].name == 'escape')):
                            if 'pupil' in subInfo['method']:
                                Close()
                            abort_task = True # will close the task
                            break # break cue for loop
            else:
                fixation.draw()
                fixation_center.draw()
                for cue_line in cue_dict[Block_Output['cue'][i_trial]]:
                    cue_line.draw() # grab correct cue
                context_dict['auditory'][Block_Output['context'][i_trial]].play(when=win.getFutureFlipTime(clock='ptb')) # only "play" tone on first iteration
                win.flip()
                core.wait(int(Trl_Durs['cue']*extra_time))
                context_dict['auditory'][Block_Output['context'][i_trial]].stop()
            if abort_task == True:
                break # break trial loop after cue for loop
            # - - - - - - - - - - - - - - - - - - - - - - - - -
            

            ### - - - -  (2) Delay Screen  - - - - - - - - - -
            # HSS Task has delays... perception tasks do not
            if 'EEG' in subInfo['method']:
                win.callOnFlip(port.write,bytes([ trigDict['delay'] ]))
            if 'MEG' in subInfo['method']:
                win.callOnFlip(port.setData, int(0))
            if subInfo['method'] == 'MRI':
                cur_delay_time = jitter_delay1[i_trial]
            else:
                cur_delay_time = np.random.uniform(Trl_Durs['delay'][0], Trl_Durs['delay'][1])
            Block_Output['delay'][i_trial] = cur_delay_time
            for frame in range(int(cur_delay_time*refresh_rate)):
                fixation.draw()
                fixation_center.draw()
                win.flip()
                # -- quit check if request to quit
                keys = kb.getKeys(keyList=accepted_keys, waitRelease=False, clear=True) #keyList=resp_keys
                if (keys!=None) and (keys!=[]):
                    if ((keys[0].name == 'q')|(keys[0].name == 'escape')):
                        if 'pupil' in subInfo['method']:
                            Close()
                        abort_task = True # will close the task
                        break # break delay for loop
            if abort_task == True:
                break # break trial loop after delay for loop
            # - - - - - - - - - - - - - - - - - - - - - - - - -
            
            
            # ### - - - -  (3) Probe Screen  - - - - - - - - - 
            if 'pupil' in subInfo['method']:
                el_tracker.sendMessage('probe_onset')
            if 'EEG' in subInfo['method']:
                win.callOnFlip(port.write,bytes( [ trigDict['probe'][Block_Output['probe_loc'][i_trial]] ] )) # CHANGE TO ACTUAL FLAG LATER
            if 'MEG' in subInfo['method']:
                win.callOnFlip(port.setData, int(trigDict['probe']))
            if subInfo['method']=='MRI':
                Block_Output['Time_Since_Run_Probe_Prez'][i_trial]=Time_Since_Run.getTime()
            kb.clearEvents() # restart on each trial 
            kb.clock.reset() # restart on each trial to get probe onset time + button press time
            already_responded=False
            probe_obj = quad_dict[Block_Output['probe_loc'][i_trial]]
            # present probe until resposne
            for frame in range(int(Trl_Durs['probe']*refresh_rate*extra_time)):
                probe_obj.draw()
                fixation.draw()
                fixation_center.draw()
                if not(already_responded):
                    keys = kb.getKeys(keyList=accepted_keys, waitRelease=False, clear=True) #keyList=resp_keys
                if (keys!=None) and (keys!=[]):
                    # first make sure they aren't asking to quit
                    if ((keys[0].name == 'q')|(keys[0].name == 'escape')):
                        if 'pupil' in subInfo['method']:
                            Close()
                        abort_task = True # will close the task
                        break # break probe for loop
                    already_responded=True
                    if subInfo['method']!='MRI':
                        break # exit for loop that is displaying stimuli
                    else:
                        Block_Output['Time_Since_Run_subRespo'][i_trial] = Time_Since_Run.getTime()
                win.flip()
            if (keys!=None) and (keys!=[]):
                for key in keys:
                    #print(key.name)
                    if key.name in resp_keys_list:
                        Block_Output['subject_response'][i_trial] = int(key.name[-1]) # key.name=[num_2] will grab 2
                        # send trigger code
                        if 'EEG' in subInfo['method']:
                            port.write(bytes([ trigDict['resp'][Block_Output['subject_response'][i_trial]] ]))
                            core.wait(0.100) # draw probe object for another 100 ms after response
                    elif key.name in laptop_resp_keys_list:
                        mod_key_name = laptop_key_press(key.name)
                        Block_Output['subject_response'][i_trial] = mod_key_name
                    Block_Output['response_time'][i_trial] = key.rt # DOUBLE CHECK THIS
                # save if response was correct
                if Block_Output['subject_response'][i_trial] == Block_Output['correct_response'][i_trial]:
                    Block_Output['subject_correct'][i_trial] = 1
            # -- check if we need to break trial loop
            if abort_task == True:
                break # break trial loop after probe for loop
            # # - - - - - - - - - - - - - - - - - - - - - - - - -
            
            if do_task == 0:
                # ### - - - -  (4) Delay Screen  - - - - - - - - - -
                if 'EEG' in subInfo['method']:
                    win.callOnFlip(port.write,bytes([ trigDict['delay'] ]))
                if 'MEG' in subInfo['method']:
                    win.callOnFlip(port.setData, int(0))
                cur_delay_time = np.random.uniform(Trl_Durs['delay'][0], Trl_Durs['delay'][1])
                for frame in range(int(cur_delay_time*refresh_rate)):
                    fixation.draw()
                    fixation_center.draw()
                    win.flip()
                    # -- quit check if request to quit
                    keys = kb.getKeys(keyList=accepted_keys, waitRelease=False, clear=True) #keyList=resp_keys
                    if (keys!=None) and (keys!=[]):
                        if ((keys[0].name == 'q')|(keys[0].name == 'escape')):
                            if 'pupil' in subInfo['method']:
                                Close()
                            abort_task = True # will close the task
                            break # break delay for loop
                if abort_task == True:
                    break # break trial loop after delay for loop
                # # - - - - - - - - - - - - - - - - - - - - - - - - -
                
                
                ### - - - -  (5) Feedback Screen  - - - - - - - - - -
                # ONLY practice has feedback
                if 'EEG' in subInfo['method']:
                    win.callOnFlip(port.write,bytes([ trigDict['feedback'][Block_Output['subject_correct'][i_trial]] ])) # ADD ACTUAL FLAG LATER
                if 'MEG' in subInfo['method']:
                    win.callOnFlip(port.setData, int(trigDict['feedback']))
                # -- display feedback
                for frame in range(int(Trl_Durs['feedback']*refresh_rate*extra_time)):
                    fixation.draw()
                    fixation_center.draw()
                    if Block_Output['subject_response'][i_trial] == Block_Output['correct_response'][i_trial]:
                        # present plus sign
                        plus.draw()
                    else:
                        # if response was not correct, present minus sign
                        minus.draw()
                    win.flip() # display stimuli
                    # -- quit check if request to quit
                    keys = kb.getKeys(keyList=accepted_keys, waitRelease=False, clear=True) #keyList=resp_keys
                    if (keys!=None) and (keys!=[]):
                        if ((keys[0].name == 'q')|(keys[0].name == 'escape')):
                            if 'pupil' in subInfo['method']:
                                Close()
                            abort_task = True # will close the task
                            break # break feedback for loop
                if abort_task == True:
                    break # break trial loop after feedback for loop
                # - - - - - - - - - - - - - - - - - - - - - - - - -
            
            
            ### - - - -  (6) ITI Screen  - - - - - - - - - - -
            if 'EEG' in subInfo['method']:
                win.callOnFlip(port.write,bytes([ trigDict['ITI'] ]))
            if 'MEG' in subInfo['method']:
                win.callOnFlip(port.setData, int(0))
            iti_timer = core.Clock()
            fixation.draw()
            fixation_center.draw()
            win.flip()
            if subInfo['method'] == 'MRI':
                rand_iti_selection = jitter_ITI[i_trial]
            else:
                rand_iti_selection = np.random.uniform(Trl_Durs['ITI'][0], Trl_Durs['ITI'][1])
            Block_Output['ITI'][i_trial] = rand_iti_selection
            # save out data from last block
            Block_Output_df = pd.DataFrame(Block_Output)
            # WE HAVE BEEN HAVING CONNECTION ISSUES... need try/except
            try:
                Block_Output_df.to_csv(blk_filename, index=False)
            except:
                if i_trial == n_trials:
                    print("connection issues... trying again since it is the last trial of the block")
                    try:
                        Block_Output_df.to_csv(blk_filename, index=False)
                    except:
                        print("failed again... saving locally so we do not lose any data")
                        if subInfo['method'] != 'MRI':
                            Block_Output_df.to_csv(local_blk_filename, index=False)
                else:
                    print("Connection issues... will skip for now and try again on next trial")
                    
            if 'pupil' in subInfo['method']:
                el_tracker.sendMessage('ITI_screen')
                # send a message to clear the Data Viewer screen as well
                el_tracker.sendMessage('!V CLEAR 128 128 128')
                el_tracker.stopRecording()
                el_tracker.sendMessage('!V TRIAL_VAR Hierarchical_Switch_Type %s' % Block_Output['hierarhical_switch_type'][i_trial])
                el_tracker.sendMessage('!V TRIAL_VAR Context %s' % Block_Output['context'][i_trial])
                el_tracker.sendMessage('!V TRIAL_VAR Task %s' % Block_Output['task'][i_trial])
                el_tracker.sendMessage('!V TRIAL_VAR Probe_Location %d' % Block_Output['probe_loc'][i_trial])
                try:
                    el_tracker.sendMessage('!V TRIAL_VAR Response %d' % int(Block_Output['subject_response'][i_trial]))
                except:
                    el_tracker.sendMessage('!V TRIAL_VAR Response %d' % 0) # tracker doesn't like -1, so change to 0
                el_tracker.sendMessage('!V TRIAL_VAR Accuracy %d' % Block_Output['subject_correct'][i_trial])
                try:
                    el_tracker.sendMessage('!V TRIAL_VAR RT %d' % int(Block_Output['response_time'][i_trial]*1000))
                except:
                    el_tracker.sendMessage('!V TRIAL_VAR RT %d' % 0) # tracker doesn't like -1, so change to 0
                el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
                rand_iti_selection = rand_iti_selection - 0.995 # remove inital fixation for next trial
            while iti_timer.getTime() < rand_iti_selection:
                fixation.draw()
                fixation_center.draw()
                win.flip()
                # -- quit check if request to quit
                keys = kb.getKeys(keyList=accepted_keys, waitRelease=False, clear=True) #keyList=resp_keys
                if (keys!=None) and (keys!=[]):
                    if ((keys[0].name == 'q')|(keys[0].name == 'escape')):
                        if 'pupil' in subInfo['method']:
                            Close()
                        abort_task = True # will close the task
                        break # break ITI for loop
            if abort_task == True:
                break # break trial loop after ITI for loop
            # - - - - - - - - - - - - - - - - - - - - - - - - -
        
        
        # end of trial loop
        if 'pupil' in subInfo['method']:
            # -- close edf file after each block WITHOUT closing window
            el_tracker = pylink.getEYELINK()
            if el_tracker.isConnected():
                # Terminate the current trial first if the task terminated prematurely
                error = el_tracker.isRecording()
                if error == pylink.TRIAL_OK:
                    abort_trial()
                
                # Put tracker in Offline mode
                el_tracker.setOfflineMode()
                
                # Clear the Host PC screen and wait for 500 ms
                el_tracker.sendCommand('clear_screen 0')
                pylink.msecDelay(500)
                
                # Close the edf data file on the Host
                el_tracker.closeDataFile()
                
                # Download the EDF data file from the Host PC to a local data folder
                # parameters: source_file_on_the_host, destination_file_on_local_drive
                local_edf = edf_file
                try:
                    el_tracker.receiveDataFile(edf_file, local_edf)
                except RuntimeError as error:
                    print('ERROR:', error)
            # -- now move into proper sub-directory
            pattern = edf_file
            source_dir = _thisDir # f"C:\\Biosemi\\Exp_Data\\Data_{subInfo['method']}"
            folder=subInfo['method'].split("+")[0]
            destination_dir = os.path.join(_thisDir, f"Data_{folder}", "EDFs")
            # -- copy actual task blocks
            try:
                for file_path in glob.glob(os.path.join(source_dir, pattern)):
                    shutil.copy2(file_path, destination_dir) # using copy2 to preserve time stamps
            except:
                print("Error when attempting to copy EDF file\nError\nError\nError when attempting to copy EDF file\nError\nError\nError when attempting to copy EDF file")
        ### - - - -  Block Screen  - - - - - - - - - - - - - - -
        if abort_task == True:
            win.flip() # clear screen
            kb.clearEvents()
            core.wait(1)
            # check if we need to exit the code entirely
            text_obj.text = f"Press [escape] or [q] again to fully quit the task\nPress any other key to continue with the next block"
            text_obj.draw()
            win.flip()
            # -- quit check if request to quit
            key = event.waitKeys(keyList=accepted_keys)
            print(key[0])
            if ((key[0] == 'q')|(key[0] == 'escape')):  
                do_task = 99
                break    
        else:
            if 'EEG' in subInfo['method']:
                win.callOnFlip(port.write,bytes([ trigDict['blockEnd'] ]))
                fixation.draw()
                fixation_center.draw()
                win.flip()
                core.wait(2) # pause 2 seconds so EEG doesn't go crazy right after last block trial
                if 'pupil' in subInfo['method']:
                    for countdown in np.asarray([15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]):
                        text_obj.text = f"please sit still\n{countdown}\nfeel free to close your eyes" # str(countdown) # drawing countdown for participant so they don't get antsy
                        text_obj.draw()
                        fixation.draw()
                        win.flip()
                        core.wait(1)
                    port.write(trigDict['stopSaveflag']) # I only want to pause inbetween blocks if also collecting pupil (since end and start can have lots of extra time)
                cur_block_acc = round(100*(Block_Output['subject_correct'].sum()/n_trials), 2)
                disp_block_text(EndofBlock_Text, cur_block_acc, ((i_block+1)==Task_Parameters['n_blocks']), max_time)
                if 'pupil' in subInfo['method']:
                    port.write(trigDict['startSaveflag']) # I only want to pause inbetween blocks if also collecting pupil (since end and start can have lots of extra time)
            elif subInfo['method']=='MRI':
                ### block screens
                core.wait(9) # add another blank screen for 9 seconds before ending the block
                blk_end_time = Time_Since_Run.getTime()
                cur_block_acc = round(100*(Block_Output['subject_correct'].sum()/n_trials), 2)
                print("Block ", str(int(i_block+1)), " duration: ", int(blk_end_time/60), " minutes ", int(blk_end_time%60), " seconds\tAccuracy was ", str(cur_block_acc))
                EndofBlock_Text.text = "Block " + str(int(i_block+1)) + " finished.\n\nAccuracy was " + str(cur_block_acc) + " percent\n\nPrepping next task block now...Please stay still"
                EndofBlock_Text.draw()
                win.flip()
                core.wait(5) # give myself enough time to turn off the triggers
                win.winHandle.minimize() # temporarily minimize the window
                win.winHandle.set_fullscreen(False)
                win.flip()
                dlg_cont = gui.DlgFromDict(dictionary={'Click OK when ready to move on':''}, title="Click okay when ready to move on", sortKeys=False)
                win.winHandle.maximize()
                win.winHandle.set_fullscreen(True) 
                win.winHandle.activate()
                win.flip()
                core.wait(0.5)
            else:
                cur_block_acc = round(100*(Block_Output['subject_correct'].sum()/n_trials), 2)
                disp_block_text(EndofBlock_Text, cur_block_acc, ((i_block+1)==Task_Parameters['n_blocks']), max_time)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        
    
    # end of block loop ... check if we need to move from practice to task
    if do_task == 0:
        end_of_blk_txt = 'Accuracy was '+str(cur_block_acc)+'\nPlease wait for the experimentor.'
        screen_dict = {0:end_of_blk_txt}
        img_dict = {0:[]}
        do_task = disp_instruct_choice(win, screen_dict, img_dict, do_task)
        if 'EEG' in subInfo['method']:
            if do_task==1:
                win.callOnFlip(port.write,bytes([ trigDict['pracEnd'] ]))
    else:
        do_task = 2 # end the loop if the subject was doing the full task


# -- add some extra time if EEG for filter and then stop save
if 'EEG' in subInfo['method']:
    text_obj.text = "ALL DONE!"
    text_obj.draw()
    win.flip()
    core.wait(1)

    text_obj.text = "STAY STILL!"
    text_obj.draw()
    win.flip()
    core.wait(1)

    for countdown in range(15):
        text_obj.text = str(int(15-countdown))
        text_obj.draw()
        win.flip()
        core.wait(1) # pause 2 seconds so EEG doesn't go crazy right after last block trial
        
    port.write(trigDict['stopSaveflag']) # TTL Pulse trigger
    core.wait(1) # just make sure it has time to send flag and stop save before we close the port
    port.close() # okay now close port

if 'pupil' in subInfo['method']:
    Close()
  
# -- Close window
win.close() # close visual window
