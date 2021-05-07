from ctypes import *
import os.path
from os import path
import numpy as np
import sys
import pandas as pd
from .frame import Frame, Skeleton
from .bone_structure import Bone_ID, Data_Format

Flag_Receiving = False
Flag_Comm_Started = False
is_finished = False

def bvhFrameDataReceived(a,b,c,data):
    global Flag_Receiving
    global Flag_Comm_Started
    Flag_Receiving = True
    Flag_Comm_Started = True
    print(sizeof(c[0]))
    #for field in c[0]._fields_:
    #    print(field[0], getattr(c[0], field[0]))
    frame_idx = getattr(c[0],c[0]._fields_[7][0])
    create_bones(frame_idx, data)
    #print('Y:{:.4}, X:{:.4}, Z:{:.4}'.format(data[3],data[4],data[5]))
    return a

def verify_connection():
    global Flag_Receiving
    global Flag_Comm_Started
    global is_finished
    if(Flag_Comm_Started):
        if(Flag_Receiving):
            Flag_Receiving = False
        else:
            if(not is_finished):
                df['Frame_Idx'] = df['Frame_Idx'].astype(int)
                df=df.drop([0], axis=0)
                #df=df.drop(df.columns[[0]], axis='columns')
                df.to_csv('platformData.csv',index=0)
                print('Comm finished')
                is_finished = True
    return is_finished

def create_bones(frame_idx, data):
    bones = np.zeros((7,6))
    for i in range(7):
        bones[i] = np.array(data[(i*6)+0:(i*6)+6])

    for idx, i in enumerate(skel.ids):
        skel.set_bone(idx,bones[Bone_ID.get_index(i)])

    cur_frame = Frame(frame_idx, skel)
    append_frame(cur_frame)

def init_load(seconds, mode_in, bvh_format):
    global skel
    global df
    global mode
    global data_format
    global Flag_Receiving
    global Flag_Comm_Started

    Flag_Receiving = False
    Flag_Comm_Started = False

    '''
    Modes to create Skeleton:
    mode = 0 Hips + Left Leg
    mode = 1 Hips + Right Leg
    mode = 2 Hips + Both Legs
    '''
    mode = mode_in
    skel = Skeleton(mode)
    data_format = bvh_format
    if(mode == 0):
        df = pd.DataFrame(columns=[ 'Frame_Idx',
                                    'Hips-T_{}'.format(str(data_format)[-3]), 'Hips-T_{}'.format(str(data_format)[-2]), 'Hips-T_{}'.format(str(data_format)[-1]),
                                    'Hips-R_{}'.format(str(data_format)[-3]), 'Hips-R_{}'.format(str(data_format)[-2]), 'Hips-R_{}'.format(str(data_format)[-1]),
                                    'LUL-T_{}'.format(str(data_format)[-3]), 'LUL-T_{}'.format(str(data_format)[-2]), 'LUL-T_{}'.format(str(data_format)[-1]),
                                    'LUL-R_{}'.format(str(data_format)[-3]), 'LUL-R_{}'.format(str(data_format)[-2]), 'LUL-R_{}'.format(str(data_format)[-1]),
                                    'LK-T_{}'.format(str(data_format)[-3]), 'LK-T_{}'.format(str(data_format)[-2]), 'LK-T_{}'.format(str(data_format)[-1]),
                                    'LK-R_{}'.format(str(data_format)[-3]), 'LK-R_{}'.format(str(data_format)[-2]), 'LK-R_{}'.format(str(data_format)[-1]),
                                    'LF-T_{}'.format(str(data_format)[-3]), 'LF-T_{}'.format(str(data_format)[-2]), 'LF-T_{}'.format(str(data_format)[-1]),
                                    'LF-R_{}'.format(str(data_format)[-3]), 'LF-R_{}'.format(str(data_format)[-2]), 'LF-R_{}'.format(str(data_format)[-1])])
    if(mode == 1):
        df = pd.DataFrame(columns=[ 'Frame_Idx',
                                    'Hips-T_{}'.format(str(data_format)[-3]), 'Hips-T_{}'.format(str(data_format)[-2]), 'Hips-T_{}'.format(str(data_format)[-1]),
                                    'Hips-R_{}'.format(str(data_format)[-3]), 'Hips-R_{}'.format(str(data_format)[-2]), 'Hips-R_{}'.format(str(data_format)[-1]),
                                    'RUL-T_{}'.format(str(data_format)[-3]), 'RUL-T_{}'.format(str(data_format)[-2]), 'RUL-T_{}'.format(str(data_format)[-1]),
                                    'RUL-R_{}'.format(str(data_format)[-3]), 'RUL-R_{}'.format(str(data_format)[-2]), 'RUL-R_{}'.format(str(data_format)[-1]),
                                    'RK-T_{}'.format(str(data_format)[-3]), 'RK-T_{}'.format(str(data_format)[-2]), 'RK-T_{}'.format(str(data_format)[-1]),
                                    'RK-R_{}'.format(str(data_format)[-3]), 'RK-R_{}'.format(str(data_format)[-2]), 'RK-R_{}'.format(str(data_format)[-1]),
                                    'RF-T_{}'.format(str(data_format)[-3]), 'RF-T_{}'.format(str(data_format)[-2]), 'RF-T_{}'.format(str(data_format)[-1]),
                                    'RF-R_{}'.format(str(data_format)[-3]), 'RF-R_{}'.format(str(data_format)[-2]), 'RF-R_{}'.format(str(data_format)[-1])])
    if(mode == 2):
        df = pd.DataFrame(columns=[ 'Frame_Idx',
                                    'Hips-T_{}'.format(str(data_format)[-3]), 'Hips-T_{}'.format(str(data_format)[-2]), 'Hips-T_{}'.format(str(data_format)[-1]),
                                    'Hips-R_{}'.format(str(data_format)[-3]), 'Hips-R_{}'.format(str(data_format)[-2]), 'Hips-R_{}'.format(str(data_format)[-1]),
                                    'RUL-T_{}'.format(str(data_format)[-3]), 'RUL-T_{}'.format(str(data_format)[-2]), 'RUL-T_{}'.format(str(data_format)[-1]),
                                    'RUL-R_{}'.format(str(data_format)[-3]), 'RUL-R_{}'.format(str(data_format)[-2]), 'RUL-R_{}'.format(str(data_format)[-1]),
                                    'RK-T_{}'.format(str(data_format)[-3]), 'RK-T_{}'.format(str(data_format)[-2]), 'RK-T_{}'.format(str(data_format)[-1]),
                                    'RK-R_{}'.format(str(data_format)[-3]), 'RK-R_{}'.format(str(data_format)[-2]), 'RK-R_{}'.format(str(data_format)[-1]),
                                    'RF-T_{}'.format(str(data_format)[-3]), 'RF-T_{}'.format(str(data_format)[-2]), 'RF-T_{}'.format(str(data_format)[-1]),
                                    'RF-R_{}'.format(str(data_format)[-3]), 'RF-R_{}'.format(str(data_format)[-2]), 'RF-R_{}'.format(str(data_format)[-1]),
                                    'LUL-T_{}'.format(str(data_format)[-3]), 'LUL-T_{}'.format(str(data_format)[-2]), 'LUL-T_{}'.format(str(data_format)[-1]),
                                    'LUL-R_{}'.format(str(data_format)[-3]), 'LUL-R_{}'.format(str(data_format)[-2]), 'LUL-R_{}'.format(str(data_format)[-1]),
                                    'LK-T_{}'.format(str(data_format)[-3]), 'LK-T_{}'.format(str(data_format)[-2]), 'LK-T_{}'.format(str(data_format)[-1]),
                                    'LK-R_{}'.format(str(data_format)[-3]), 'LK-R_{}'.format(str(data_format)[-2]), 'LK-R_{}'.format(str(data_format)[-1]),
                                    'LF-T_{}'.format(str(data_format)[-3]), 'LF-T_{}'.format(str(data_format)[-2]), 'LF-T_{}'.format(str(data_format)[-1]),
                                    'LF-R_{}'.format(str(data_format)[-3]), 'LF-R_{}'.format(str(data_format)[-2]), 'LF-R_{}'.format(str(data_format)[-1])])

def swap_format(bone_array):
    if(data_format==Data_Format.XZY):
        for i in range(1,df.shape[1],3):
            bone_array[i+1], bone_array[i+2] = bone_array[i+2], bone_array[i+1]
    if(data_format==Data_Format.XZY):
        for i in range(1,df.shape[1],3):
            bone_array[i+1], bone_array[i+2] = bone_array[i+2], bone_array[i+1]
    if(data_format==Data_Format.XZY):
        for i in range(1,df.shape[1],3):
            bone_array[i+1], bone_array[i+2] = bone_array[i+2], bone_array[i+1]
    print(bone_array)
    return bone_array


def append_frame(frame):
    sk = frame.get_skeleton()
    tmp_bone = []
    for idx, _ in enumerate(sk.ids):
        #print('idx: {0} values:{1} type:{2}'.format(idx,sk.get_bone(idx),type(sk.get_bone(idx))))
        tmp_bone.append(sk.get_bone(idx))
    tmp_bone = np.insert(tmp_bone,0, frame.get_id())
    #tmp_bone = swap_format(tmp_bone)
    tmp_bone = np.asarray(tmp_bone)

    df.loc[len(df)] = tmp_bone
    



"""

        df = pd.DataFrame(columns=[ 'Frame_Idx',
                                    'Hips-T_{}'.format(str(data_format)[-3]), 'Hips-T_{}'.format(str(data_format)[-2]), 'Hips-T_{}'.format(str(data_format)[-1]),
                                    'Hips-R_{}'.format(str(data_format)[-3]), 'Hips-R_{}'.format(str(data_format)[-2]), 'Hips-R_{}'.format(str(data_format)[-1]),
                                    'RUL-T_{}'.format(str(data_format)[-3]), 'RUL-T_{}'.format(str(data_format)[-2]), 'RUL-T_{}'.format(str(data_format)[-1]),
                                    'RUL-R_{}'.format(str(data_format)[-3]), 'RUL-R_{}'.format(str(data_format)[-2]), 'RUL-R_{}'.format(str(data_format)[-1]),
                                    'RK-T_{}'.format(str(data_format)[-3]), 'RK-T_{}'.format(str(data_format)[-2]), 'RK-T_{}'.format(str(data_format)[-1]),
                                    'RK-R_{}'.format(str(data_format)[-3]), 'RK-R_{}'.format(str(data_format)[-2]), 'RK-R_{}'.format(str(data_format)[-1]),
                                    'RF-T_{}'.format(str(data_format)[-3]), 'RF-T_{}'.format(str(data_format)[-2]), 'RF-T_{}'.format(str(data_format)[-1]),
                                    'RF-R_{}'.format(str(data_format)[-3]), 'RF-R_{}'.format(str(data_format)[-2]), 'RF-R_{}'.format(str(data_format)[-1]),
                                    'LUL-T_{}'.format(str(data_format)[-3]), 'LUL-T_{}'.format(str(data_format)[-2]), 'LUL-T_{}'.format(str(data_format)[-1]),
                                    'LUL-R_{}'.format(str(data_format)[-3]), 'LUL-R_{}'.format(str(data_format)[-2]), 'LUL-R_{}'.format(str(data_format)[-1]),
                                    'LK-T_{}'.format(str(data_format)[-3]), 'LK-T_{}'.format(str(data_format)[-2]), 'LK-T_{}'.format(str(data_format)[-1]),
                                    'LK-R_{}'.format(str(data_format)[-3]), 'LK-R_{}'.format(str(data_format)[-2]), 'LK-R_{}'.format(str(data_format)[-1]),
                                    'LF-T_{}'.format(str(data_format)[-3]), 'LF-T_{}'.format(str(data_format)[-2]), 'LF-T_{}'.format(str(data_format)[-1]),
                                    'LF-R_{}'.format(str(data_format)[-3]), 'LF-R_{}'.format(str(data_format)[-2]), 'LF-R_{}'.format(str(data_format)[-1])])

"""
