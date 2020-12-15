from common.bone_structure import Bone_ID, Data_Format
from common.data_structure import BvhDataHeader as BVH
from common.LoadDLL import bvhFrameDataReceived as frame_received
from common.LoadDLL import init_load, verify_connection
from ctypes import *
import os
import time

def conn_has_finished():
    global is_finished
    return is_finished

def conn_second_conn():
    global is_finished
    is_finished = False
    init_load(seconds=20, mode_in = 2, bvh_format = Data_Format.XZY)

def conn_main():
    global is_finished
    is_finished = False
    if(not is_finished):
        
        '''
        * Define max seconds
        * Modes to create Skeleton:
            * mode = 0 Hips + Left Leg
            * mode = 1 Hips + Right Leg
            * mode = 2 Hips + Both Legs
        * Format to receive data:
            * Enum related to this aspect.
        '''
        init_load(seconds = 20, mode_in = 2, bvh_format = Data_Format.XZY)

        lib = "..\\StepByStepReader\\NeuronDataReaderLib\\lib\\x64\\NeuronDataReader.dll"
        ip = str("127.0.0.1").encode('utf-8') #Cambio     127.0.0.1
        buf = create_unicode_buffer(544)
        port = c_int(8012)  #CAMBIO 7001  
        null_ptr = POINTER(c_int)()
        CALLBACKFUNC = WINFUNCTYPE(c_bool, c_void_p, c_void_p, POINTER(BVH), POINTER(c_float))
        cb_func = CALLBACKFUNC(frame_received)

        libc = cdll.LoadLibrary(lib)
        conn = libc.BRConnectTo(ip, port)
        callback = libc.BRRegisterFrameDataCallback(null_ptr,cb_func)

        while(not is_finished):    
            # Avoid overcharging the CPU with the open process
            is_finished = verify_connection()
            time.sleep(3)
            #if(is_finished):
            #    return is_finished

conn_main()