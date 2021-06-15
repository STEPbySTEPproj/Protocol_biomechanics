from numpy.lib.function_base import msort
import pandas as pd
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, freqz, filtfilt
import re
import pickle
from ruamel.yaml import YAML
import sys
from termcolor import colored
from pkgutil import get_loader
import math
#import matplotlib.pyplot as plt


# Check https://stackoverflow.com/questions/5003755/how-to-use-pkgutils-get-data-with-csv-reader-in-python

def get_data_smart(package, resource, as_string=True):
    """Rewrite of pkgutil.get_data() that actually lets the user determine if data should
    be returned read into memory (aka as_string=True) or just return the file path.
    """

    loader = get_loader(package)
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, '__file__'):
        return None

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.join(*parts)
    if as_string:
        return loader.get_data(resource_name)
    else:
        return resource_name

def smooth_signal(data):
    """[summary]
    This function applies a smoothing filter to the data
    Args:
        data (Pandas DataFrame): The dataframe containing the original  data

    Returns:
        data (Pandas DataFrame): Smoothed data
    """
    for col in data.columns:
        data[col] = butter_lowpass_filter(data[col], 2, 60)
    return data

def butter_lowpass(cutoff, fs, order=5):
    """[summary]
    Calculating butterworth filter parameters for smoothing the signal
    Args:
        cutoff (Scalar): Parameter to control the time window that is going to be smoothed.
        fs (Scalar): Sample rate

    Returns:
        data (Pandas DataFrame): Smoothed data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, fs=None)
    return b, a

def butter_lowpass_filter(x, cutoff, fs, order=5):
    """[summary]
    This function applies butterworth filter to the data in both directions
    Args:
        x (Pandas DataFrame): One variable of the dataframe
        cutoff (Scalar): Parameter to control the time window that is going to be smoothed.
        fs (Scalar): Sample rate

    Returns:
        data (Pandas DataFrame): Smoothed data
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, x)


def extract_features(data):
    """[summary]
    This function returns the action sequence and indexes where changes 
    occur.
    Args:
        data (Pandas DataFrame): The dataframe containing the original 
                                 or cleaned data

    Returns:
        action_sequence [List]: Contains the cluster ID of the action 
                                executed.
        change_index [List]: Contains the frame ID where the action
                             contained in action_sequence
    """
    change_index = []
    action_sequence = []
    cur_value = data.iloc[0].values
    action_sequence.append(str(cur_value[0]))
    for i in range(len(data)):
        if(cur_value != data.iloc[i].values):
            cur_value = data.iloc[i].values
            change_index.append(i)
            action_sequence.append(str(cur_value[0]))
    
    return action_sequence, change_index

def generate_gait_seq(gait_seq, action_seq, index_seq):
    """[summary]
    From a sequence of numbers, it emits the position of the indices 
    in which a specific sequence is found
    Args:
        gait_seq [List]: The desired digit sequence
        action_sequence [List]: Contains the cluster ID of the action 
                                executed.
        change_index [List]: Contains the frame ID where the action
                             contained in action_sequence
    Returns:
        gait_list [List]: The position of the searched 
                            sequence in the index list
    """
    gait_length = len(gait_seq)
    gait_list = [int(m.start()) for m in re.finditer(gait_seq,
                                                action_seq)]
    return gait_list

def clean_data(data, n_frames_tol = 5):
    """[summary]
    Optional step. Consists on removing the miscalculations, understanding
    miscalculations as a couple of frames that are misclassified. 
    
    Args:
        data (Pandas DataFrame): The dataframe containing the original 
                                 data
        n_frames_tol (Int): The maximum ammount of frames of tolerance
                            to detect a misclassification.

    Returns:
        cleaned_data [Pandas DataFrame]: The dataframe after removing the 
                                         noise or misclassifications.
    """
    change_index = []
    change_ii = 0
    cur_value = data.iloc[0].values
    change_index.append(cur_value)
    cleaned_data = data.copy()
    for i in range(len(data)):
        if(cur_value != data.iloc[i].values):
            if(i - n_frames_tol > change_index[change_ii]):
                change_ii = change_ii + 1
                change_index.append(i)
            else:
                repeat_val = np.repeat( data.iloc[i].values, i - 
                                        change_index[change_ii])
                repeat_val = np.expand_dims(repeat_val, 1)
                cleaned_data[change_index[change_ii]:i] = repeat_val
            cur_value = data.iloc[i].values
    return cleaned_data

def yaml_export(data, output_dir, category, datatype):
        """[summary]
        Create a yaml file
        Args:
            data (Pandas DataFrame): The dataframe containing the original 
                                    or cleaned data
            output_dir (str): Output directory
            category (str): Name of the metric
            datatype (str): Type of the data, scalar or list.


        Returns:
            action_sequence [List]: Contains the cluster ID of the action 
                                    executed.
            change_index [List]: Contains the frame ID where the action
                                contained in action_sequence
        """

        tmp_dict = dict(type=datatype, value=data.tolist())

        yaml = YAML()
        if tmp_dict['type'] == 'vector':
            yaml.default_flow_style = None

        with open('{}/{}.yml'.format(output_dir,category), 'w') as file:
            _ = yaml.dump(tmp_dict, file)


def PI_Sensors_LUL(sensors_data):
    """
    Maximum, minimum, mean and std of the LUL sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['LUL-R_Y']), \
            np.min(sensors_data['LUL-R_Y']), \
            np.mean(sensors_data['LUL-R_Y']), \
            np.std(sensors_data['LUL-R_Y'])

def PI_Sensors_LK(sensors_data):
    """
    Maximum, minimum, mean and std of the LK sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['LK-R_Y']), \
            np.min(sensors_data['LK-R_Y']), \
            np.mean(sensors_data['LK-R_Y']), \
            np.std(sensors_data['LK-R_Y'])

def PI_Sensors_LF(sensors_data):
    """
    Maximum, minimum, mean and std of the LF sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['LF-R_Y']), \
            np.min(sensors_data['LF-R_Y']), \
            np.mean(sensors_data['LF-R_Y']), \
            np.std(sensors_data['LF-R_Y'])


def PI_Sensors_RUL(sensors_data):
    """
    Maximum, minimum, mean and std of the RUL sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['RUL-R_Y']), \
            np.min(sensors_data['RUL-R_Y']), \
            np.mean(sensors_data['RUL-R_Y']), \
            np.std(sensors_data['RUL-R_Y'])

def PI_Sensors_RK(sensors_data):
    """
    Maximum, minimum, mean and std of the RK sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['RK-R_Y']), \
            np.min(sensors_data['RK-R_Y']), \
            np.mean(sensors_data['RK-R_Y']), \
            np.std(sensors_data['RK-R_Y'])

def PI_Sensors_RF(sensors_data):
    """
    Maximum, minimum, mean and std of the RF sensor
    Args:
            data (Pandas DataFrame): The dataframe containing the original data
    Returns:
            Maximum (scalar)
            Minumum (scalar)
            Mean(scalar)
            Std(scalar)
    """
    return  np.max(sensors_data['RF-R_Y']), \
            np.min(sensors_data['RF-R_Y']), \
            np.mean(sensors_data['RF-R_Y']), \
            np.std(sensors_data['RF-R_Y'])

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
#import matplotlib.pyplot as plt

def Generate_PI(argv):


    USAGE = """usage: run_pi file_in folder_out
            file_in: csv file containing at least a timestamp column
            folder_out: folder where the PI yaml file yaml file will be stored
            """
    if len(argv) != 3:
        print(colored("Wrong input parameters !", "red"))
        print(colored(USAGE, "yellow"))
        sys.exit(-1)

    file_in = argv[1]
    output_dir = argv[2]

    # check input parameters are good
    if not os.path.exists(file_in):
        print(colored("Input file {} does not exist".format(file_in), "red"))
        sys.exit(-1)
    if not os.path.isfile(file_in):
        print(colored("Input path {} is not a file".format(file_in), "red"))
        sys.exit(-1)

    if not os.path.exists(output_dir):
        print(colored(
            "Output folder {} does not exist".format(output_dir),
            "red"))
        sys.exit(-1)
    if not os.path.isfile(file_in):
        print(colored("Output path {} is not a folder".format(file_in), "red"))
        sys.exit(-1)
        

    #Loading and preparing the data
    #print(file_in)

    try:
        test = pd.read_csv(file_in)
    except FileNotFoundError as err:
        print(colored(
            "Could not load file {}: {}".format(file_in, err),
            "red"))
        sys.exit(-1)

    #Delete label and frame_idx columns
    if(test.columns[-1]=='Label'):
        test=test.drop(columns='Label')
    #test=test.drop(columns='Frame_Idx')
    test2=smooth_signal(test)

    '''
    #Classify the movement as ascend or descend
    data_path = get_data_smart(__name__, "tests/data/protocol_1/input/models/model_descend_ascend.sav", False)
    clas_ascend_descend = pickle.load(open(data_path, 'rb'))
    descend = clas_ascend_descend.predict(test[0:500].to_numpy().flatten().reshape(1,-1))

    #Loading the correct model
    if(descend==1):
        #print("Loading descend model")
        data_path = get_data_smart(__name__, "tests/data/protocol_1/input/models/model_descend.sav", False)
        model=pickle.load(open(data_path, 'rb'))
    else:
        #print("Loading ascend model")
        data_path = get_data_smart(__name__, "tests/data/protocol_1/input/models/model_ascend.sav", False)
        model=pickle.load(open(data_path, 'rb'))
    '''

    '''
    y=smooth(test["LK-R_X"],19)
    plt.plot(test, 'g-', lw=2)
    plt.plot(y, 'r-', lw=2)
    plt.show()
    '''  
    y=test["RK-R_X"]
    x=np.arange(0,len(y))
    crop_x = 0
    x=x[crop_x:len(x)]
    y=y[crop_x:len(y)]
    
    threshold=0.7
    ydiff = np.diff(y)
    ydiff=np.append(ydiff,0)
    '''
    plt.plot(x,ydiff)
    plt.show()
    '''
    less1 = abs(ydiff) < threshold
    ydiff[less1] = 0
    more1 = abs(ydiff) > threshold
    ydiff[more1] = 1
    ydiff= [int(x) for x in ydiff]
    '''
    plt.plot(x,ydiff)
    plt.show()
    '''
    str_int="".join(map(str, ydiff))
    num_zeros=300
    num_zeros_str=""
    for i in range(num_zeros):
        num_zeros_str=num_zeros_str+"0"

    ixflatstart = [i.start() for i in re.finditer("1"+num_zeros_str, str_int)]
    ixflatstart = [n + crop_x for n in ixflatstart]
    # the return value is the index of the first zero, so we need to add zeros to get the index of 1.
    ixflatend = [i.start() for i in re.finditer(num_zeros_str+"1", str_int)]
    ixflatend = [n + num_zeros + crop_x for n in ixflatend]
    '''
    plt.plot(x,y, 'g-', lw=2)
    plt.plot(ixflatstart, y[ixflatstart],'r.')
    plt.plot(ixflatend, y[ixflatend],'b.')
    plt.show()
    '''
    index_stop_ascending = ixflatstart[0] + math.floor((ixflatend[1]-ixflatstart[0])/2)
    index_start_descending = ixflatstart[1] + math.floor((ixflatend[2]-ixflatstart[1])/2)
    '''
    plt.figure(figsize=(12, 5))
    plt.plot(x,y, color='grey')
    plt.plot(x[index_stop_ascending],y[index_stop_ascending],"o",color='r')
    plt.plot(x[index_start_descending],y[index_start_descending],"o",color='r')
    plt.show()
    '''

    frecuencia=120
    # --------------- Ascending model -------------------------------
    #print("Loading ascend model")
    data_path = get_data_smart(__name__, "tests/data/protocol_1/input/models/model_ascend.sav", False)
    model=pickle.load(open(data_path, 'rb'))
    test2=test.iloc[0:index_stop_ascending]

    #The variables are chosen and a window is created for each observation
    S_test=test2[['RK-R_X','LK-R_X']].to_numpy()
    datos=[]
    window=30
    for i in range(window,S_test.shape[0]-window):
        datos.append(S_test[i-window:i+window,:].flatten())
    X_test=np.asarray(datos)

    #The prediction is performed and the observations removed due to the window are added again
    pred=model.predict(X_test)
    pred=np.asarray(pred,dtype=int)    
    pred=list(np.pad(pred, (window, window), 'constant'))

    #Estimate events based on the sequence of predictions, 67 is left leg and 34 rigth leg
    action_seq = clean_data(pd.DataFrame(pred))
    action_seq, index_seq = extract_features(action_seq)

    separator = ''
    action_seq = separator.join(action_seq)
    list_gait_seq = ["67","34"]

    # Create lists of Left/Rigth Toe-Off, Mid-swing and Foot-Strike
    L_TO=[]
    R_TO=[]

    L_MS=[]
    R_MS=[]

    L_FS=[]
    R_FS=[]

    #Define the Toe-Off, Mid-swing and Foot-Strike as the first, second and third element of each sequence
    for i in list_gait_seq:
        gait_seq=i
        f=generate_gait_seq(gait_seq, action_seq, index_seq)
        #print(f)
        for k in f:
            if(gait_seq=="67"):
                L_TO.append(index_seq[k-1])
                L_MS.append(index_seq[k])
                L_FS.append(index_seq[k+1])
            elif(gait_seq=="34"):
                R_TO.append(index_seq[k-1])
                R_MS.append(index_seq[k])
                R_FS.append(index_seq[k+1])
                

    # Decide which leg has gone first and assign it to the events to the P_TO, P_MS 
    #   and P_FS lists and the second leg to the S lists S_TO, S_MS and S_FS
    P_TO=[]
    S_TO=[]

    P_MS=[]
    S_MS=[]

    P_FS=[]
    S_FS=[]

    if(R_TO[0]>=L_TO[0]):
        P_TO= L_TO
        S_TO= R_TO

        P_MS= L_MS
        S_MS= R_MS

        P_FS= L_FS
        S_FS= R_FS
    elif(L_TO[0]>R_TO[0]):
        P_TO= R_TO
        S_TO= L_TO

        P_MS= R_MS
        S_MS= L_MS

        P_FS= R_FS
        S_FS= L_FS

    # Calculate gate phase events
    # Ipsilateral Foot-Strike1: IFS1
    # Contralateral Toe-Off: CTO
    # Mid-swing1: MS1
    # Contralateral Foot-Strike: CFS
    # Ipsilateral Toe-Off: ITO
    # Mid-swing2: MS2
    # Ipsilateral Foot-Strike2: IFS2

    IFS1 = []
    CTO = []
    MS1 = []
    CFS = []
    ITO = []
    MS2 = []
    IFS2 = []

    for i in range(len(P_FS)-1):
        IFS1.append(P_FS[i])
        ITO.append(P_TO[i+1])
        MS2.append(P_MS[i+1])

    for i in range(len(S_FS)):
        CTO.append(S_TO[i])
        MS1.append(S_MS[i])
        CFS.append(S_FS[i])

    for i in range(1, len(P_FS)):
        IFS2.append(P_FS[i])

    #Remove the last indexes if they are not the same length. Delete the last value since a whole phase has not been added.
    len1 = len(IFS1)
    len2 = len(CTO)
    len3 = len(IFS2)
    if(len1 != len2 or len1 != len3 or len2 != len3):
        min_value = min(len1, len2, len3)
        if (len1 > min_value):
            IFS1.pop()
            ITO.pop()
            MS2.pop()
        if (len2 > min_value):
            CTO.pop()
            MS1.pop()
            CFS.pop()
        if (len3 > min_value):
            IFS2.pop()        


    #Ascend metrics
    total_time = IFS2[-1]-IFS1[0]
    asc_total_time = []
    asc_total_time.append((total_time)/frecuencia)
    
    yaml_export(np.float64(asc_total_time[0]), output_dir,  'ascending_total_time', 'scalar')
    
    #weight_acceptance_ascend 

    results=[]

    for i in range(len(CTO)):
        results.append((CTO[i]-IFS1[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'ascending_weight_acceptance', 'scalar')

    #pull up 

    results=[]

    for i in range(len(MS1)):
        results.append((MS1[i]-CTO[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'ascending_pull_up', 'scalar')

    # forward_continuance_ascend

    results=[]

    for i in range(len(CFS)):
        results.append((CFS[i]-MS1[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'ascending_forward_continuance', 'scalar')


    # push up

    results=[]

    for i in range(len(ITO)):
        results.append((ITO[i]-CFS[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'ascending_push_up', 'scalar')


    # swing_foot_clearance

    results=[]

    for i in range(len(MS2)):
        results.append((MS2[i]-ITO[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'ascending_swing_foot_clearance', 'scalar')


    # swing_foot_placement
    results=[]

    for i in range(len(IFS2)):
        results.append((IFS2[i]-MS2[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'ascending_swing_foot_placement', 'scalar')


    # --------------- Descending model -------------------------------
    #print("Loading descend model")
    data_path = get_data_smart(__name__, "tests/data/protocol_1/input/models/model_descend.sav", False)
    model=pickle.load(open(data_path, 'rb'))
    test2=test.iloc[index_start_descending:len(test)-1]

    #The variables are chosen and a window is created for each observation
    S_test=test2[['RK-R_X','LK-R_X']].to_numpy()
    datos=[]
    window=30
    for i in range(window,S_test.shape[0]-window):
        datos.append(S_test[i-window:i+window,:].flatten())
    X_test=np.asarray(datos)

    #The prediction is performed and the observations removed due to the window are added again
    pred=model.predict(X_test)
    pred=np.asarray(pred,dtype=int)
    pred=list(np.pad(pred, (window, window), 'constant'))

    #Estimate events based on the sequence of predictions, 67 is left leg and 34 rigth leg
    action_seq = clean_data(pd.DataFrame(pred))
    action_seq, index_seq = extract_features(action_seq)

    separator = ''
    action_seq = separator.join(action_seq)
    list_gait_seq = ["67","34"]

    # Create lists of Left/Rigth Toe-Off, Mid-swing and Foot-Strike
    L_TO=[]
    R_TO=[]

    L_MS=[]
    R_MS=[]

    L_FS=[]
    R_FS=[]

    #Define the Toe-Off, Mid-swing and Foot-Strike as the first, second and third element of each sequence
    for i in list_gait_seq:
        gait_seq=i
        f=generate_gait_seq(gait_seq, action_seq, index_seq)
        #print(f)
        for k in f:
            if(gait_seq=="67"):
                L_TO.append(index_seq[k-1])
                L_MS.append(index_seq[k])
                L_FS.append(index_seq[k+1])
            elif(gait_seq=="34"):
                R_TO.append(index_seq[k-1])
                R_MS.append(index_seq[k])
                R_FS.append(index_seq[k+1])
                

    # Decide which leg has gone first and assign it to the events to the P_TO, P_MS 
    #   and P_FS lists and the second leg to the S lists S_TO, S_MS and S_FS
    P_TO=[]
    S_TO=[]

    P_MS=[]
    S_MS=[]

    P_FS=[]
    S_FS=[]

    if(R_TO[0]>=L_TO[0]):
        P_TO= L_TO
        S_TO= R_TO

        P_MS= L_MS
        S_MS= R_MS

        P_FS= L_FS
        S_FS= R_FS
    elif(L_TO[0]>R_TO[0]):
        P_TO= R_TO
        S_TO= L_TO

        P_MS= R_MS
        S_MS= L_MS

        P_FS= R_FS
        S_FS= L_FS

    
    # Calculate gate phase events
    # Foot-Strike1: FS1
    # Toe-Off1: TO1
    # Mid-swing1: MS1
    # Toe-Off2: TO2
    # Mid-swing2: MS2
    # Foot-Strike2: FS2

    FS1 = []
    TO1 = []
    MS1 = []
    TO2 = []
    MS2 = []
    FS2 = []

    for i in range(len(P_FS)-1):
        FS1.append(P_FS[i])
    
    for i in range(1, len(P_FS)):
        TO2.append(P_TO[i])
        MS2.append(P_MS[i])

    for i in range(len(S_FS)):
        TO1.append(S_TO[i])
        MS1.append(S_MS[i])

    for i in range(1, len(P_FS)):
        FS2.append(P_FS[i])



    #Descend metrics
    total_time = FS2[-1]-FS1[0]
    des_total_time = []
    des_total_time.append((total_time)/frecuencia)

    #yaml_export(np.mean([S_FS[-1]-P_FS[0]])/frecuencia, output_dir,  'descending_total_time', 'scalar')
    yaml_export(np.float64(des_total_time[0]), output_dir,  'descending_total_time', 'scalar')

    
    #weight_acceptance_descend 

    results=[]

    for i in range(len(TO1)):
        results.append((TO1[i]-FS1[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'descending_weight_acceptance', 'scalar')
    
    # forward_continuance_descend 

    results=[]

    for i in range(len(MS1)):
        results.append((MS1[i]-TO1[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'descending_forward_continuance', 'scalar')
    
    # Controlled_lowering_descend
    
    results=[]

    for i in range(len(TO2)):
        results.append((TO2[i]-MS1[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'descending_controlled_lowering', 'scalar')
    
    #  Leg_pull_through_descend

    results=[]

    for i in range(len(MS2)):
        results.append((MS2[i]-TO2[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'descending_swing_leg_pull_through', 'scalar')

    # Foot_placement_descend
    results=[]

    for i in range(len(FS2)):
        results.append((FS2[i]-MS2[i])/(total_time))

    results = [0 if i < 0 else i for i in results]

    yaml_export(np.sum(results), output_dir,  'descending_swing_foot_placement', 'scalar')

    # Ascending model
    # Reading again the original file
    test=pd.read_csv(file_in)
    test=test[0:index_stop_ascending]
    #Calculate the metrics of maximum, minimum, mean and std of various variables

    lul_max, lul_min, lul_mean, lul_std = PI_Sensors_LUL(test)
    lul_vector = np.around(np.array([lul_min, lul_max, lul_mean, lul_std]),5)
    yaml_export(lul_vector, output_dir,'ascending_hip_left_angle', 'vector')

    lk_max, lk_min, lk_mean, lk_std = PI_Sensors_LK(test)
    lk_vector = np.around(np.array([lk_min, lk_max, lk_mean, lk_std]),5)
    yaml_export(lk_vector, output_dir,'ascending_knee_left_angle', 'vector')

    lf_max, lf_min, lf_mean, lf_std = PI_Sensors_LF(test)
    lf_vector = np.around(np.array([lf_min, lf_max, lf_mean, lf_std]),5)
    yaml_export(lf_vector, output_dir,'ascending_ankle_left_angle', 'vector')

    rul_max, rul_min, rul_mean, rul_std = PI_Sensors_RUL(test)
    rul_vector = np.around(np.array([rul_min, rul_max, rul_mean, rul_std]),5)
    yaml_export(rul_vector, output_dir,'ascending_hip_right_angle', 'vector')

    rk_max, rk_min, rk_mean, rk_std = PI_Sensors_RK(test)
    rk_vector = np.around(np.array([rk_min, rk_max, rk_mean, rk_std]),5)
    yaml_export(rk_vector, output_dir,'ascending_knee_right_angle', 'vector')

    rf_max, rf_min, rf_mean, rf_std = PI_Sensors_RF(test)
    rf_vector = np.around(np.array([rf_min, rf_max, rf_mean, rf_std]),5)
    yaml_export(rf_vector, output_dir,'ascending_ankle_right_angle', 'vector')
    
    # Descending model
    # Reading again the original file
    test=pd.read_csv(file_in)
    test=test[index_start_descending:len(test)-1]
    #Calculate the metrics of maximum, minimum, mean and std of various variables

    lul_max, lul_min, lul_mean, lul_std = PI_Sensors_LUL(test)
    lul_vector = np.around(np.array([lul_min, lul_max, lul_mean, lul_std]),5)
    yaml_export(lul_vector, output_dir,'descending_hip_left_angle', 'vector')

    lk_max, lk_min, lk_mean, lk_std = PI_Sensors_LK(test)
    lk_vector = np.around(np.array([lk_min, lk_max, lk_mean, lk_std]),5)
    yaml_export(lk_vector, output_dir,'descending_knee_left_angle', 'vector')

    lf_max, lf_min, lf_mean, lf_std = PI_Sensors_LF(test)
    lf_vector = np.around(np.array([lf_min, lf_max, lf_mean, lf_std]),5)
    yaml_export(lf_vector, output_dir,'descending_ankle_left_angle', 'vector')

    rul_max, rul_min, rul_mean, rul_std = PI_Sensors_RUL(test)
    rul_vector = np.around(np.array([rul_min, rul_max, rul_mean, rul_std]),5)
    yaml_export(rul_vector, output_dir,'descending_hip_right_angle', 'vector')

    rk_max, rk_min, rk_mean, rk_std = PI_Sensors_RK(test)
    rk_vector = np.around(np.array([rk_min, rk_max, rk_mean, rk_std]),5)
    yaml_export(rk_vector, output_dir,'descending_knee_right_angle', 'vector')

    rf_max, rf_min, rf_mean, rf_std = PI_Sensors_RF(test)
    rf_vector = np.around(np.array([rf_min, rf_max, rf_mean, rf_std]),5)
    yaml_export(rf_vector, output_dir,'descending_ankle_right_angle', 'vector')
    
    
    print("The metrics generation finished successfully")