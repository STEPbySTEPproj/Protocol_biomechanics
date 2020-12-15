from enum import Enum
import numpy as np
from .bone_structure import Bone_ID

class Bone:
    def __init__(self, idx, t_x = 0, t_y = 0, t_z = 0, r_x = 0, r_y = 0, r_z = 0):
        self.id = idx
        self.translation = np.array([t_x, t_y, t_z])
        self.rotation = np.array([r_x, r_y, r_z])

    def get_rotation(self):
        return self.rotation

    def get_position(self):
        return self.translation

class Skeleton:
    def __init__(self, mode = 0):
        self.mode = mode
        if(mode == 0): # Left leg
            self.ids = np.array([Bone_ID.Hips, Bone_ID.LeftUpLeg, 
                                    Bone_ID.LeftLeg, Bone_ID.LeftFoot])
        if(mode == 1): # Right leg
            self.ids = np.array([Bone_ID.Hips, Bone_ID.RightUpLeg, 
                                    Bone_ID.RightLeg, Bone_ID.RightFoot])

        if(mode == 2): # Both legs
            self.ids = np.array([Bone_ID.Hips, Bone_ID.RightUpLeg, 
                                    Bone_ID.RightLeg, Bone_ID.RightFoot,
                                    Bone_ID.LeftUpLeg, Bone_ID.LeftLeg,
                                    Bone_ID.LeftFoot])
        self.bones = np.empty((self.ids.shape[0],6))
        for idx in range(len(self.ids)):
            self.bones[idx] = np.zeros(6)

    def set_bone(self, idx, value):
        self.bones[idx] = value
    
    def get_bone(self, idx):
        return self.bones[idx]
        
    def get_mode(self):
        return self.mode

    def get_ids(self):
        return self.ids
    
    def __len__(self):
        return len(self.ids)

class Frame: 
    def __init__(self, idx, skeleton_frame):
        self.frame_id = idx
        self.skeleton = skeleton_frame

    def get_id(self):
        return self.frame_id

    def get_skeleton(self):
        return self.skeleton