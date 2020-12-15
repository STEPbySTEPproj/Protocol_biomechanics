from ctypes import c_uint8,c_uint16,c_uint32,c_char, Structure, Union

class VER_Struct(Structure):
    _fields_ = [("BuildNumb",   c_uint8),
                ("Revision",    c_uint8),
                ("Minor",       c_uint8),
                ("Major",       c_uint8)]

class DATA_VER(Union):
    _anonymous_ = ('u',)
    _fields_ = [("_VersionMask",    c_uint32), 
                ('u',               VER_Struct)]
                
class BvhDataHeader(Structure):
    _pack_ = 1
    _fields_ = [("Token1",          c_uint16),
                ("DataVersion",     DATA_VER),
                ("DataCount",       c_uint16),
                ("WithDisp",        c_uint8),
                ("WithReference",   c_uint8),
                ("AvatarIndex",     c_uint32),
                ("AvatarName",      c_char * 32),
                ("FrameIndex",      c_uint32),
                ("Reserved",        c_uint32),
                ("Reserved1",       c_uint32),
                ("Reserved2",       c_uint32),
                ("Token2",          c_uint16)]

