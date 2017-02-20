import numpy as np

def left(VG,If,Ib):
    hyst=~np.isclose(If,Ib)
    i=np.argmin(VG[hyst])
    return VG[hyst][i],If[hyst][i],Ib[hyst][i]
def right(VG,If,Ib):
    hyst=~np.isclose(If,Ib)
    i=np.argmax(VG[hyst])
    return VG[hyst][i],If[hyst][i],Ib[hyst][i]
