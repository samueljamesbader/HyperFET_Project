import numpy as np
from hyperfet.devices import Direction

def left(VG,If,Ib):
    hyst=(~np.isclose(If,Ib))
    i=np.argmin(VG[hyst])
    return VG[hyst][i],If[hyst][i],Ib[hyst][i]

def right(VG,If,Ib):
    hyst=~np.isclose(If,Ib)
    i=np.argmax(VG[hyst])
    return VG[hyst][i],If[hyst][i],Ib[hyst][i]

def is_point_hysteretic(hyperfet,VD,VG):
    If,Ib=hyperfet.I_double(VD,VG)
    return not np.isclose(If,Ib)

def boundaries_nonhysteretic(hyperfet,VDD):
    return not (is_point_hysteretic(hyperfet,VDD,0) or is_point_hysteretic(hyperfet,VDD,VDD))

def shift(VG,hyperfet,If,Ib,I,VDD):
    Vl=left(VG,If,Ib)[0]
    assert Vl>0, "Curve already hysteretic at VG=0, left-shifting cannot improve device."
    i=np.argmin(Ib-np.ravel(I)[0])
    Vshift=VG[i]
    if Vshift>Vl:
        print("Shifting limited by left-hysteretic bound.")
        Vshift=Vl-(VG[1]-VG[0])
    assert not is_point_hysteretic(hyperfet,VDD,VDD), "Curve still hysteretic at VG=VDD"
    return Vshift, hyperfet.I(VDD,VG+Vshift,Direction.FORWARD)


    #assert is_point_hysteretic(hyperfet,VD=VDD,VG=0)
