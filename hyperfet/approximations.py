import numpy as np
from scipy.special import lambertw as complex_lambertw

def lambertw(x):
    return np.real(complex_lambertw(x))

def shorthands(hyperfet, VD,VG, *args, gridinput=True):
    if gridinput:
        VD,VG=np.meshgrid(VD,VG)

    # Component devices
    mosfet=hyperfet.mosfet
    pcr=hyperfet.pcr

    # HyperFET shorthands
    d={}
    d['VTp']=VTp=mosfet.VT0-mosfet.delta*VD
    d['VTm']=VTm=VTp-mosfet.alpha*mosfet.Vth
    d['R_insp']=R_insp=pcr.R_ins*(1+mosfet.delta)
    d['R_metp']=R_metp=pcr.R_met*(1+mosfet.delta)
    d['app_Ioff']=app_Ioff=mosfet.n*mosfet.k*mosfet.Vth*np.exp(-VTm/(mosfet.n*mosfet.Vth))

    # Assemble requested variables form above dictionary or the component devices
    res=[VD,VG]
    for a in args:
        if a in d: res+=[d[a]]
        elif hasattr(mosfet,a): res+=[getattr(mosfet,a)]
        elif hasattr(pcr,a): res+=[getattr(pcr,a)]
        else: raise Exception("Not found: "+a)
    return res

def leakagefloor(hyperfet,VD,VG):
    VD,VG,R_ins,Gleak=shorthands(hyperfet,VD,VG,"R_ins","Gleak")
    return 1/(R_ins+1/Gleak)*VD

def lowerbranch(hyperfet,VD,VG):
    VD,VG,n,Vth,k,VTm,R_ins,R_insp,Gleak=shorthands(hyperfet,VD,VG,"n","Vth","k","VTm","R_ins","R_insp","Gleak")
    return n*Vth/R_insp*lambertw(k*R_insp/(1+Gleak*R_ins) \
         *np.exp((VG-VTm-(Gleak*VD*R_ins)/(1+Gleak*R_ins))/(n*Vth))) \
           +(Gleak*VD/(1+Gleak*R_ins))

def lowernoleak(hyperfet,VD,VG):
    VD,VG,n,Vth,k,VTm,R_insp=shorthands(hyperfet,VD,VG,"n","Vth","k","VTm","R_insp")
    return n*Vth/R_insp*lambertw(k*R_insp*np.exp((VG-VTm)/(n*Vth)))

def upperbranchsubthresh(hyperfet,VD,VG):
    VD,VG,n,Vth,k,VTm,R_metp,V_met=shorthands(hyperfet,VD,VG,"n","Vth","k","VTm","R_metp","V_met")
    return n*Vth/R_metp*lambertw(k*R_metp*np.exp((VG-V_met-VTm)/(n*Vth)))

def upperbranchinversion(hyperfet,VD,VG):
    VD,VG,k,VTp,R_metp,V_met=shorthands(hyperfet,VD,VG,"k","VTp","R_metp","V_met")
    return k*(VG-V_met-VTp)/(1+k*R_metp)

def Vleft(hyperfet,VD):
    VD,VG,k,VTm,Ff,alpha,delta,V_MIT,n,Vth,I_MIT=\
        shorthands(hyperfet,VD,None,"k","VTm","Ff","alpha","delta","V_MIT","n","Vth","I_MIT",gridinput=False)
    Vleft0=VTm+(1+delta)*V_MIT-n*Vth*np.log(n*k*Vth/I_MIT)
    return Vleft0+alpha*Vth*(1-Ff(VD-V_MIT,Vleft0-V_MIT))

def Vright(hyperfet,VD):
    VD,VG,VTm,delta,V_IMT,n,Vth,k,I_IMT= \
        shorthands(hyperfet,VD,None,"VTm","delta","V_IMT","n","Vth","k","I_IMT",gridinput=False)
    return VTm+(1+delta)*V_IMT-n*Vth*np.log(n*k*Vth/I_IMT)

def shift(hyperfet,VD):
    VD,VG,app_Ioff,R_insp=shorthands(hyperfet,VD,None,"app_Ioff","R_insp",gridinput=False)
    return -app_Ioff*R_insp

def shiftedgain(self):
    return (1+(self._approx_Ioff*self.R_insp-self.vo2.V_met)/(self.VDD-self.VTp))/(1+self.mosfet.k*self.R_metp)

def shiftedsr(self):
    return (self.VDD-self.VTp)*self.mosfet.k*self.R_metp+self.vo2.V_met-self._approx_Ioff*self.R_insp
