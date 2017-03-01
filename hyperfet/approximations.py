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
    #d['app_Ioff']=app_Ioff=mosfet.n*mosfet.k*mosfet.Vth*np.exp(-VTm/(mosfet.n*mosfet.Vth))
    d['app_Ioff']=app_Ioff=mosfet.ID(VD,0)

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

def Ill(hyperfet,VD):
    VD,VG,n,Vth,R_insp,I_MIT,V_MIT=\
        shorthands(hyperfet,VD,None,"n","Vth","R_insp","I_MIT","V_MIT",gridinput=None)
    return (n*Vth/R_insp)*lambertw((I_MIT*R_insp/(n*Vth))*np.exp(V_MIT/(n*Vth)))

def shift(hyperfet,VD,boundary='clipapprox'):
    VD,VG,app_Ioff,R_insp=shorthands(hyperfet,VD,None,"app_Ioff","R_insp",gridinput=False)
    shift=-app_Ioff*R_insp
    return shift
    #if boundary=='clipapprox':
    #    max1=Vright(hyperfet,VD)
    #    if +shift > VD:
    #        return None
    #    else:
    #        return shift

def shiftedgain(hyperfet,VDD):
    VD,VG,app_Ioff,R_insp,V_met,VTp,k,R_metp=\
        shorthands(hyperfet,VDD,None,"app_Ioff","R_insp","V_met","VTp","k","R_metp",gridinput=False)
    return (1+(app_Ioff*R_insp-V_met)/(VDD-VTp))/(1+k*R_metp)

#def shiftedsr(self):
    #return (self.VDD-self.VTp)*self.mosfet.k*self.R_metp+self.vo2.V_met-self._approx_Ioff*self.R_insp

def optsize(fet,VDD,Ml=1,Mr=0,**vo2params):
    VTp=fet.VT0-fet.delta*VDD
    VTm=VTp-fet.alpha*fet.Vth
    app_Ioff=fet.n*fet.k*fet.Vth*np.exp(-VTm/(fet.n*fet.Vth))
    Vth=fet.Vth
    v=vo2params

    l1=(VDD-Vth/2)/(v['J_IMT']*v['rho_i'])
    l2=((VDD-Mr*Vth-VTm+fet.n*Vth*np.log(fet.n*fet.Vth*fet.k*v['J_MIT']/(v['J_IMT']*app_Ioff)))/(v['J_IMT']*v['rho_i']*(1+fet.delta)-v['J_MIT']*v['rho_m']-v['v_met']))
    l=min(l1,l2)

    print("l1 ",l1*1e9)
    print("l2 ",l2*1e9)
    print("l ",l*1e9)

    wti=(fet.n*Vth/(Ml*app_Ioff*l*v['rho_i']*(1+fet.delta)))*lambertw((v['J_MIT']*v['rho_i']*(1+fet.delta)*l/(fet.n*Vth))*np.exp((v['J_MIT']*v['rho_m']+v['v_met'])*l/(fet.n*Vth)))

    l2=(VDD-Mr*Vth-VTm+fet.n*fet.Vth*np.log(fet.n*fet.k*fet.Vth*wti/v['J_IMT']))/(v['J_IMT']*v['rho_i']*(1+fet.delta)-app_Ioff*v['rho_i']*(1+fet.delta)*wti)
    print("l2 ",l2*1e9)
    l=min(l1,l2)

    print("l ",l*1e9)
    print("w ",1/wti*1e18)
