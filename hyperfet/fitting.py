import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.optimize import curve_fit

def curve_fit_scaled(f,xdata,ydata,p0,bounds,param_scalings):
    def f2(xdata,*p):
        p=[pi*si for pi,si in zip(p,param_scalings)]
        return f(xdata,p)
    p0=[p0i/si for p0i,si in zip(p0,param_scalings)]
    bounds=[[b/si for b,si in zip(b,param_scalings)] for b in bounds]
    return curve_fit(f2,xdata=xdata,ydata=ydata,p0=p0,bounds=bounds)

def show_transistor(trans, VDD, data=None, **opts):
    plt.figure(figsize=opts['figsize'] if 'figsize' in opts else None)

    VD=np.array(VDD)
    VG=np.linspace(0,VDD,500)
    I=trans.ID(VD=VD,VG=VG)
    plt.plot(VG,I/trans.W,label=r"$V_D={:.2g}$".format(VDD))
    if data: plt.plot(data['idvg']['v'],data['idvg']['i/W'],'.')
    plt.yscale('log')
    plt.xlabel(r"$V_G\;\mathrm{[V]}$",fontsize=18)
    plt.legend(loc='lower left',fontsize=16)
    plt.xlim(0,VDD)
    plt.ylabel("$I_D\;\mathrm{[\mu A/\mu m]}$",fontsize=18)

    if 'linidvgpos' in opts:
        plt.axes(opts['linidvgpos'])
        I=trans.ID(VD=VD,VG=VG)
        plt.plot(VG,I/trans.W,label=r"$V_D={:.2g}$".format(VDD))
        if data: plt.plot(data['idvg']['v'],data['idvg']['i/W'],'.')
        if 'linidvgxticks' in opts: plt.xticks(opts['linidvgxticks'])#
        if 'linidvgxlim' in opts: plt.xlim(opts['linidvgxlim'])
        if 'linidvgyticks' in opts: plt.yticks(opts['linidvgyticks'])
        plt.title("$\mathrm{Lin}\;I_D-V_G$")

    if 'linidvdpos' in opts:
        plt.axes(opts['linidvdpos'])
        VD=np.linspace(0,VDD,500)
        VG=np.linspace(0,VDD,11)
        I=trans.ID(VD=VD,VG=VG)
        plt.plot(VD,I.T/trans.W)
        if data:
            for fn in data.keys():
                if fn.startswith("idvd"):
                    plt.plot(data[fn]['v'],data[fn]['i/W'],'.')
        plt.ylim(0)
        plt.xlim(0,VDD)
        if 'linidvdxticks' in opts: plt.xticks(opts['linidvdxticks'])
        if 'linidvdyticks' in opts: plt.yticks(opts['linidvdyticks'])
        plt.title("$\mathrm{Lin}\;I_D-V_D$")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        plt.tight_layout()
