import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.optimize import curve_fit
from IPython.display import display_html

def curve_fit_scaled(f,xdata,ydata,p0,bounds,param_scalings):
    def f2(xdata,*p):
        p=[pi*si for pi,si in zip(p,param_scalings)]
        return f(xdata,*p)
    p0=[p0i/si for p0i,si in zip(p0,param_scalings)]
    bounds=[[b/si for b,si in zip(b,param_scalings)] for b in bounds]
    p,pcov=curve_fit(f2,xdata=xdata,ydata=ydata,p0=p0,bounds=bounds)
    p=[pi*si for pi,si in zip(p,param_scalings)]
    return p,None

def show_transistor(trans, VDD, data=None, **opts):
    plt.figure(figsize=opts['figsize'] if 'figsize' in opts else None)

    if 'subplots' in opts:
        plt.subplot(131)
        plt.title("$\mathrm{Log}\;I_D-V_G$")
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

    a2=None
    if 'subplots' in opts:
        a2=plt.subplot(132)
    elif 'linidvgpos' in opts:
        a2=plt.axes(opts['linidvgpos'])
        a2.yaxis.tick_right()
    if a2:
        I=trans.ID(VD=VD,VG=VG)
        plt.plot(VG,I/trans.W,label=r"$V_D={:.2g}$".format(VDD))
        if data: plt.plot(data['idvg']['v'],data['idvg']['i/W'],'.')
        if 'linidvgxticks' in opts: plt.xticks(opts['linidvgxticks'])#
        if 'linidvgxlim' in opts: plt.xlim(opts['linidvgxlim'])
        if 'linidvgyticks' in opts: plt.yticks(opts['linidvgyticks'])
        plt.title("$\mathrm{Lin}\;I_D-V_G$")

    a3=None
    if 'subplots' in opts:
        a3=plt.subplot(133)
    elif 'linidvdpos' in opts:
        a3=plt.axes(opts['linidvdpos'])
    if a3:
        VD=np.linspace(0,VDD,500)
        if data:
            VG=np.array([float(fname.split("_")[1]) for fname in data.keys() if fname.startswith('idvd')])
        else:
            VG=np.linspace(0,VDD,11)
        VDgrid,VGgrid=np.meshgrid(VD,VG)
        I=trans.ID(VD=VDgrid,VG=VGgrid)
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

def display_table(od):
    string="<table width='100%'><tr>"
    string+="".join(["<th>{}</th>".format(k) for k in od.keys()])
    string+="</tr><tr>"
    string+="".join(["<td>{:.4g}</td>".format(k) for k in od.values()])
    string+="</tr></table>"
    display_html(string,raw=True)


