cimport cython
cimport numpy as cnp
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from libc.math cimport pow, exp, log, abs, log10
from functools import wraps
from scipy.constants import k as kb, elementary_charge as e
cnp.import_array()

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Optimized generic Cython loops over numpy arrays
##### ##### ##### ##### ##### ##### ##### ##### ##### #####

ctypedef void NpyIter
ctypedef struct NewNpyArrayIterObject:
    cnp.PyObject base
    NpyIter *iter
cdef void* nullptr

@cython.boundscheck(False)
cdef cnp.ndarray dimsimple(cnp.ndarray inarr,
                           double (*func)(double,void*),void* args=nullptr,
                           cnp.ndarray outarr=None):
    r"""dimsimple: Does Sphinx find this?

    :param inarr:
    :param func:
    :param args:
    :param outarr:
    :return:
    """
    cdef int j
    cdef cnp.ndarray[double] subinarr, suboutarr
    cdef void* cargs

    assert inarr.dtype==np.float
    if outarr is None:
        outarr=np.empty_like(inarr)
    cargs=<void*>args

    it = np.nditer([inarr,outarr], flags=['external_loop','buffered'],
                   op_flags=[['readonly'], ['writeonly']])
    for subinarr, suboutarr in it:
        for j in range(subinarr.shape[0]):
            suboutarr[j]=func(subinarr[j],cargs)
    return outarr

@cython.boundscheck(False)
cdef cnp.ndarray gridinput(cnp.ndarray inarr1, cnp.ndarray inarr2,
                           double (*func)(double,double,void*),void* args=nullptr,
                           cnp.ndarray outarr=None):

    cdef int j
    cdef cnp.ndarray[double] subinarr1,subinarr2,suboutarr
    cdef void* cargs

    assert inarr1.dtype==np.float
    assert inarr2.dtype==np.float
    inarr1,inarr2=np.meshgrid(inarr1,inarr2)
    if outarr is None:
        outarr = np.empty_like(inarr1)
    cargs=<void*>args

    it = np.nditer([inarr1,inarr2,outarr], flags=['external_loop','buffered'],
                   op_flags=[['readwrite'], ['readwrite'], ['readwrite']])
    for subinarr1,subinarr2,suboutarr in it:
        for j in range(subinarr1.shape[0]):
            suboutarr[j]=func(subinarr1[j],subinarr2[j],cargs)
    return outarr

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# Basic devices
##### ##### ##### ##### ##### ##### ##### ##### ##### #####


cdef class SCMOSFET:
    r""" Short-channel MOSFET.

    The short-channel MOSFET model described in `(Khakifirooz 2009) <https://doi.org/10.1109/TED.2009.2024022>`_, with
    the addition of a leakage term.  All named parameters are the same as in the paper, and the leakage term is
    :math:`I_\mathrm{leak} = G_\mathrm{leak}  V_D`.

    :param T: temperature [Kelvin]
    :param W: device width [meters]
    :param Cinv_vxo: transconductance per width in [Siemens/meter]
    :param VT0: threshold voltage [Volt] pre-DIBL
    :param alpha: threshold shift between strong and weak inversion is ``alpha`` times thermal voltage
    :param SS: subthreshold swing in [Volt/decade]
    :param delta: DIBL parameter
    :param VDsats: saturation voltage [Volt]
    :param beta: saturation parameter
    :param Gleak: Leakage conductance [Siemens]
    """

    cdef public double Vth, W, Cinv_vxo, VT0, alpha, SS, delta, VDsats, beta, Gleak, n, k

    @cython.cdivision(True)
    def __init__(SCMOSFET self, double T=300,
                 double W=100e-9, double Cinv_vxo=2500,
                 double VT0=.9, double alpha=3.5, double SS=90e-3, double delta=0,
                 double VDsats=.3, double beta=2, double Gleak=1e-14):

        self.Vth= kb * T / e
        self.W=W
        self.Cinv_vxo=Cinv_vxo
        self.VT0=VT0
        self.alpha=alpha
        self.SS=SS
        self.delta=delta
        self.VDsats=VDsats
        self.beta=beta
        self.Gleak=Gleak
        self.n=SS/(self.Vth * log(10))
        self.k=W*Cinv_vxo

    cdef double VT(SCMOSFET self, double VD, double VG):
        r""" Threshold voltage accounting for DIBL and strong/weak inversion shift

        Equation 4.4 of Ujwal's masters thesis.

        :param VD: (double) drain voltage
        :param VG: (double) gate voltage
        :return: (double) the threshold voltage
        """
        return self.VT0-self.delta*VD-self.alpha*self.Vth * self.Ff(VD, VG)

    # Eq 4.5
    @cython.cdivision(True)
    cdef double Ff(SCMOSFET self, double VD, double VG):
        r""" Fermi inversion regime smoothing function

        Equation 4.5 of Ujwal's masters thesis.  If ``self.alpha`` is 0, ``Ff`` is just 0, no inversion VT shifts.

        :param VD: (double) drain voltage
        :param VG: (double) gate voltage
        :return: (double) the Fermi smoothing factor (0-1)
        """
        cdef double exparg

        # if alpha==0, no inversion shifting
        if self.alpha==0:
            return 0

        # alpha!=0
        else:
            # the argument that will go into the exponential
            exparg= (VG - (self.VT0 - self.delta * VD - self.alpha * self.Vth / 2)) / (self.alpha * self.Vth)

            # If the argument is large in magnitude, avoid overflow by just taking the limit
            if exparg>15: return 0
            if exparg<-15: return 1

            # Otherwise, actually evaluate the smoothing
            return 1/(1+exp(exparg))

    @cython.cdivision(True)
    cdef double Fsat(SCMOSFET self, double VD, double VG):
        r""" Saturation function (interpolate linear to constant behavior)

        Equation 4.6 of Ujwal's masters thesis.

        :param VD: (double) drain voltage
        :param VG: (double) gate voltage
        :return: (double) unitless saturation factor
        """
        VDsat=self.VDsats*(1-self.Ff(VD,VG))+ self.Vth * self.Ff(VD, VG)
        return (VD/VDsat)/(1+(VD/VDsat)**self.beta)**(1/self.beta)

    # Drain current
    @cython.cdivision(True)
    @staticmethod
    cdef double _ID(double VD, double VG, void* args):
        r""" Drain current for scalar input

        This is Eq 4.2,4.7,4.10 from Ujwal's thesis, plus a leakage term :math:``Gleak*VD``.

        This scalar function is intended to be looped by a construct from :py:mod:`pynitride.util.cython_loops`.
        So to match the signature required where "additional non-looping arguments" are passed at the end,
        it is formally declared static, with ``self`` passed via the ``args`` parameter.

        :param VD: (double) drain voltage
        :param VG: (double) gate voltage
        :param args: (double) args should be the "self" argument, a GaNHEMT_iMVGs, cast as a void*
        :return: (double) the drain current
        """
        self=<SCMOSFET> args

        # what will go inside the exponential
        exparg=(VG-(self.VT(VD,VG)))/(self.n*self.Vth)

        # if the argument is not too large, evaluate normally
        if exparg<15:
            return self.W*self.Cinv_vxo*self.Fsat(VD,VG)*self.n*self.Vth \
                *log(1+exp(exparg)) \
                +VD*self.Gleak
        # but if the argument is large, avoid overflow by taking the limit
        else:
            return self.W*self.Cinv_vxo*self.Fsat(VD,VG)*self.n*self.Vth \
                *(exparg) \
                +VD*self.Gleak

    def ID(SCMOSFET self, VD, VG):
        r""" Compute the drain current.

        Forms a meshgrid from the combinations of ``VD`` (varies along rows) and ``VG`` varies along columns and
        computes the drain current in a matching shape.

        :param VD: (numpy double array) drain voltage
        :param VG: (numpy double array) gate voltage
        :return: 2-D numpy double array drain current
        """
        return gridinput(np.asarray(VD,dtype='double'),np.asarray(VG,dtype='double'),self._ID,args=<void*>self)

    def shifted(SCMOSFET self, VT0_shift):
        return SCMOSFET(T=e*self.Vth / kb,
                        W=self.W, Cinv_vxo=self.Cinv_vxo,
                        VT0=self.VT0+VT0_shift, alpha=self.alpha, SS=self.SS, delta=self.delta,
                        VDsats=self.VDsats, beta=self.beta, eta=self.eta, Gleak=self.Gleak)


r""" Represents a piecewise linear current-controlled i-v (potentially non-unique in V)

"Lower i-v" passes through origin.  Higher i-v may have an offset voltage.

I_trans: current transition point between the linear i-v regions
R_low: resistance in the lower i-v region
R_high: resistance in the higher i-v region
V_off_high: voltage offset (ie V-intercept) for the higher i-v region
"""
cdef struct PiecewiseLinearVI:
    double I_trans
    double R_low
    double R_high
    double V_off_high

r""" For sweep directions"""
cpdef enum Direction:
    FORWARD = 0
    BACKWARD = 1

cdef class PCR:
    r""" Models a phase-change resistor by two piecewise linear i-v's, one for forward bias, one for reverse.

    R_met should not be zero.  (This may be relaxed in the future.)

    :param I_IMT: insulator-to-metal (forward) transition current [A]
    :param V_IMT: insulator-to-metal (forward) transition voltage [V]
    :param I_MIT: metal-to-insulator (reverse) transition current [A]
    :param V_MIT: metal-to-insulator (reverse) transition voltage [V]
    :param R_met: resistance in the metallic state
    """

    cdef public double I_IMT, I_MIT, V_IMT, V_MIT, R_ins, V_met, R_met, I_smooth
    cdef PiecewiseLinearVI[2] VIs

    @cython.cdivision(True)
    def __init__(PCR self, double I_IMT=.01e-3, double V_IMT=1, double I_MIT=.0075e-3, double V_MIT=.5, double R_met=0):

        self.I_IMT=I_IMT
        self.I_MIT=I_MIT
        self.V_IMT=V_IMT
        self.V_MIT=V_MIT
        self.R_ins=V_IMT/I_IMT
        self.V_met=V_MIT-self.I_MIT*R_met
        self.R_met=R_met

        assert R_met!=0

        self.VIs[<int> Direction.FORWARD]= \
                PiecewiseLinearVI(I_trans=self.I_IMT,R_low=self.R_ins,R_high=self.R_met,V_off_high=self.V_met)
        self.VIs[<int> Direction.BACKWARD]= \
                PiecewiseLinearVI(I_trans=self.I_MIT,R_low=self.R_ins,R_high=self.R_met,V_off_high=self.V_met)

    @staticmethod
    cdef double _V(double i,void* args):
        r""" Voltage as a function of current for scalar inputs

        This scalar function is intended to be looped by a construct from :py:mod:`pynitride.util.cython_loops`.
        So to match the signature required where "additional non-looping arguments" are passed at the end,
        it is formally declared static, with a ``PiecewiseLinearVI*`` passed via the ``args`` parameter.

        :param i: (scalar double) the current
        :param args: pointer to the appropriate ``PiecewiseLinearVI``, cast as void*
        :return: (scalar double) the voltage
        """
        cdef PiecewiseLinearVI vi
        vi=(<PiecewiseLinearVI*>args)[0]
        if i<vi.I_trans:
            return i*vi.R_low
        else:
            return i*vi.R_high+vi.V_off_high

    @cython.boundscheck(False)
    def V(PCR self, I, Direction direc):
        r""" Compute the drain current.

        Forms a meshgrid from the combinations of ``VD`` (varies along rows) and ``VG`` varies along columns and
        computes the drain current in a matching shape.

        :param I: (numpy double array) currents
        :param direc: ``Direction.FORWARD`` or ``Direction.BACKWARD`` indicating the sweep
        :return: numpy double array of of voltages, same shape as ``I``
        """
        return dimsimple(np.asarray(I,dtype='double'), PCR._V, <void*>&(self.VIs[<int>direc]))




# rho_m=si("1e-4 ohm cm")
# rho_i=si("2 ohm cm")
# J_MIT=si("1e6 A/cm^2")
# J_IMT=si("1e5 A/cm^2")
# Makes a VO2 resistor of a given length and width
def VO2(W,L,T,V_met=0,rho_m=1e-6,rho_i=2e-2,J_MIT=1e10,J_IMT=1e9):
    I_IMT=J_IMT*T*W
    I_MIT=J_MIT*T*W

    R_ins=rho_i*L/(W*T)
    R_met=rho_m*L/(W*T)

    V_IMT=I_IMT*R_ins
    V_MIT=I_MIT*R_met+V_met

    return PCR(I_IMT=I_IMT, V_IMT=V_IMT, I_MIT=I_MIT, V_MIT=V_MIT, R_met=R_met)

cdef class HyperFET:
    r""" Models a :py:class:`VO2Res` in series with the source of a :py:class:`SCMOSFET`."""

    cdef public SCMOSFET mosfet
    cdef public PCR pcr

    def __init__(HyperFET self, SCMOSFET mosfet, PCR pcr):
        self.mosfet=mosfet
        self.pcr=pcr

    cdef double _I_low(HyperFET self, double VD, double VG, Direction direc) except -2:
        r""" Returns the solution current along the lower (insulating) branch if one exists at this ``VD,VG``.

        :param VD: (double) drain voltage :math:`V_D > 0`
        :param VG: (double) gate voltage
        :param direc: ``Direction.FORWARD`` or ``Direction.BACKWARD`` indicating sweeep
        :return: a valid positive current on the insulating branch, or -1 if no solution
        """

        # The appropriate piecewise linear i-v
        rVI=self.pcr.VIs[<int>direc]

        # the upper limiting allowed current as given by the transition current
        imax_r=rVI.I_trans

        # the upper limiting allowed current as given by requiring a positive voltage drop on the transistor
        imax_vd=VD/rVI.R_low

        # Tak the tightest of the limits
        imax=min(imax_r,imax_vd)

        # The error function for this problem is:
        # Given a trial current, compute the VO2 voltage; then, given that voltage, compute the transistor current
        # and return the log ratio of the computed current versus the trial current
        # However, we'll use the log current instead to make life easier for the root-finding procedure
        # Ideally, the error will be positive for currents too small, negative for currents too high
        def ierr( double log_i_try, HyperFET self, double VD, double VG):
            cdef double i_try, v_r, i_calc
            i_try=exp(log_i_try)
            v_r=i_try*rVI.R_low
            i_calc=SCMOSFET._ID(VD - v_r, VG - v_r, <void*>(self.mosfet))
            return log(i_calc)-log_i_try

        # Reasonable lower current limit
        logimin=-50

        # Safe upper current limit to make sure we never pass a negative VD
        logimax=log(imax*.9999)

        # Check that the root finding will not be limited by this lower current limit
        if(ierr(logimin,self,VD,VG)<0):
            # Note that this exception will not get caught, but will generally get printed
            raise Exception("Min too high")
            return -2

        # If the root finding is limiting by the transition current, there is no solution in this branch
        if(ierr(logimax,self,VD,VG)>0):
            return -1

        # Find the root, and, if it's been identified to sufficient precision, return it
        x=brentq(ierr,logimin,logimax,args=(self,VD,VG),xtol=1e-5,rtol=1e-5)
        if abs(ierr(x,self,VD,VG))< .001:
            return exp(x)
        else: return -1

    cdef double _I_high(HyperFET self, double VD, double VG, Direction direc) except -2:
        r""" Returns the solution current along the upper (metallic) branch if one exists at this ``VD,VG``.

        :param VD: (double) drain voltage :math:`V_D > 0`
        :param VG: (double) gate voltage
        :param direc: ``Direction.FORWARD`` or ``Direction.BACKWARD`` indicating sweeep
        :return: a valid positive current on the metallic branch, or -1 if no solution
        """

        # The appropriate piecewise linear i-v
        rVI=self.pcr.VIs[<int>direc]

        # the lower limiting allowed current as given by the transition current
        imin=rVI.I_trans

        # the upper limiting allowed current as given by requiring a positive voltage drop on the transistor
        imax=(VD-rVI.V_off_high)/rVI.R_high

        # The error function for this problem is:
        # Given a trial current, compute the VO2 voltage; then, given that voltage, compute the transistor current
        # and return the log ratio of the computed current versus the trial current
        # However, we'll use the log current instead to make life easier for the root-finding procedure
        # Ideally, the error will be positive for currents too small, negative for currents too high
        def ierr( double log_i_try, HyperFET self, double VD, double VG):
            cdef double i_try, v_r
            i_try=exp(log_i_try)
            v_r=i_try*rVI.R_high+rVI.V_off_high
            i_calc=SCMOSFET._ID(VD - v_r, VG - v_r, <void*>(self.mosfet))
            return log(i_calc)-log_i_try

        # Lower limit at transition
        logimin=log(imin)

        # Safe upper current limit to make sure we never pass a negative VD
        logimax=log(.9999*imax)

        # If the root finding is limiting by the transition current, there is no solution in this branch
        if(ierr(logimin,self,VD,VG)<0):
            return -1

        # Check that the root finding will not be limited by this lower current limit
        if(ierr(logimax,self,VD,VG)>0):
            # Note that this exception will not get caught, but will generally get printed
            raise Exception("Max too low")
            return -2

        # Find the root, and, if it's been identified to sufficient precision, return it
        x,r=brentq(ierr,logimin,logimax,args=(self,VD,VG),xtol=1e-5,rtol=1e-5,full_output=True)
        if abs(ierr(x,self,VD,VG))< .001:
            return exp(x)
        else: return -1

    def I(HyperFET self, VD, VG, Direction direc):
        r""" Drain current along a particular sweep direction.

        For simplicity, direction is specified separately from ``VD`` and ``VG``, but this function will run much more
        efficiently if ``VD`` and ``VG`` change monotonically in the given direction, because then the function will
        know it doesn't need to keep checking solutions for branches which have already disappeared

        Forms a meshgrid from the combinations of ``VD`` (varies along rows) and ``VG`` varies along columns and
        computes the drain current in a matching shape.

        :param VD: numpy double array of drain voltage
        :param VG: numpy double array of gate voltage
        :param direc: ``Direction.FORWARD`` or ``Direction.BACKWARD`` indicating the sweep
        :return: 2D numpy double array of the current values along the ``VD-VG`` grid
        """

        cdef int j
        cdef double[:] iarr, vdarr, vgarr
        cdef double vd_prev, vg_prev, i_prev, i_trans, vd, vg

        # transition between branches
        i_trans=self.pcr.VIs[<int>direc].I_trans

        # i_prev will track the previously computed current.
        # depending on the sweep direction, if i_prev is strictly larger/smaller than i_trans, certain branches may
        # be skipped in the next solves.  Choosing i_prev exactly equal to i_trans avoids this accidentally happening
        # on the first run-through
        i_prev=i_trans

        # Form input/output arrays
        VDgrid,VGgrid=np.meshgrid(VD,VG)
        I = np.empty_like(VDgrid)
        it = np.nditer([VDgrid,VGgrid,I], flags=['external_loop','buffered'],
                       op_flags=[['readwrite'], ['readwrite'], ['readwrite']])

        # Python iterate over external chunks
        for vdarr,vgarr,iarr in it:

            # Internal fast Cython loop
            for j in range(vdarr.shape[0]):
                vd=vdarr[j]
                vg=vgarr[j]

                # Nominally going forward
                if direc==Direction.FORWARD:

                    # We can skip the low branch if we're really going forward and we've passed i_trans before
                    skip_low=(vd>vd_prev and vg>vg_prev and i_prev>i_trans)
                    vg_prev=vg
                    vd_prev=vd

                    # Check low branch if necessary
                    if not skip_low:
                        i=self._I_low(vd,vg,Direction.FORWARD)
                        if i>0:
                            i_prev=iarr[j]=i
                            continue

                    # If low branch came up empty, check high branch
                    i=self._I_high(vd,vg,Direction.FORWARD)
                    if i>0:
                        i_prev=iarr[j]=i

                    # If both failed, NaN
                    else:
                        iarr[j]=np.NaN
                        #raise Exception("Solution not found")

                # Nominally going backward
                else:

                    # We can skip the high branch if we're really going backward and we've passed i_trans before
                    skip_high=(vd<vd_prev and vg<vg_prev and i_prev<i_trans)

                    # Check high branch if necessary
                    if not skip_high:
                        i=self._I_high(vd,vg,Direction.BACKWARD)
                        if i>0:
                            i_prev=iarr[j]=i
                            continue

                    # If high branch came up empty, check low branch
                    i=self._I_low(vd,vg,Direction.BACKWARD)
                    if i>0:
                        i_prev=iarr[j]=i

                    # If both failed, NaN
                    else:
                        iarr[j]=np.NaN
                        #raise Exception("Solution not found")
        return I

    def I_double(HyperFET self, VD, VG):
        r""" Return a current sweep in both directions

        ``VD`` and ``VG`` should be specified for just the **forward** direction.  This is essentially two calls to
        :py:func:`HyperFET.I` but the reversing of inputs and outputs will be handled automatically.

        :param VD: numpy double array of drain voltage
        :param VG: numpy double array of gate voltage
        :return: 2D numpy double array of the current values along the ``VD-VG`` grid
        """
        def rev(arr):
            if len(arr.shape)>0:
                arr=np.flipud(arr)
            if len(arr.shape)>1:
                arr=np.fliplr(arr)
            return arr

        If=self.I(VD,VG,Direction.FORWARD)
        Ib=rev(self.I(rev(VD),rev(VG),Direction.BACKWARD))
        return If,Ib

