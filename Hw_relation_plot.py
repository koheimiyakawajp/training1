#!/usr/bin/env python

import numpy as np
import sys
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os
from multiprocessing import Pool
from astropy import units as u

from scipy import interpolate
from scipy.optimize import minimize

#-- load spectra files here
def load_spectra(file_path):
    return np.loadtxt(file_path, comments='#', dtype='f8').T

def read_map_multi(spectra_all):
    p   = Pool(os.cpu_count())
    data_spec   = p.map(load_spectra, spectra_all) # map で並列読み込み
    p.close()
    return data_spec
#--

def cal_photon(input_arrays):
    # 1.透過関数をsplineで関数化.
    # 2.透過関数とFluxを波長を乗じて積分し、光子数に換算.
    # 3.最終的に比の計算を行うため、各係数は無視.
    spectra_array,filter_func,Av    = input_arrays

    ff  = interpolate.interp1d(x=filter_func[1],\
                               y=filter_func[2],kind="cubic")

    spec_narrow     = spectra_array[:,((filter_func[1,0]<spectra_array[0])&\
                                       (spectra_array[0]<filter_func[1,-1]))]

    transmit_f  = ff(spec_narrow[0])*spec_narrow[1] #transmitted flux
    dx          = np.diff(spec_narrow[0])

    extinction  =  10** (-1*A_lambda(Av, spec_narrow[0,:-1])/2.5)
    photon      = np.sum(transmit_f[:-1] * dx * spec_narrow[0,:-1] * extinction)

    return photon


def calphoton_map_multi(data_spec, filter_func, Av):

    data_array  = [(x,filter_func,Av) for x in data_spec]
    xlen    = len(filter_func)
    ylen    = len(data_spec)

    p   = Pool(os.cpu_count())

    data_photon = p.map(cal_photon, data_array)
    p.close()
    data_photon = np.array(data_photon, dtype='f8')

    return data_photon


def load_filter():
    fl_J    = "./filter/J_filter.dat"
    fl_H    = "./filter/H_filter.dat"

    fltJ    = np.loadtxt(fl_J, comments="#", dtype='f8').T
    fltH    = np.loadtxt(fl_H, comments="#", dtype='f8').T
    fltJ[1] = fltJ[1]*1e4 #mic -> angstrom
    fltH[1] = fltH[1]*1e4

    return fltJ,fltH

def Hw_func(lower, upper):
    lw  = lower #angstrom
    up  = upper #
    nd  = 150 #number of data

    x   = np.linspace(lw-1000, up+1000, nd)
    y   = np.where((lw <= x)&(x <= up),1.,0.)
    index   = np.linspace(1,nd,nd)
    Hw  = np.array((index,x,y))

    return Hw

def A_lambda(Av, x):
    Aj_Av   = 0.282
    Ak_Av   = 0.112

    Aj  = Aj_Av*Av
    Ak  = Ak_Av*Av

    return ((x-12000)*Ak + (20000-x)*Aj)/(20000 - 12000)

def quad_func(x, args):
    a   = args[0]
    b   = args[1]
    return a*x**2 + b*x

def least_sq(coeff, *args):
    x,y     = args
    chi2    = np.sum(np.square(quad_func(x,coeff)-y))**0.5
    return chi2


def main(regex_spec, Hw_l, Hw_u):

    #>> >> load spectra file names
    spectra_all = sorted(glob.glob(regex_spec+"/*"))
    data_spec   = read_map_multi(spectra_all)
    #<< <<

    # >> >> >> set range of Hw band
    fil_Hw  = Hw_func(Hw_l, Hw_u) #
    # << << <<

    # >> >> >> zero magnitude spectra
    spec_a0v    = np.loadtxt("./spectra/uka0v.dat",comments='#',dtype='f8').T
    fil_J,fil_H = load_filter()

    p_Jo    = cal_photon([spec_a0v, fil_J, 0])
    p_Ho    = cal_photon([spec_a0v, fil_H, 0])
    p_Hwo   = cal_photon([spec_a0v, fil_Hw, 0])
    # << << <<

    plt.figure(figsize=(5,5.5))
    plt.rcParams['font.family'] = 'Arial'

    ax0     = plt.subplot2grid((5,1),(0,0),rowspan=4)
    ax1     = plt.subplot2grid((5,1),(4,0),rowspan=1, sharex=ax0)
    ax0.grid(color='gray', ls=':', lw=0.5)
    ax1.grid(color='gray', ls=':', lw=0.5)

    Av_ar   = np.linspace(0,60,5)

    ar_Hw_H     = []
    ar_J_H      = []
    for Av in Av_ar: # -- roop for Av
        p_J     = calphoton_map_multi(data_spec, fil_J, Av)
        p_H     = calphoton_map_multi(data_spec, fil_H, Av)
        p_Hw    = calphoton_map_multi(data_spec, fil_Hw, Av)

        rel_J   = -2.5*(np.log10(p_J) - np.log10(p_Jo))
        rel_H   = -2.5*(np.log10(p_H) - np.log10(p_Ho))
        rel_Hw  = -2.5*(np.log10(p_Hw) - np.log10(p_Hwo))

        J_H     = rel_J  - rel_H
        Hw_H    = rel_Hw - rel_H

        ar_Hw_H.append(Hw_H)
        ar_J_H.append(J_H)
        A_str   = str(Av)
        #plt.scatter(J_H, Hw_H, s=5, label='Av = '+A_str)
        ax0.scatter(J_H, Hw_H, s=5, label='Av = '+A_str)

    ar_Hw_H     = np.ravel(np.array(ar_Hw_H))
    ar_J_H      = np.ravel(np.array(ar_J_H))

    colors      = (ar_J_H,ar_Hw_H)

    # >> >> >> optimize coefficients
    x0  = [1., 0.8]
    res = minimize(least_sq, x0, args=colors, method='Nelder-Mead', tol=1e-11)
    # << << <<

    zansa   = colors[1] - quad_func(colors[0], res.x)
    sigma   = np.std(zansa)
    print(sigma)

    # >> >> >> plot

    a_str   = str('{:.5f}'.format(res.x[0]))
    b_str   = str('{:.5f}'.format(res.x[1]))
    pl_txt1 = "$y$ = "+a_str+" $x^2$ + "+b_str+" $x$"
    pl_txt2 = str('{:.2f}'.format(Hw_l*1e-4)) + "\u03bcm < $Hw$ < " \
        + str('{:.2f}'.format(Hw_u*1e-4) +"\u03bcm")

    x_pl    = np.linspace(min(colors[0]),max(colors[0]), 1000)
    y_pl    = quad_func(x_pl, res.x)
    ax0.plot(x_pl, y_pl, ls='--', c='black', lw=1)
    ax0.text(x_pl[int((len(x_pl)*0.3))], y_pl[int((len(y_pl)*0.1))],\
            pl_txt1, fontsize=12)
    ax0.text(x_pl[int((len(x_pl)*0.4))], y_pl[int((len(y_pl)*0.05))],\
            pl_txt2, fontsize=12)

    #ax0.set_xlabel("$J - H$", fontsize=15)
    ax0.set_ylabel("$Hw - H$", fontsize=15)
    ax0.legend(fontsize=10, loc='upper left')
    ax0.plot(x_pl, y_pl, ls='--', c='black', lw=1)
    #plt.text(x_pl[int((len(x_pl)*0.3))], y_pl[int((len(y_pl)*0.1))],\
    #         pl_txt1, fontsize=12)
    #plt.text(x_pl[int((len(x_pl)*0.4))], y_pl[int((len(y_pl)*0.05))],\
    #         pl_txt2, fontsize=12)

    #plt.xlabel("$J - H$", fontsize=15)
    #plt.ylabel("$Hw - H$", fontsize=15)
    #plt.legend(fontsize=10, loc='upper left')
    #plt.show()

    ax1.scatter(colors[0], zansa, s=5, c='gray')
    ax1.set_xlabel("$J - H$", fontsize=15)
    plt.subplots_adjust(hspace=.0)
    #plt.show()

    plt.savefig("Hwcolor_"+str(int(Hw_l)) + "_" +str(int(Hw_u))+".png", dpi=200)
    # << << <<


if __name__=='__main__':
    #print(sys.argv)
    if len(sys.argv) == 4:
        regex_spec  = sys.argv[1]
        Hw_low      = float(sys.argv[2])
        Hw_up       = float(sys.argv[3])
        main(regex_spec, Hw_low, Hw_up)
    else:
        print("usage) [regex of spectra dir] [Hw lower] [Hw upper]")
        print("ex) "+sys.argv[0]+" ./spectra/ 9000 15000")
