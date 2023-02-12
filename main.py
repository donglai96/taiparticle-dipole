
import sys
import taichi
import numpy as np
import constants as cst
from taichiphysics import *
from res_energy import *

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("The running id:", sys.argv[1:])
    else:
        print("No input id provided.")
    
    # read the input file
    id = sys.argv[1:][0]
    with open(id + '/input.txt', 'r') as f:
        lines = f.readlines()
    paras = {}
    for line in lines:
        if ':' in line:
            key, value = line.strip().split(': ')
            try:
                paras[key] = int(value)
            except ValueError:
                try:
                    paras[key] = float(value)
                except ValueError:
                    paras[key] = value
    # read the paras
    L_shell = paras['L_shell']
    # calculate the wce, wpe at equator
    _,_,bz0 = dipole_field(cst.B0,L_shell,cst.Planet_Radius,0,0,0)

    # get wce0
    wce0 = gyrofrequency(cst.Charge,cst.Me,bz0)
    print('gyrofrequency(rad/s) at L ',L_shell,' = ', wce0 )
    n0 = paras['n0']
    n_dis = paras['n_dis']
    


    # number of charged particles
    Np = paras['Np']
    t_total_num = paras['t_total_num']
    record_num = paras['record_num']
    dt_num =paras['dt_num']
    w_res_num =paras['w_res_num']
    w_lc_num = paras['w_lc_num']#!
    w_uc_num = paras['w_uc_num']#!
    w_width_num = paras['w_width_num']#!
    if w_res_num < 1:
        w_res = w_res_num * wce0
    else:
        w_res = w_res_num * 2 * np.pi
    nw = paras['nw']
    # wave frequency width
    alpha_eq = np.deg2rad(paras['alpha_eq'])
    res_lat = np.deg2rad(paras['res_lat'])
    lat_init = np.deg2rad(paras['lat_init'])
    Bw = paras['Bw']
    # init mass and charge
    mass = cst.Me
    charge = cst.Charge * -1
    direction = -1

    #############################
    print('***************')
    print('Calculate the resonating energy')
    print(alpha_eq,'alpha_eq')
    alpha = get_pitchanlge_from_eq(alpha_eq,res_lat,direction)
    _,_,bz_lat = dipole_field(cst.B0,L_shell,cst.Planet_Radius,0,0,res_lat)
    wce_alpha = gyrofrequency(cst.Charge,cst.Me,bz_lat)

    n_lat = electron_density('cos',n0,res_lat)
    p0, k0 = get_resonance_p_whistler(w_res,wce_alpha,n_lat,alpha,nres = -1)
    print('momentum of resonating particles are:',p0)
    print('resonant wave number at reslat is :', k0)
    print('resonating frequency is (Hz) ', w_res/(2 * np.pi))
    print('E0 is ', erg2ev(p2e(p0))/1000, ' keV')
