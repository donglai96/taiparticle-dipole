
import sys
import taichi
import numpy as np
import constants as cst
from taichiphysics import *
from res_energy import *
from particle import Particle
import time

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
    print(alpha_eq,'alpha_eq',np.rad2deg(alpha_eq))
    alpha = get_pitchanlge_from_eq(alpha_eq,res_lat,direction) # at res
    alpha_init = get_pitchanlge_from_eq(alpha_eq,lat_init,direction)
    print('alpha_init is:',np.rad2deg(alpha_init),'rad',alpha_init)
    print('alpha_eq at init is', get_equator_pitchangle(alpha_init,lat_init))
    _,_,bz_lat = dipole_field(cst.B0,L_shell,cst.Planet_Radius,0,0,res_lat)
    wce_alpha = gyrofrequency(cst.Charge,cst.Me,bz_lat)

    n_lat = electron_density('cos',n0,res_lat)
    p0, k0 = get_resonance_p_whistler(w_res,wce_alpha,n_lat,alpha,nres = -1)
    print('momentum of resonating particles are:',p0)
    print('resonant wave number at reslat is :', k0)
    print('resonating frequency is (Hz) ', w_res/(2 * np.pi))
    print('E0 is ', erg2ev(p2e(p0))/1000, ' keV')

    # First write the particle motion

    # determine total T
    T_bounce = 1 / bouncef(L_shell,cst.Planet_Radius,p0,mass,alpha_eq)
    T_total = t_total_num * T_bounce

    gamma = (1 + p0**2 / (cst.Me**2*cst.C**2))**0.5
    wce_rel = wce0/gamma
    T_gyro = 2 * np.pi/ wce_rel

    dt = T_gyro * dt_num
    Nt = int(T_total/dt)

    print('time step is ', dt)
    print('total time is',T_total)
    print('total time step is', Nt)

    #
    ti.init(arch = ti.cpu,default_fp=ti.f64)

    particles = Particle.field(shape = (Np,))
    dt_taichi = ti.field(ti.f64, shape=())
    dt_taichi[None] = dt
    ################################################################
    # generate initial moment
    dphi = 2 * np.pi / Np

    pperp = p0 * np.sin(alpha_init)

    px_numpy = np.zeros(Np)
    py_numpy = np.zeros(Np)
    pz_numpy = np.zeros(Np)

    for n in range(Np):
        phi = dphi * n
        px_numpy[n] = pperp * np.cos(phi)
        py_numpy[n] = pperp * np.sin(phi)
        pz_numpy[n] = p0 * np.cos(alpha_init)

    px_init = ti.field(dtype = ti.f64,shape = (Np,))
    py_init= ti.field(dtype = ti.f64,shape = (Np,))
    pz_init= ti.field(dtype = ti.f64,shape = (Np,))

    px_init.from_numpy(px_numpy)
    py_init.from_numpy(py_numpy)
    pz_init.from_numpy(pz_numpy)

    ################################################################
    # record
    print('Record num is', Nt//record_num)
    p_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num + 1, Np))
    # E_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))

    # B_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num, Np))

    r_record_taichi = ti.Vector.field(n=3,dtype = ti.f64,shape = (Nt//record_num + 1, Np))
    phi_record_taichi = ti.field(dtype = ti.f64,shape = (Nt//record_num + 1, Np))
    ################################################################
    #
    @ti.kernel
    def init():
        for n in range(Np):
            particles[n].initParticles(mass, charge)
            particles[n].initPos(0.0,0.0,lat_init,L_shell)
            print('init position',particles[n].r)
            particles[n].initMomentum(px_init[n], py_init[n],pz_init[n])
            particles[n].get_pitchangle()
            print('The real init p',1e20*particles[n].p.norm())
            print('The initial pitchangle is',particles[n].alpha)
        B = dipole_field_taichi(L_shell,particles[n].r,cst.B0)
        print('The init B is', B)
        E = ti.Vector([0.0,0.0,0.0])

        for n in range(Np):
            particles[n].boris_push(-dt_taichi[None]/2,E,B)
            print('after first step')
            print(1e20*particles[n].p.norm())

    @ti.kernel
    def simulate_t():
        for n in range(Np):
            for tt in range(Nt):
                
                # B =ti.Vector([0.0,0.0,bz0])
                E = ti.Vector([0.0,0.0,0.0])
                B = dipole_field_taichi(L_shell,particles[n].r,cst.B0)
                # update the wave
                particles[n].t += dt_taichi[None] 
                particles[n].leap_frog(dt_taichi[None],E,B) # change nth particle's p and r
                particles[n].Ep = E
                particles[n].Bp = B
                phip = ti.atan2(particles[n].p[1],particles[n].p[0])
                particles[n].phi = phip
                if (particles[n].phi < 0):
                    particles[n].phi += 2*ti.math.pi
                if tt%record_num ==0:
                    #print('tt',tt)
                    # print('tt',tt)
                    # print('r',particles[n].r)
                    # print(tt//record_num)
                    
                    p_record_taichi[tt//record_num, n] = particles[n].p
                    r_record_taichi[tt//record_num, n] = particles[n].r
                    phi_record_taichi[tt//record_num, n] = particles[n].phi
    start_time = time.time()
    init()
    #print(particles[1].r)
    simulate_t()



    print('finished')
    # End of the main loop
    ###################################################
    time_used = time.time() - start_time
    print("--- %s seconds ---" % (time_used))
    
    ###################################################

    p_results = p_record_taichi.to_numpy()
    r_results = r_record_taichi.to_numpy()
    print('r0')
    print(r_results[0,0,2])
    # Ep_results = E_record_taichi.to_numpy()
    # Bp_results = B_record_taichi.to_numpy()
    phi_results = phi_record_taichi.to_numpy()
    with open(id + '/' + 'p_r_phi.npy','wb') as f:
        np.save(f,p_results)
        np.save(f,r_results)
        np.save(f,phi_results)
    # save the output info for checking
    with open(id + '/' + 'output.txt', 'w') as f:
        f.write(f"The resonating frequency is {w_res/(2 * np.pi)} (Hz)\n")

        f.write(f"momentum of resonating particles are: {p0}\n")
        f.write(f"E0 is  {erg2ev(p2e(p0))/1000} keV\n")
        f.write(f"--- %s seconds ---" % (time_used))