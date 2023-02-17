import numpy as np
import taichi as ti
import constants as cst
from scipy.optimize import fsolve

C = 3e10
M = 9.1094e-28 
Q = 4.8032e-10
def gyrofrequency(q, m ,B):
    return q * B / (m * cst.C)


def plasmafrequency(q,m,n):
    return np.sqrt(4 * np.pi *n / m) *q




def ev2erg(ev):
    return ev * 1.60218e-12
def erg2ev(erg):
    return erg / 1.60218e-12
# for electrons
def e2p(e, E0=M*C**2):
    return (e * (e + 2 * E0))**0.5 / C 

def p2e(p, E0=M*C**2):
    return (p**2 * C**2 + E0**2)**0.5 - E0 

def p2v(p,m= M):
    gamma_m = m * (1 + p**2 /(m * m*C**2)) **0.5
    return p/gamma_m

def v2p(v,m = M):
    gamma_m = m * 1 / (1 - v**2 / C**2)*0.5
    return gamma_m * v

def dipole_field(B0,L,R,x,y,latitude):
    lat_sin = np.sin(latitude)
    lat_cos = np.cos(latitude)
    b_z = B0/(L **3 * lat_cos**6) * np.sqrt(1 + 3 * lat_sin**2)

    dBdz = 3 * b_z * lat_sin/ (L * R * np.sqrt(1 + 3 * lat_sin**2)) \
        * (1 / (1 + 3 * lat_sin**2) + 2 /(lat_cos **2))
    b_x = -dBdz * (x / 2.0)
    b_y = -dBdz * (y / 2.0)

    return b_x,b_y,b_z


def get_pitchanlge_from_eq(alpha0,lat,signpb):
    tmp = (1 + 3 * np.sin(lat)**2)**0.25 / np.cos(lat)**3 * np.sin(alpha0)
    alpha = np.arcsin(tmp)
    if signpb< 0:
        alpha = np.pi - alpha
    return alpha

def get_equator_pitchangle(alpha,lat):
    return np.arcsin( (1 + 3 * np.sin(lat)**2)**( -0.25) * np.cos(lat)**3 * np.sin(alpha) )

@ti.func
def dipole_field_taichi(L, r, B0):
    """get dipole field vector
    The coordinates is z along the field line, x and y satisfy the
    divergence free law

    Args:
        L (ti.f32): L shell
        r (ti vector): location of particle
        B0 (ti.f32): magnetic field at equator
    """
    lat = r[2]
    x = r[0]
    y = r[1]
    cos_lat= ti.cos(lat)
    sin_lat = ti.sin(lat)
    Bz = B0/(L**3 * cos_lat**6) * ti.sqrt(1 + 3 * sin_lat ** 2)

    dBdz = 3 * Bz * sin_lat / (L * cst.Planet_Radius * ti.sqrt(1 + 3 * sin_lat ** 2)) \
        * (1 / (1 + 3 * sin_lat ** 2) + 2 /(cos_lat ** 2))

    Bx = -dBdz * (x / 2.0)
    By = -dBdz * (y / 2.0)
    B = ti.Vector([Bx,By,Bz])
    return B

def electron_density(dis,n0,lat):
    if dis == 'const':
        return n0
    elif dis == 'cos':
        return n0*(np.cos(lat)**-4)

def mirrorpoints(alpha_eq):
    def get_mirrorpoint2(x):
        return (1 - x)**3 - np.sin(alpha_eq)**2 * (1 + 3*x)**0.5
    sin2alpha_m = fsolve(get_mirrorpoint2, 0)
    
    alpha_m = np.arcsin(np.sqrt(sin2alpha_m))
    print('The mirror point is: ',np.rad2deg(alpha_m))
    return alpha_m
def bouncef(L,R,p,m,alpha_eq):
    T = 1.3802 - 0.31985 * (np.sin(alpha_eq) + np.sqrt(np.sin(alpha_eq)))
    gamma = (1 + p**2 /(m **2 * cst.C**2))**0.5
    return p/(4 * gamma * m * L * R* T)

def s_numpy(L, R, lat):
    x = np.sin(lat)
    tt = np.sqrt(x**2 + 1.0 / 3)
  
    return L * R * np.sqrt(3) * (0.5 * x * tt + np.log(np.sqrt(3) * abs(x + tt)) / 6.0)
def get_dipole_numpy(L, lat, B0):
    cos_lat= np.cos(lat)
    sin_lat = np.sin(lat)
    Bz = B0/(L**3 * cos_lat**6) * np.sqrt(1 + 3 * sin_lat ** 2)
    return Bz

@ti.func
def interp(xx,x,n):
    jl = -1
    ju = n
    jm = 0
    j = 0
    frac = 0.0
    
    ascnd = (xx[n - 1]>= xx[0])
    #print(ascnd)
    while ((ju - jl) > 1):
        jm = ti.floor((ju + jl)/2,dtype = ti.i32)
    
        if ((x >= xx[jm]) == ascnd):
            jl = jm
        else:
            ju = jm
    #print(jm)
            
    if (x == xx[0]):
        j = 0
    elif (x == xx[n-1]):
        j = n-2
    else:
        j = jl
    #print('find',j)
    if (j == -1):
        frac = -1
    elif (j == n-1):
        frac = n-1
    else:
        frac = j + (x - xx[j]) / (xx[j + 1] - xx[j])
        
    
        
    return frac
