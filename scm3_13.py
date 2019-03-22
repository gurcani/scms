#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:39:27 2018

@author: ogurcan
"""

# Three term single chain model of two dimensional euler system

import numpy as np
import h5py as h5
import time
from scipy.stats import norm
import scipy.integrate as spi

# l = 1, m = 3

N=110                       # number of nodes
nu=1.0e-33                  # kinematic viscosity.
nuL=1.0e2                  # kinematic viscosity.
t0=0.0                      # initial time
t1=10000.0                   # final time
dt=1e-4;                     # time step for output.
dtout=1e-1;
wecontinue=False            # do we continue from an existing file.
flname="out_scm3_13.h5" 
n=np.arange(0,N)
g=np.sqrt(1/3*(1+1/2**(1/3)*((29+3*np.sqrt(93))**(1/3)+(29-3*np.sqrt(93))**(1/3))))
alpha=np.arccos(g**(-3)/2)
#alpha=np.pi-np.arccos(-g**3/2)
th=np.mod(alpha*n,2*np.pi)
k0=1.0
kn=k0*g**n
k=kn*np.exp(1j*th)
Mn=kn**2*np.sin(alpha)

an=Mn*g**(-11)*(g**2-1); an[0:3]=0
bn=Mn*g**(-3)*(1-g**6); bn[0:2]=0; bn[N-1]=0
cn=Mn*g**3*(g**4-1); cn[N-3:]=0

An=np.array((an,bn,cn)).T.ravel()
Dn=nu*kn**4+nuL/kn**6

Nf=2
kf0=1.0j*np.sqrt(kn[-1]*kn[0])
kf=[k[np.argmin(np.abs(k-kf0))]]
for l in range(Nf-1):
    kf=np.append(kf,(k[~(np.isin(k,kf))])[np.argmin(np.abs(k[~(np.isin(k,kf))]-kf0))])
kf0=-1.0j*np.sqrt(kn[-1]*kn[0])
for l in range(Nf):
    kf=np.append(kf,(k[~(np.isin(k,kf))])[np.argmin(np.abs(k[~(np.isin(k,kf))]-kf0))])
fk=1e-4*np.complex_(np.isin(k,kf))/Nf/20
gk=fk.copy()

def force_update():
    global fk
    fk=gk*(norm.rvs(size=(gk.shape[0]))+1j*norm.rvs(size=(gk.shape[0])))


dydt=np.zeros(N*2)

def func(t,y):
    phi=y.view(dtype=complex);
#    dphidt=an*phi[n-3].conj()*phi[n-1]+bn*phi[n-2]*phi[(n+1)%N]+cn*phi[(n+2)%N]*phi[(n+3)%N].conj()-Dn*phi+fk
    dphidt=an*phi[n-2].conj()*phi[n-3].conj()+bn*phi[n-1].conj()*phi[(n+2)%N].conj()+cn*phi[(n+3)%N].conj()*phi[(n+1)%N].conj()-Dn*phi+fk
    return dphidt.view(dtype=float)

if(wecontinue==True):
    #setting up the output hdf5 file
    fl=h5.File(flname,"r")
    phi=fl["fields/u"].value
    tt=fl["fields/t"].value
    fl.close()
    t0=tt[-1]
    phi0=phi[-1,:,:]
    fl=h5.File(flname,"w")
    grp=fl.create_group("fields")
    grp.create_dataset("k",data=k)
    i=phi.shape[0]
    grp=fl["fields"]
    phires=grp.create_dataset("phi",(i,N),maxshape=(None,N),dtype=complex)
    tres=grp.create_dataset("t",(i,),maxshape=(None,),dtype=float)
    phires[:,:]=phi;
    tres[:]=tt;
else:
    phi0=np.zeros(N,dtype=complex)
    i=0;
    fl=h5.File(flname,"w")
    grp=fl.create_group("fields")
    grp.create_dataset("k",data=k)
    phires=grp.create_dataset("phi",(1,N),maxshape=(None,N),dtype=complex)
    tres=grp.create_dataset("t",(1,),maxshape=(None,),dtype=float)
#r=spi.RK45(func,t0,u0.view(dtype=float),t1,max_step=dt,atol=1e-12,rtol=1e-6)
r=spi.RK45(func,t0,phi0.view(dtype=float),t1,max_step=dt,atol=1e-12,rtol=1e-6)

#r = ode(func, 0).set_integrator('dopri5',nsteps=1E6)
#r.set_initial_value(u0.view(dtype=float), t0)
force_update()
ct=time.time()
epst=1e-12
while(r.status=='running'):
    print("t=",r.t);
    toldout=r.t
    while(r.t<np.round(toldout/dtout+1)*dtout-epst and r.status=='running'):
        told=r.t
        force_update()
        while(r.t<np.round(told/dt+1)*dt-epst and r.status=='running'):
            res=r.step()
    print(time.time()-ct,"seconds elapsed.")
    phi=r.y.view(dtype=complex);
    phires.resize((i+1,N))
    tres.resize((i+1,))
    phires[i,:]=phi
    tres[i]=r.t
    fl.flush()
    i=i+1;
fl.close()
