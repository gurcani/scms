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

N=440                       # number of nodes
nu=1.0e-25                  # kinematic viscosity.
nuL=1.0e1                  # kinematic viscosity.
t0=0.0                      # initial time
t1=200.0                   # final time
dt=1e-4;                     # time step for output.
dtout=1e-1;
wecontinue=False            # do we continue from an existing file.
flname="out_fcm6E.h5" 
n=np.arange(0,N)
g0=1/2**(1/6)*np.sqrt((1-np.sqrt(23/27))**(1/3)+(1+np.sqrt(23/27))**(1/3))
alpha0=np.arccos(-g0**3/2)
g=g0**(1/4)
alpha=alpha0/4
th=np.mod(alpha*n,2*np.pi)
k0=1.0
kn=k0*g**n
k=kn*np.exp(1j*th)

Mn=kn*np.sin(alpha)

an=Mn*g**(-40)*(g**8-1); an[:16]=0
bn=Mn*g**(-12)*(g**16-1); bn[:4]=0
cn=Mn*(g**(-24)-2*g**16+g**(-8))
dn=Mn*g**(-12)*(g**32-1); dn[N-4:]=0
en=Mn*g**(16)*(g**8-1); en[N-8:]=0

Dn=nu*kn**4+nuL/kn**6

#Nf=16
#kf0=1.0j*np.sqrt(kn[-1]*kn[0])
#kf=[k[np.argmin(np.abs(k-kf0))]]
#for l in range(Nf-1):
#    kf=np.append(kf,(k[~(np.isin(k,kf))])[np.argmin(np.abs(k[~(np.isin(k,kf))]-kf0))])
#kf0=-1.0j*np.sqrt(kn[-1]*kn[0])
#for l in range(Nf):
#    kf=np.append(kf,(k[~(np.isin(k,kf))])[np.argmin(np.abs(k[~(np.isin(k,kf))]-kf0))])
#fk=np.float_(np.isin(k,kf))/Nf/20

Nf=2
kf0=1.0j*2e3
kf=[k[np.argmin(np.abs(k-kf0))]]
for l in range(Nf-1):
    kf=np.append(kf,(k[~(np.isin(k,kf))])[np.argmin(np.abs(k[~(np.isin(k,kf))]-kf0))])
kf0=-1.0j*2e3
for l in range(Nf):
    kf=np.append(kf,(k[~(np.isin(k,kf))])[np.argmin(np.abs(k[~(np.isin(k,kf))]-kf0))])
fk=1e-1*np.float_(np.isin(k,kf))/Nf/20

#fk=np.zeros(N)
#fk[200:232]=1e-1

dydt=np.zeros(N*2)
eps=1e-30
def func(t,E):
    E[E<eps]=0
    tn=E**(3/2)
    dEdt=an*tn[n-16]+bn*tn[n-4]+cn*tn[n]+dn*tn[(n+4)%N]+en*tn[(n+8)%N]-Dn*E+fk
    return dEdt

if(wecontinue==True):
    #setting up the output hdf5 file
    fl=h5.File(flname,"r")
    E=fl["fields/E"].value
    tt=fl["fields/t"].value
    fl.close()
    t0=tt[-1]
    E0=E[-1,:,:]
    fl=h5.File(flname,"w")
    grp=fl.create_group("fields")
    grp.create_dataset("k",data=k)
    grp.create_dataset("fk",data=fk)
    i=E.shape[0]
    grp=fl["fields"]
    Eres=grp.create_dataset("E",(i,N),maxshape=(None,N),dtype=float)
    tres=grp.create_dataset("t",(i,),maxshape=(None,),dtype=float)
    Eres[:,:]=E;
    tres[:]=tt;
else:
    E0=np.zeros(N)
    i=0;
    fl=h5.File(flname,"w")
    grp=fl.create_group("fields")
    grp.create_dataset("k",data=k)
    grp.create_dataset("fk",data=fk)
    Eres=grp.create_dataset("E",(1,N),maxshape=(None,N),dtype=float)
    tres=grp.create_dataset("t",(1,),maxshape=(None,),dtype=float)

#r=spi.RK45(func,t0,u0.view(dtype=float),t1,max_step=dt,atol=1e-12,rtol=1e-6)
r=spi.RK45(func,t0,E0,t1,max_step=dt,atol=1e-12,rtol=1e-6)

#r = ode(func, 0).set_integrator('dopri5',nsteps=1E6)
#r.set_initial_value(u0.view(dtype=float), t0)
ct=time.time()
epst=1e-12
while(r.status=='running'):
    print("t=",r.t);
    toldout=r.t
    while(r.t<np.round(toldout/dtout+1)*dtout-epst and r.status=='running'):
        told=r.t
        while(r.t<np.round(told/dt+1)*dt-epst and r.status=='running'):
            res=r.step()
    print(time.time()-ct,"seconds elapsed.")
    E=r.y;
    Eres.resize((i+1,N))
    tres.resize((i+1,))
    Eres[i,:]=E
    tres[i]=r.t
    fl.flush()
    i=i+1;
fl.close()
