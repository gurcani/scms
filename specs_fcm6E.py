#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:04:10 2018

@author: ogurcan
"""
import matplotlib.tri as tri
import matplotlib.pylab as plt
import h5py as h5
import numpy as np

flname="out_fcm6E.h5"
fl=h5.File(flname,"r")
k=fl["fields/k"][:]
fk=fl["fields/fk"][:]
tt=fl["fields/t"][:]
Et=fl["fields/E"][:,:]
E=np.mean(fl["fields/E"][-100:,:],0)
fl.close()
N=k.shape[0]
kn=np.abs(k)
g=kn[1]/kn[0]
alpha=np.angle(k[1])

Mn=kn*np.sin(alpha)

an=Mn*g**(-40)*(g**8-1); an[:16]=0
bn=Mn*g**(-12)*(g**16-1); bn[:4]=0
cn=Mn*(g**(-24)-2*g**16+g**(-8))
dn=Mn*g**(-12)*(g**32-1); dn[N-4:]=0
en=Mn*g**(16)*(g**8-1); en[N-8:]=0

n=np.arange(0,N)
tn=E**(3/2)
PE=-np.cumsum(an*tn[n-16]+bn*tn[n-4]+cn*tn[n]+dn*tn[(n+4)%N]+en*tn[(n+8)%N])
PW=-np.cumsum(an*tn[n-16]*kn**2+bn*tn[n-4]*kn**2+cn*tn[n]*kn**2+dn*tn[(n+4)%N]*kn**2+en*tn[(n+8)%N]*kn**2)
#PW=np.cumsum(an*tn[n-16]*kn[n-16]**2+bn*tn[n-4]*kn[n-4]**2+cn*tn[n]*kn**2+dn*tn[(n+4)%N]*kn[(n+4)%N]**2+en*tn[(n+8)%N]*kn[(n+8)%N]**2)

#kk=np.hstack((k,-k))
kk=np.hstack((np.log10(np.abs(k))*np.exp(1j*np.angle(k)),np.log10(np.abs(k))*np.exp(1j*(np.angle(k)+np.pi))))
ffk=np.hstack((fk,fk))
#plt.loglog(kn,np.mean(E[-100:],0)/kn,'s-')

Espr=np.hstack((E/kn,E/kn))

#j1=(np.abs(kk)>2E3)
#j2=(np.abs(kk)>2E3)
tr=tri.Triangulation(np.real(kk),np.imag(kk))
plt.figure(dpi=200)
plt.rc('font', family='serif',size=14)
plt.rc('text', usetex=True)
#plt.tripcolor(tr,np.log10(Espr[Espr<1e-8]), shading='flat',vmin=-20,vmax=-8,cmap='ocean')
#tr=tri.Triangulation(np.real(kk[Espr>1e-8]),np.imag(kk[Espr>1e-8]))
trc=plt.tripcolor(tr,np.log10(Espr), shading='flat',vmin=-14,vmax=0.0,cmap='seismic',rasterized=True)
plt.tight_layout()
plt.axis('square')
plt.title('$\\log(E(k))$')
plt.xlabel('$\\log(k)\\cos(\\theta_k)$')
plt.ylabel('$\\log(k)\\sin(\\theta_k)$')
plt.colorbar()
trc.axes.axis([-7,7,-7,7])
plt.scatter(np.real(kk[ffk>0]),np.imag(kk[ffk>0]),marker='x',c='k')
ax2=trc.axes.twinx()
ax2.set_ylabel('$\\log(E(k))$')
ax2.plot(np.log10(kn),np.log10(E/kn),c='orange')
ax2.axis([-7,7,-20,-2])
ax2.plot(np.log10(kn[kn<2e3]),np.log10(1e-1*kn[kn<2e3]**(-5/3)),'w')
ax2.plot(np.log10(kn[kn>4e3]),np.log10(4e3*kn[kn>4e3]**(-3)),'w')
plt.figure(dpi=200)
plt.pcolormesh(tt,np.log10(kn),np.log10(Et/kn).T,vmin=-14,vmax=0,cmap='seismic',rasterized=True)
plt.tight_layout()
plt.axis([0,30,0,np.log10(kn[-1])])
plt.title('$\\log(E(k,t))$')
plt.xlabel('$\\log(k)$')
plt.ylabel('$t$')
plt.colorbar()
plt.figure(dpi=200)
plt.semilogx(kn,PE/np.max(np.abs(PE)),kn,PW/np.max(np.abs(PW)))
plt.xlabel('$k_n$')
plt.ylabel('$\\Pi^E_n, \\Pi^W_n$')
nj=np.arange(1,11)
Sn=np.zeros((nj.shape[0],N))
for j in nj:
    Sn[j-nj[0],:]=np.mean(Et[-100:,:]**(j/2),0)
zeta1=np.zeros(nj.shape[0])
zeta2=np.zeros(nj.shape[0])
ptr1=np.arange(70,140)
ptr2=np.arange(250,350)
for j in nj:
    a,b=np.polyfit(np.log(kn[ptr1]), np.log(Sn[j-nj[0],ptr1]), 1)
    zeta1[j-nj[0]]=-a
    a,b=np.polyfit(np.log(kn[ptr2]), np.log(Sn[j-nj[0],ptr2]), 1)
    zeta2[j-nj[0]]=-a
plt.figure(dpi=200)
plt.plot(nj,zeta1,'kx-',nj,zeta2,'rx-',nj,nj/3,'--',nj,nj,'--')
plt.xlabel('$n$')
plt.xlabel('$S_n$')
