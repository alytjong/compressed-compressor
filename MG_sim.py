#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:21:38 2020

@author: anovelia based on Yiming's MG code
Non-trivial equilibrium
"""

import sys
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib import animation, rc
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import time
from matplotlib import rc
from IPython.display import HTML
import pickle
import cmath
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


class MG_sim:
    def __init__(self, N,gamma,T,dt, IC, **parameters):
        
        self.N=N
        self.gamma=gamma
        self.T=T
        self.dt = dt
        self.g0=IC[0]
        self.Phi0=IC[1]
        self.Psi0=IC[2]
        self.dict_para={}
        for key, value in parameters.items(): #e.g. para={'l_c':8,'D':2,'H':0.18,'W':0.25}
            self.dict_para[key]=value 
        try:
            self.l_c=self.dict_para['l_c']
            self.B  =self.dict_para['B']
            self.H  =self.dict_para['H']
            self.W  =self.dict_para['W']
            self.m  =self.dict_para['m']
            self.nu  =self.dict_para['nu']
            self.a  =self.dict_para['a']
            
        except KeyError as e:
            print('Parameter Error', e)
            pass
    
        self.psi_c0=1.67*self.H
        self.Delta=self.__Delta()
        print('Delta=', self.Delta)
        self.phie_c,self.psie_c,self.gamma_c_ode,self.gamma_c_pde=self.__EP_critical()
        
    def __Delta(self):
        r=np.sqrt(1-self.nu*self.W/3/self.a/self.H)
        Num=self.psi_c0+self.H*(1+1.5*r-0.5*np.power(r,3))
        Den=self.W*(1+r)
        return Num/Den-self.a/4/np.square(self.B)/self.nu
    
    def __EP_critical(self):
        #critical bifurcation point of ode, pde, and equilibrium

        X = (self.W**3)*self.psi_c0/self.H
        #Y as a function of gamma
        Y = lambda x: (2*self.W**3/3/self.H)*(1/x**2-3*self.H/2/self.W**2)
        #Phi_e as a function of gamma
        Phi_e = lambda x: (np.cbrt(X - Y(x)**3 + np.sqrt(X**2 - 2*X*Y(x)**3)) 
            + np.cbrt(X - Y(x)**3 - np.sqrt(X**2 - 2*X*Y(x)**3)) - Y(x))
        
        #psi_c_prime as a function of gamma
        psi_c_prime = lambda x: (3*self.H/2/self.W**2)*Phi_e(x)*(2- Phi_e(x)/self.W)
        
        ODE_bif = lambda x: x**2/8/self.B**2/Phi_e(x) - psi_c_prime(x)
        muc_ode = fsolve(ODE_bif, self.gamma)
        
        #PDE_bif as a function of gamma
        PDE_bif = lambda x: psi_c_prime(x) - self.nu/2/self.a
        muc_pde = fsolve(PDE_bif, self.gamma)
        
        PDE_eig = (psi_c_prime(self.gamma) - self.nu/2/self.a - 0.5*1j/self.a)*self.a/(self.a*self.m)
        print("PDE eigenvalue is" + str(PDE_eig))
        if np.real(PDE_eig) < 0:
            print("PDE is stable")
        else:
            print("PDE is unstable")
            
        ODE_eig = (psi_c_prime(self.gamma) - self.gamma**2/8/self.B**2/Phi_e(self.gamma) + cmath.sqrt((psi_c_prime(self.gamma) + self.gamma**2/8/self.B**2/Phi_e(self.gamma))**2 - 1/self.B**2))/2/self.l_c
        print("ODE eigenvalue is" + str(ODE_eig))
        if np.real(ODE_eig) < 0:
            print("ODE is stable")
        else:
            print("ODE is unstable")
        
        #psi_c=lambda x: self.psi_c0+self.H*(1+1.5*(x/self.W-1)-0.5*(np.power((x/self.W-1),3)))-x**2/self.gamma/self.gamma
        xec = Phi_e(self.gamma)
        yec = np.square(xec/self.gamma)

        self.xec = xec
        self.yec = yec
        return (xec,yec,muc_ode,muc_pde)
        
    def __setup_domain(self):
        L=2*np.pi
        self.tht2=np.linspace(-L/2, L/2, self.N+1)
        self.tht=self.tht2[0:self.N]
        self.dtht = 2*np.pi/self.N
        self.t=np.arange(0,self.T,self.dt)
        k1 = np.linspace(0,int(self.N/2)-1,int(self.N/2))
        k2=np.linspace(-int(self.N/2),-1,int(self.N/2))
        self.k=np.append(k1,k2)*(2*np.pi/L)
        
    def __setup_IC(self):
        self.__setup_domain()
        self.ICfft = np.fft.fft(self.g0(self.tht))
        self.ICfft = np.append(self.ICfft, self.Phi0)
        self.ICfft = np.append(self.ICfft, self.Psi0)
        
    def __EP(self, mu):
        #equilibrium point of big system
        psi_c=lambda x: self.psi_c0+self.H*(1+1.5*(x/self.W-1)-0.5*(np.power((x/self.W-1),3)))-x**2/mu/mu
        xe=fsolve(psi_c, 0.5)
        ye=np.square(xe/mu)
        self.xe = xe
        self.ye = ye
        return (xe,ye)
        
    def dynamics(self,t,z):
        
        psi_c = lambda x: self.psi_c0+self.H*(1+1.5*(x/self.W-1)-0.5*(np.power((x/self.W-1),3)))
        psi_c_prime = lambda x: 1.5*(self.H/self.W)*(1 - np.power((x/self.W-1),2))
        psi_c_pprime = lambda x: -3*(self.H/np.square(self.W))*(x/self.W-1)
        psi_c_ppprime = -3*self.H/np.power(self.W,3)
        Kinv = np.abs(self.k)/(np.abs(self.k) + self.a*self.m)
        n=np.size(z)
        ghat=z[0:n-2]   
        Phi=z[n-2]
        Psi=z[n-1]
        
        g=np.fft.ifft(ghat).real
        
        psi_c_bar = np.sum(psi_c(Phi + g)*self.dtht)/2/np.pi
        
        dghat = (-0.5*Kinv*self.nu*np.power(self.k,2)*ghat - 0.5*Kinv*1j*self.k*ghat + 
            + self.a*Kinv*np.fft.fft(psi_c(Phi + g) - psi_c_bar))
        
        dPhi = (psi_c_bar - Psi)/self.l_c
        
        dPsi = (Phi - self.gamma*np.sqrt(Psi))/(4*np.square(self.B)*self.l_c)
        
        a=np.append(dghat,dPhi)
        a=np.append(a,dPsi)
        return a
    
    def Phi_e (self, x,mu):
        psi_c=lambda x: self.psi_c0+self.H*(1+1.5*(x/self.W-1)-0.5*(np.power((x/self.W-1),3)))
        return psi_c(x)-x**2/mu/mu
    
    def __plotEP(self):
        xe=[]
        ye=[]
        
        if (self.Delta<0):
            mu_0=0.45
            N=int((self.gamma_c_ode-mu_0)/0.01)
            for i in range (0,20):
                mu2=mu_0+i*0.01
                x,y=self.__EP(mu2)        
                xe.append(x[0])
                ye.append(y[0])
            plt.plot(xe[0:N+1],ye[0:N+1],'r--', label='Unstable Equilibrium Points Region')
            plt.plot(xe[N:20],ye[N:20],'g-',  label='Stable Equilibrium Points Region')
            plt.legend(loc='lower right')
        else:
            mu_0=0.0001
            for i in range (0,100):
                mu2=mu_0+i*0.01
                x,y=self.__EP(mu2)        
                xe.append(x[0])
                ye.append(y[0])
            plt.plot(xe[0:60],ye[0:60],'r--', label='Unstable Equilibrium Points')
            plt.plot(xe[60:100],ye[60:100],'g-', label='Stable Equilibrium Points')
            plt.legend(loc='lower right')
    
    def Solve(self):
        tic=time.time()
        print('Solving MG Model...')
        self.__setup_IC()
        self.soln_fourier=solve_ivp(self.dynamics, [0,self.T], self.ICfft, t_eval=self.t)#,full_output=True)
        print('Soln Generated')
        self.n=np.size(self.ICfft)
        self.nt=np.size(self.t)
        #self.gsoln0=np.zeros((int(self.nt),self.N))
        #self.gsoln=np.zeros((int(self.nt),self.N+1))
        self.gsoln=np.zeros((int(self.nt),self.N))
        for i in range (int(self.nt)):
            
            gsoln_fourier=self.soln_fourier.y[0:self.n-2,i] 
            #self.gsoln0[i]=np.fft.ifft(gsoln_fourier).real
            #self.gsoln[i]=np.append(self.gsoln0[i],self.gsoln0[i,0])
            self.gsoln[i]=np.fft.ifft(gsoln_fourier).real
            
        self.Phi_soln=self.soln_fourier.y[self.n-2,:].real
        self.Psi_soln=self.soln_fourier.y[self.n-1,:].real
        
        toc=time.time()-tic
        print('Computation Time:', toc, 'seconds')
        self.gsoln = self.gsoln[:,0:self.N]
        return (self.gsoln, self.Phi_soln, self.Psi_soln)
        
    def plot(self):
        print('Plotting...')
        xe,ye=self.__EP(self.gamma)
        plt.figure(1)
        plt.ylabel('$\Psi$')
        plt.xlabel('$\Phi$')
        plt.title('Phase Portrait of $\Phi$ and $\Psi$')
        #if (self.Delta<0):
        #    plt.axis([0.325,0.550, 0.54, 0.70])
        #else:
        #    plt.axis([-0.3, 0.9, 0, 0.85])
        #self.__plotEP()
        #plt.plot(self.Phi0,self.Psi0,'-bo', label='Initial Point')
        psi_c = lambda x: self.psi_c0+self.H*(1+1.5*(x/self.W-1)-0.5*(np.power((x/self.W-1),3)))
        xe_range = np.linspace(0,xe+0.5,100)
        plt.plot(xe[0], ye[0],'-yo', label='Equilibrium Point')
        plt.plot(xe_range, psi_c(xe_range), label='Compressor characteristic')
        plt.plot(xe_range, xe_range**2/self.gamma**2, label = "Throttle characteristic")
        #plt.plot(self.Phi_soln,self.Psi_soln, label='Phase Portrait')
        plt.legend(loc='lower right')
        
    def plotPP(self):
        print('Plotting...')
        xe,ye=self.__EP(self.gamma)
        plt.figure(2)
        plt.ylabel('$\Psi$')
        plt.xlabel('$\Phi$')
        plt.title('Phase Portrait of $\Phi$ and $\Psi$')
        plt.plot(self.Phi_soln,self.Psi_soln, label='Phase Portrait')
        
    def createAnimation(self):
        print('Creating Animation...')
        fig, ax = plt.subplots()
        ax.set_xlim((-np.pi, np.pi))
        ax.set_ylim((-0.01, 0.01))
        line1, = ax.plot([], [], lw=3)
        
        #initialization function: background of each frame
        def init():
            line1.set_data([], [])
            return (line1,)
        
        def animate(iii):
            x1 = self.tht
            y1 = self.gsoln[iii,:]
            line1.set_data(x1, y1)
            return (line1,)
        
        #call the animator. blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init, blit=True)
        HTML(anim.to_html5_video())

        # v=self.gsoln
        # fig=plt.figure(2)
        # ax=plt.axes(xlim=(-np.pi,np.pi), ylim=(-3,3))
        # line,=ax.plot([],[], lw=2 )
        # Nv=int(np.size(v)/self.N)
        # def init_line():
        #     line.set_data([], [])
        #     return line,
        # def animate(i):
        #     x=np.linspace(-np.pi,np.pi,self.N+1) 
        #     y=v[i]
        #     line.set_data(x,y)
        #     plt.xlabel(r'$\theta$')
        #     plt.ylabel('g(t,'+ r'$\theta$)')
        #     plt.title('Dynamics of g(t,' +r'$\theta$)')   
        #     plt.show()                      
        #     return line,
    
        # anim = animation.FuncAnimation(fig, animate, interval=20,init_func=init_line,
        #                        frames=self.T,  blit=True)
        # anim.save('gmode_animation'+'.gif', fps=30, extra_args=['-vcodec', 'libx264'])
        # HTML(anim.to_html5_video())
        return 
   
#def main():
    #N=128                  #spatial discretization 
    #gamma=0.6              #throttle coefficient
    #T=500                  #time steps
    #IC=[lambda x:0.005*np.sin(x), 0.1, 0.1]     #I.C. 
    #para={'l_c':8,'B':0.5,'H':0.18,'W':0.25, 'm':1.75, 'nu':0.1, 'a':1/3.5}
    #parameter set
    #MG_eq=MG_equilibrium_sim(N,gamma,T,IC,**para)    #Initialize the class MG_Sim    
    #[G,Phi,Psi] = MG_eq.Solve()  
    #with open('MGdata.pickle', 'wb') as f:
    #    pickle.dump([G,Phi,Psi], f)                      #Solve MG model
    #MG_eq.plot()
    #MG_eq.createAnimation()
        
#if __name__=='__main__':
#    main()     
#%%
# print('New version, fixed the a*Kinv bug')
# N=128                #spatial discretization 
# gamma=0.6          #throttle coefficient
# T=1000               #time steps
# dt = 0.1
# IC=[lambda x:0.1*np.sin(x), 0.1, 0.1]     #I.C. 
# L = 2*np.pi
# tht2=np.linspace(-L/2, L/2, N+1)
# tht=tht2[0:N]
    
# #stable gamma = 0.66
# #para={'l_c':8,'B':0.5,'H':0.18,'W':0.25, 'm':1.75, 'nu':0.1, 'a':1/3.5}
# #surge gamma = 0.6
# para={'l_c':8,'B':2,'H':0.18,'W':0.25, 'm':1.75, 'nu':0.1, 'a':1/3.5}
# #stall gamma = 0.57
# #para={'l_c':8,'B':0.15,'H':0.18,'W':0.25, 'm':1.75, 'nu':0.1, 'a':1/3.5}
# #combo gamma=0.5
# #para={'l_c':8,'B':1,'H':0.18,'W':0.25, 'm':1.75, 'nu':0.1, 'a':1/3.5}

# #parameter set
# MG_eq = MG_sim(N,gamma,T,dt, IC,**para)    #Initialize the class MG_Sim    
# [G,Phi,Psi] = MG_eq.Solve()                #Solve MG model
# MG_eq.plot()
# MG_eq.plotPP()

# #%%
# t = np.arange(0,T,dt)
# plt.figure(3)
# plt.plot(t,Phi)
# plt.plot(t,MG_eq.xec*np.ones(np.shape(t)))
# plt.ylabel('Phi'), plt.xlabel('t')
# plt.figure(4)
# plt.plot(t,Psi)
# plt.plot(t,MG_eq.yec*np.ones(np.shape(t)))
# plt.ylabel('Psi'), plt.xlabel('t')
# plt.figure(5)
# ampG = np.zeros(np.size(t))
# for iii in range(0,np.size(t)):
#     ampG[iii] = max(G[iii,:])
# plt.plot(t, ampG)
# plt.ylabel('Amplitude G'), plt.xlabel('t')


    