# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:39:25 2022

IEEE C37.118 (2018) Standard PMU Tests
To run the tests pass a valid pmu class, followed by
- plot (boolean value, 0 = do not plot, 1 = plot results)
- tests (array of strings or single string 'all' to run all tests)
   * Note: all tests can have any capitalization *
    - 'frequency'  : steady state freq test
    - 'magnitude'  : steady state magnitude test
    - 'harmonic'   : harmonic test
    - 'oob'        : out of bound interference test (only for m class pmus)
    - 'amodulation': amplitude modulation
    - 'pmodulation': phase modulation
    - 'pramp'      : positive ramp
    - 'nramp'      : negative ramp
    - 'pmagstep'   : positive magnitude step
    - 'nmagstep'   : negative magnitude step
    - 'pphasestep' : positive phase step
    - 'nphasestep' : negative phase step

@author: Dylan Tarter
"""
from IPython import get_ipython

get_ipython().magic('reset -sf')
get_ipython().magic('clear -sf')

import matplotlib.pyplot as plt
import numpy as np
import math

def run(pmu,doPlot,tests,name="None"):
    if(isinstance(tests,list)):
        tests = [x.lower() for x in tests] # convert everything to lower case          
    else:
        if(tests == 'all'):
            tests = ['frequency','magnitude','harmonic','oob','amodulation','pmodulation','pramp','nramp','pmagstep','nmagstep','pphasestep','nphasestep'] # make all contain all cases
        else:
            print('[PMU/Tests] ERROR! Unrecognized Test Input. You gave "'+str(tests)+'"')
            print('use either \'all\' or an array of tests defined in the function header.')
    
    results = dict()
    
    # if no name is specified for the test, grab the PMU's name
    if(name == "None"):
        name = pmu.estimator.__name__()+', Fs='+str(pmu.fs)+', F0='+str(pmu.f0)+', RR='+str(pmu.RR)
    results['name']  = name
    results['class'] = pmu.estimator.tclass
    results['fs']    = pmu.fs
    results['f0']    = pmu.f0
    results['RR']    = pmu.RR
    
    # perform tests if they are contained in the tests array
    if('frequency' in tests):
        results['frequency'] = ssFreq (pmu,doPlot)
    if('magnitude' in tests):
        results['magnitude'] = ssMag  (pmu,doPlot)
    if('harmonic' in tests):
        results['harmonic'] = ssHarm (pmu,doPlot)
    if('oob' in tests):
        results['oob'] = ssOOB  (pmu,doPlot)
    if('amodulation' in tests): # freq. modulation
        results['amodulation'] = dynOsc (pmu,1,doPlot)
    if('pmodulation' in tests): # phase modulation
        results['pmodulation'] = dynOsc (pmu,0,doPlot)
    if('pramp' in tests): # positive Ramp
        results['pramp'] = dynRamp(pmu,1,doPlot)
    if('nramp' in tests): # negative Ramp
        results['nramp'] = dynRamp(pmu,0,doPlot)
    if('pmagstep' in tests): # Positive Mag Step
        results['pmagstep'] = dynStep(pmu,1,1,doPlot)
    if('nmagstep' in tests): # Negative Mag Step
        results['nmagstep'] = dynStep(pmu,1,0,doPlot)
    if('pphasestep' in tests):# Positive Phase Step
        results['pphasestep'] = dynStep(pmu,0,1,doPlot)
    if('nphasestep' in tests):# Negative Phase Step
        results['nphasestep'] = dynStep(pmu,0,0,doPlot)
    
    return results

"""Section 6.3"""    
def ssFreq(pmu,doPlot):
    fs = pmu.fs
    f0 = pmu.f0
    RR = pmu.RR
    tclass = pmu.estimator.tclass
    
    """Table 2 & 3"""
    maxtve = 1
    if(tclass == 'p'):
        maxfe  = 0.005
        maxrfe = 0.4
        frange = np.arange(-2,2+0.1,0.1)
    elif(tclass == 'm'):
        maxfe  = 0.005
        maxrfe = 0.1
        if(RR <= 10):
            frange = np.arange(-2,2+0.1,0.1)
        elif(RR < 25):
            frange = np.arange(-RR/5,RR/5+0.1,0.1)
        else:
            frange = np.arange(-5,5+0.1,0.1)
    Xm = 1
    t  = np.arange(0,5,1/fs)

    k = 0
    tve = np.empty(np.size(frange))*np.nan
    fe  = np.empty(np.size(frange))*np.nan
    rfe = np.empty(np.size(frange))*np.nan
    for df in frange:
        xa = Xm*np.cos(2*math.pi*(f0+df)*t) # eq. 12
        xb = Xm*np.cos(2*math.pi*(f0+df)*t - 2*math.pi/3) # eq. 13
        xc = Xm*np.cos(2*math.pi*(f0+df)*t + 2*math.pi/3) # eq. 14
        
        x = np.vstack((xa,xb,xc))
        
        T,X,F,DF = pmu.run(t,x)
        
        # eq. 21
        XrM = np.ones(np.size(T)) * Xm * math.sqrt(2)/2
        XrA = 2*math.pi*df*T
        Xr  = XrM*np.cos(XrA) + 1j*XrM*np.sin(XrA);
        Fr  = np.ones(np.size(T)) * (f0+df);
        DFr = np.zeros(np.size(T));
        
        tve[k] = np.nanmax(abs(TVE(Xr,X[:,3])));
        fe [k] = np.nanmax(abs(FE(Fr,F)));
        rfe[k] = np.nanmax(abs(RFE(DFr,DF)));
        k += 1;
    
    frange = frange + f0
    
    results = dict()
    results['range'] = frange
    results['tve'] = tve
    results['fe'] = fe
    results['rfe'] = rfe
    results['limits'] = [maxtve,maxfe,maxrfe]
    
    # if plot, mimic a dictionary that has just this test, and pass it to the plot function
    if(doPlot):
        k = dict()
        k['frequency'] = results
        k['name'] = '_nolegend_'
        plot([k])
    
    return results
 
"""Section 6.3"""       
def ssMag(pmu,doPlot):
    fs = pmu.fs
    f0 = pmu.f0
    RR = pmu.RR
    tclass = pmu.estimator.tclass
    
    """Table 2 & 3"""
    maxtve = 1
    if(tclass == 'p'):
        maxfe  = 0.005
        maxrfe = 0.4
        mrange = np.arange(0.8,1.2+0.1,0.1)
    elif(tclass == 'm'):
        maxfe  = 0.005
        maxrfe = 0.1
        mrange = np.arange(0.1,1.2+0.1,0.1)
        
    Xm = 1
    t  = np.arange(0,5,1/fs)

    k = 0
    tve = np.empty(np.size(mrange))*np.nan
    fe  = np.empty(np.size(mrange))*np.nan
    rfe = np.empty(np.size(mrange))*np.nan
    for m in mrange:
        xa = Xm*m*np.cos(2*math.pi*(f0)*t)
        xb = Xm*m*np.cos(2*math.pi*(f0)*t - 2*math.pi/3)
        xc = Xm*m*np.cos(2*math.pi*(f0)*t + 2*math.pi/3)
        
        x = np.vstack((xa,xb,xc))
        
        T,X,F,DF = pmu.run(t,x)
        
        XrM = np.ones(np.size(T)) * Xm * math.sqrt(2)/2 * m
        XrA = 2*math.pi*0*T
        Xr  = XrM*np.cos(XrA) + 1j*XrM*np.sin(XrA);
        Fr  = np.ones(np.size(T)) * (f0);
        DFr = np.zeros(np.size(T));
        
        tve[k] = np.nanmax(abs(TVE(Xr,X[:,3])));
        fe [k] = np.nanmax(abs(FE(Fr,F)));
        rfe[k] = np.nanmax(abs(RFE(DFr,DF)));
        k += 1;
    
    results = dict()
    results['range'] = mrange
    results['tve'] = tve
    results['fe'] = fe
    results['rfe'] = rfe
    results['limits'] = [maxtve,maxfe,maxrfe]
    
    if(doPlot):
        k = dict()
        k['magnitude'] = results
        k['name'] = '_nolegend_'
        plot([k])
        
    return results    

def ssHarm(pmu,doPlot):
    fs = pmu.fs
    f0 = pmu.f0
    RR = pmu.RR
    tclass = pmu.estimator.tclass
    
    Xm = 1
    if(tclass == 'p'):
        maxtve = 1
        maxfe  = 0.005
        maxrfe = 0.4
        kx = Xm * 0.01
    elif(tclass == 'm'):
        kx = Xm * 0.1
        maxtve = 10
        maxfe  = 0.005
        maxrfe = 1
        if(RR > 20):
            maxfe = 0.025
        else:
            maxfe = 0.005
    
    hrange = np.arange(2,((fs/2)/f0),1)
    t  = np.arange(0,5,1/fs)

    k = 0
    tve = np.empty(np.size(hrange))*np.nan
    fe  = np.empty(np.size(hrange))*np.nan
    rfe = np.empty(np.size(hrange))*np.nan
    for h in hrange:
        xa = Xm*np.cos(2*math.pi*(f0)*t) + Xm*kx*np.cos(2*math.pi*(f0*h)*t) 
        xb = Xm*np.cos(2*math.pi*(f0)*t - 2*math.pi/3) + Xm*kx*np.cos(2*math.pi*(f0*h)*t - 2*math.pi/3*h) 
        xc = Xm*np.cos(2*math.pi*(f0)*t + 2*math.pi/3) + Xm*kx*np.cos(2*math.pi*(f0*h)*t + 2*math.pi/3*h) 
        
        x = np.vstack((xa,xb,xc))
        
        T,X,F,DF = pmu.run(t,x)
        
        XrM = np.ones(np.size(T)) * Xm * math.sqrt(2)/2
        XrA = np.zeros(np.size(T))
        Xr  = XrM*np.cos(XrA) + 1j*XrM*np.sin(XrA);
        Fr  = np.ones(np.size(T)) * (f0);
        DFr = np.zeros(np.size(T));
        
        tve[k] = np.nanmax(abs(TVE(Xr,X[:,3])));
        fe [k] = np.nanmax(abs(FE(Fr,F)));
        rfe[k] = np.nanmax(abs(RFE(DFr,DF)));
        k += 1;
    
    results = dict()
    results['range'] = hrange*f0
    results['tve'] = tve
    results['fe'] = fe
    results['rfe'] = rfe
    results['limits'] = [maxtve,maxfe,maxrfe]
    
    if(doPlot):
        k = dict()
        k['harmonic'] = results
        k['name'] = '_nolegend_'
        plot([k])
        
    return results  
        
        
def dynRamp(pmu,pos,doPlot):
    fs    = pmu.fs
    f0    = pmu.f0
    RR    = pmu.RR
    tclass = pmu.estimator.tclass
    
    Xm = 1
    Framp = (pos==0)*-1 + (pos==1)
    if(tclass == 'p'):
        fmax = 2*np.sign(Framp)
        excl = 2
        maxtve = 1
        maxfe  = 0.01
        maxrfe = 0.4
    elif(tclass == 'm'):
        fmax = min([5,RR/5])*np.sign(Framp)
        excl = 7
        maxtve = 1
        maxfe  = 0.01
        maxrfe = 0.2
    
    rampTime = abs(2*fmax/Framp)
    tedge = 2
    
    t = np.arange(0,(rampTime+2*tedge),1/fs)
    t1 = t[np.where(t==0)[0][0]:np.where(t==tedge)[0][0]]
    t2 = t[np.where(t==tedge)[0][0]:np.where(t==tedge+rampTime)[0][0]]
    t3 = t[np.where(t==tedge+rampTime)[0][0]:]
    
    xa = np.concatenate((Xm*np.cos(2*math.pi*(f0-fmax)*t1), Xm*np.cos(2*math.pi*f0*t2 + math.pi*Framp*pow((t2 - fmax/Framp - tedge),2)), Xm*np.cos(2*math.pi*(f0+fmax)*t3)))
    xb = np.concatenate((Xm*np.cos(2*math.pi*(f0-fmax)*t1 - 2*math.pi/3), Xm*np.cos(2*math.pi*f0*t2 + math.pi*Framp*pow((t2 - fmax/Framp - tedge),2) - 2*math.pi/3), Xm*np.cos(2*math.pi*(f0+fmax)*t3 - 2*math.pi/3)))
    xc = np.concatenate((Xm*np.cos(2*math.pi*(f0-fmax)*t1 + 2*math.pi/3), Xm*np.cos(2*math.pi*f0*t2 + math.pi*Framp*pow((t2 - fmax/Framp - tedge),2) + 2*math.pi/3), Xm*np.cos(2*math.pi*(f0+fmax)*t3 + 2*math.pi/3)))
    x = np.vstack((xa,xb,xc))

    T,X,F,DF = pmu.run(t,x)
    
    ## Real Phasors
    T1 = T[np.where(T==0)[0][0]:np.where(T==tedge)[0][0]]
    T2 = T[np.where(T==tedge)[0][0]:np.where(T==tedge+rampTime)[0][0]]
    T3 = T[np.where(T==tedge+rampTime)[0][0]:]
    N1,N2,N3 = np.size(T1),np.size(T2),np.size(T3)
    
    XM  = np.ones(N1+N2+N3)*math.sqrt(2)/2*Xm
    XA  = np.concatenate((2*math.pi*(f0-fmax)*T1,math.pi*Framp*pow(T2-rampTime/2-tedge,2),2*math.pi*(f0+fmax)*T3))
    Xr  = XM*np.cos(XA) + 1j*XM*np.sin(XA)
    Fr  = np.concatenate((np.ones(N1)*(f0-fmax),np.arange(0,N2)*2*fmax/N2 + f0 - fmax,np.ones(N3)*(f0+fmax)))
    DFr = np.concatenate((np.zeros(N1),np.ones(N2)*Framp,np.zeros(N3)))
    
    # Exclusion Points
    TXS = np.arange(np.floor(np.where(T==tedge)[0][0]-(excl-1)-1),np.floor(np.where(T==tedge)[0][0]+(excl-1))+2).astype(int)
    TXE = np.arange(np.floor(np.where(T==tedge+rampTime)[0][0]-(excl-1)-1),np.floor(np.where(T==tedge+rampTime)[0][0]+(excl-1))+2).astype(int)
    # excluded measurements are 2 and 7 reports after and before the steady state section. I also exlude the points after for the same of analysis in steady state
    X [TXS] = np.nan
    X [TXE] = np.nan
    F [TXS] = np.nan
    F [TXE] = np.nan
    DF[TXS] = np.nan
    DF[TXE] = np.nan
    
    tve = abs(TVE(Xr,X[:,3]))
    fe  = abs(FE(Fr,F))
    rfe = abs(RFE(DFr,DF))
    
    results = dict()
    results['range'] = T
    results['tve'] = tve
    results['fe'] = fe
    results['rfe'] = rfe
    results['limits'] = [maxtve,maxfe,maxrfe]
    results['inputs'] = [pos]
    
    if(doPlot):
        k = dict()
        if(pos):
            k['pramp'] = results
        else:
            k['nramp'] = results
        k['name'] = '_nolegend_'
        plot([k])
        
    return results  
            
def dynOsc(pmu,amp,doPlot):
    fs    = pmu.fs
    f0    = pmu.f0
    RR    = pmu.RR
    tclass = pmu.estimator.tclass
    
    Xm = 1
    if(tclass == 'p'):
        Frt = min([RR/10,2])
        maxfe  = 0.03*Frt
        maxrfe = 0.18 * math.pi * Frt * Frt
    elif(tclass == 'm'):
        Frt = min([RR/5,5])
        maxfe  = 0.06*Frt
        maxrfe = 0.18 * math.pi * Frt * Frt
    maxtve = 3
    frange = np.arange(0.1,Frt+0.1,0.1)
    
    if(amp):
        kx,ka = 0.1,0.0
    else:
        kx,ka = 0.0,0.1
        
    k = 0
    tve = np.empty(np.size(frange))*np.nan
    fe  = np.empty(np.size(frange))*np.nan
    rfe = np.empty(np.size(frange))*np.nan
    t  = np.arange(0,5,1/fs)
    for fm in frange:
        xa = Xm*(1+kx*np.cos(2*math.pi*fm*t))*np.cos(2*math.pi*f0*t + ka*np.cos(2*math.pi*fm*t-math.pi))
        xb = Xm*(1+kx*np.cos(2*math.pi*fm*t))*np.cos(2*math.pi*f0*t - 2*math.pi/3 +  ka*np.cos(2*math.pi*fm*t-math.pi))
        xc = Xm*(1+kx*np.cos(2*math.pi*fm*t))*np.cos(2*math.pi*f0*t + 2*math.pi/3 + ka*np.cos(2*math.pi*fm*t-math.pi))
        x = np.vstack((xa,xb,xc))
        
        T,X,F,DF = pmu.run(t,x)
        
        N   = np.size(T)
        XM  = Xm/math.sqrt(2)*(1+kx*np.cos(2*math.pi*fm*T))
        XA  = ka*np.cos(2*math.pi*fm*T-math.pi)
        Xr  = XM*np.cos(XA) + 1j*XM*np.sin(XA)
        Fr  = f0 - ka*fm*np.sin(2*math.pi*fm*T-math.pi)
        DFr = -ka*2*math.pi*pow(fm,2)*np.cos(2*math.pi*fm*T-math.pi)
        
        tve[k] = np.nanmax(abs(TVE(Xr,X[:,3])));
        fe [k] = np.nanmax(abs(FE(Fr,F)));
        rfe[k] = np.nanmax(abs(RFE(DFr,DF)));
        k += 1;
    
    results = dict()
    results['range'] = frange
    results['tve'] = tve
    results['fe'] = fe
    results['rfe'] = rfe
    results['limits'] = [maxtve,maxfe,maxrfe]
    results['inputs'] = [amp]
    
    if(doPlot):
        k = dict()
        if(amp):
            k['amodulation'] = results
        else:
            k['pmodulation'] = results
        
        k['name'] = '_nolegend_'
        plot([k])
    return results  
        
            
def ssOOB(pmu,doPlot):
    fs    = pmu.fs
    f0    = pmu.f0
    RR    = pmu.RR
    tclass = pmu.estimator.tclass
    
    if(tclass == 'p'):
        maxtve, maxfe, maxrfe = -1,-1,-1
        fin = []
        fint = []
        tve = []
        fe= []
        rfe = []
    elif(tclass == 'm'):
        maxtve,maxfe,maxrfe = 1.3, 0.01, -1
        
        if(RR > 100):
            fin = np.arange(-5,5+1,5) + f0
        else:
            fin = np.array([-RR/2*0.1,0,RR/2*0.1]) + f0
            
        fint  = np.arange(10,2*f0+1)
        fintL = fint[:np.where(fint == f0-RR/2)[0][0]+1]
        fintR = fint[np.where(fint == f0+RR/2)[0][0]:]
        fint  = np.concatenate((fintL,fintR))
        
        Xm,kx = 1,0.1
        t = np.arange(0,5,1/fs)
        k = 0
        l = 0
        tve = np.empty((np.size(fin),np.size(fint)))*np.nan
        fe  = np.empty((np.size(fin),np.size(fint)))*np.nan
        rfe = np.empty((np.size(fin),np.size(fint)))*np.nan
        for j in fin:
            l = 0
            for i in fint:
                xa = Xm*np.cos(2*math.pi*j*t) + Xm*kx*np.cos(2*math.pi*i*t)
                xb = Xm*np.cos(2*math.pi*j*t - 2*math.pi/3) + Xm*kx*np.cos(2*math.pi*i*t - 2*math.pi/3)
                xc = Xm*np.cos(2*math.pi*j*t + 2*math.pi/3) + Xm*kx*np.cos(2*math.pi*i*t + 2*math.pi/3)
                x = np.vstack((xa,xb,xc))
                
                T,X,F,DF = pmu.run(t,x)
                
                N = np.size(T)
                XrM = np.ones(N)*Xm*math.sqrt(2)/2;
                XrA = 2*math.pi*(j-f0)*(T);
                Xr  = XrM*np.cos(XrA) + 1j*XrM*np.sin(XrA);
                Fr = np.ones(N) * (j);
                DFr = np.zeros(N);

                tve[k,l] = np.nanmax(abs(TVE(Xr,X[:,3])));
                fe [k,l] = np.nanmax(abs(FE(Fr,F)));
                rfe[k,l] = np.nanmax(abs(RFE(DFr,DF)));
                l += 1
            k += 1
        
    results = dict()
    results['range'] = [fin,fint]
    results['tve'] = tve
    results['fe'] = fe
    results['rfe'] = rfe
    results['limits'] = [maxtve,maxfe,maxrfe]
    
    if(doPlot):
        k = dict()
        k['oob'] = results
        
        k['name'] = '_nolegend_'
        plot([k])
        
    return results 
                
def dynStep(pmu,mag,pos,doPlot):
    fs    = pmu.fs
    f0    = pmu.f0
    RR    = pmu.RR
    tclass = pmu.estimator.tclass
    
    if(mag):
        if(pos):
            kx,ka = 0.1,0.0
        else:
            kx,ka = -0.1,0.0
    else:
        if(pos):
            kx,ka = 0.0,math.pi/18
        else:
            kx,ka = 0.0,-math.pi/18
    Xm = 1
    
    step = 0.1
    maxtve = 1
    maxfe  = 0.005
    
    if(tclass == 'p'):
        delayTime  = 1/(4*RR)
        maxshoot   = 0.05
        respTime   = 2/f0
        frespTime  = 4.5/f0
        dfrespTime = 6/f0
        maxrfe     = 0.4
    elif(tclass == 'm'):
        delayTime  = 1/(4*RR)
        maxshoot   = 0.1
        respTime   = max([7/f0,7/RR])
        frespTime  = max([14/f0,14/RR])
        dfrespTime = max([14/f0,14/RR])
        maxrfe     = 0.1
        
    t = np.arange(0,4,1/fs)
    N = np.size(t)
    u = np.concatenate((np.zeros(int(N/2)),np.ones(int(N/2))))
    
    xa = Xm * (1 + kx*u) * np.cos(2*math.pi*f0*t + ka*u)
    xb = Xm * (1 + kx*u) * np.cos(2*math.pi*f0*t + ka*u - 2*math.pi/3)
    xc = Xm * (1 + kx*u) * np.cos(2*math.pi*f0*t + ka*u + 2*math.pi/3)
    x = np.vstack((xa,xb,xc))
    
    T,X,F,DF = pmu.run(t,x)
    
    N = np.size(T)
    U = u[0:np.size(t):int(fs/RR)]
    
    if(mag):
        XrM = math.sqrt(2)/2*Xm*(1 + kx*U)
        XrA = np.zeros(N)
    else:
        XrM = math.sqrt(2)/2*Xm
        XrA = U*ka
        
    Xr  = XrM*np.cos(XrA) + 1j*XrM*np.sin(XrA)
    Fr  = np.ones(N)*f0
    DFr = np.zeros(N)
    T = T - 2 # center the step
    
    # TVE
    tve = TVE(Xr,X[:,3]) 
    fe = FE(Fr,F)
    rfe = RFE(DFr,DF)
    
    results = dict()
    results['range']  = T
    results['tve']    = tve
    results['fe']     = fe
    results['rfe']    = rfe
    results['limits'] = [maxtve,maxfe,maxrfe,respTime,frespTime,dfrespTime,maxshoot]
    results['inputs'] = [mag,pos,Xm,kx,ka]
    
    if(mag):
        results['X']      = np.array(abs(X[:,3]))
    else:
        results['X']      = np.array(np.angle(X[:,3]))
        
    if(doPlot):
        k = dict()
        if(mag):
            if(pos):
                k['pmagstep'] = results
            else:
                k['nmagstep'] = results
        else:
            if(pos):
                k['pphasestep'] = results
            else:
                k['nphasestep'] = results
        
        k['name'] = '_nolegend_'
        plot([k])
        
    return results 
               
               
               
def plot(results):
    N = len(results)
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if(hasany(results,'frequency')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('frequency' in results[i]):
                ssFreqPlot(f,axs,results[i]['frequency'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'magnitude')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('magnitude' in results[i]):
                ssMagPlot(f,axs,results[i]['magnitude'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)

    if(hasany(results,'harmonic')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('harmonic' in results[i]):
                ssHarmPlot(f,axs,results[i]['harmonic'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'pramp')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('pramp' in results[i]):
                dynRampPlot(f,axs,results[i]['pramp'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'nramp')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('nramp' in results[i]):
                dynRampPlot(f,axs,results[i]['nramp'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'amodulation')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('amodulation' in results[i]):
                dynOscPlot(f,axs,results[i]['amodulation'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'pmodulation')):
        f, (axs) = plt.subplots(1, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('pmodulation' in results[i]):
                dynOscPlot(f,axs,results[i]['pmodulation'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'oob')):
        f, (ax) = plt.subplots(3, 3,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('oob' in results[i]):
                plotted = ssOOBPlot(f,ax,results[i]['oob'],cmap[i])
                if(plotted):
                    labels.append(results[i]['name'])
        f.legend(labels)
    
    if(hasany(results,'pmagstep')):
        f, (ax) = plt.subplots(2, 2,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('pmagstep' in results[i]):
                dynStepPlot(f,ax,results[i]['pmagstep'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)

    if(hasany(results,'nmagstep')):
        f, (ax) = plt.subplots(2, 2,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('nmagstep' in results[i]):
                dynStepPlot(f,ax,results[i]['nmagstep'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)

    if(hasany(results,'pphasestep')):
        f, (ax) = plt.subplots(2, 2,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('pphasestep' in results[i]):
                dynStepPlot(f,ax,results[i]['pphasestep'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)

    if(hasany(results,'nphasestep')):
        f, (ax) = plt.subplots(2, 2,figsize=(18, 6))
        labels = list()
        for i in range(N):
            if('nphasestep' in results[i]):
                dynStepPlot(f,ax,results[i]['nphasestep'],cmap[i])
                labels.append(results[i]['name'])
        f.legend(labels)
    


def hasany(results,key):
    hasit = 0
    for i in range(len(results)):
        if(key in results[i]):
            hasit = 1
    return hasit
    
def ssFreqPlot(f,axs,q,cmap):
    frange = q['range']
    tve    = q['tve']
    fe     = q['fe']
    rfe    = q['rfe']
    maxtve = q['limits'][0]
    maxfe  = q['limits'][1]
    maxrfe = q['limits'][2]
    
    axs[0].plot(frange,tve,color=cmap)
    axs[0].hlines(maxtve,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[0].set(xlabel='Frequency (Hz)',ylabel='TVE (%)')
    axs[1].plot(frange,fe,color=cmap)
    axs[1].hlines(maxfe ,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[1].set(xlabel='Frequency (Hz)',ylabel='FE (Hz)')
    axs[2].plot(frange,rfe,color=cmap)
    axs[2].hlines(maxrfe,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[2].set(xlabel='Frequency (Hz)',ylabel='RFE (Hz/s)')
    f.suptitle('Steady State Frequency Test')

def ssMagPlot(f,axs,q,cmap):
    frange = q['range']
    tve    = q['tve']
    fe     = q['fe']
    rfe    = q['rfe']
    maxtve = q['limits'][0]
    maxfe  = q['limits'][1]
    maxrfe = q['limits'][2]
    
    axs[0].plot(frange,tve,color=cmap)
    axs[0].hlines(maxtve,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[0].set(xlabel='Magnitude (p.u.)',ylabel='TVE (%)')
    axs[1].plot(frange,fe,color=cmap)
    axs[1].hlines(maxfe ,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[1].set(xlabel='Magnitude (p.u.)',ylabel='FE (Hz)')
    axs[2].plot(frange,rfe,color=cmap)
    axs[2].hlines(maxrfe,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[2].set(xlabel='Magnitude (p.u.)',ylabel='RFE (Hz/s)')
    f.suptitle('Steady State Magnitude Test')
    
def ssHarmPlot(f,axs,q,cmap):
    frange = q['range']
    tve    = q['tve']
    fe     = q['fe']
    rfe    = q['rfe']
    maxtve = q['limits'][0]
    maxfe  = q['limits'][1]
    maxrfe = q['limits'][2]
    
    axs[0].plot(frange,tve,color=cmap)
    axs[0].hlines(maxtve,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[0].set(xlabel='Harmonic Frequency (Hz)',ylabel='TVE (%)')
    axs[1].plot(frange,fe,color=cmap)
    axs[1].hlines(maxfe ,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[1].set(xlabel='Harmonic Frequency (Hz)',ylabel='FE (Hz)')
    axs[2].plot(frange,rfe,color=cmap)
    axs[2].hlines(maxrfe,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[2].set(xlabel='Harmonic Frequency (Hz)',ylabel='RFE (Hz/s)')
    f.suptitle('Steady State Harmonic Test')
    

def dynRampPlot(f,axs,q,cmap):
    T      = q['range']
    tve    = q['tve']
    fe     = q['fe']
    rfe    = q['rfe']
    maxtve = q['limits'][0]
    maxfe  = q['limits'][1]
    maxrfe = q['limits'][2]
    pos    = q['inputs'][0]
    
    axs[0].plot(T,tve,color=cmap)
    axs[0].hlines(maxtve ,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[0].set(xlabel='Time (s)',ylabel='TVE (%)')
    axs[1].plot(T,fe,color=cmap)
    axs[1].hlines(maxfe ,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[1].set(xlabel='Time (s)',ylabel='FE (Hz)')
    axs[2].plot(T,rfe,color=cmap)
    axs[2].hlines(maxrfe ,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[2].set(xlabel='Time (s)',ylabel='RFE (Hz/s)')
    if(pos):
        f.suptitle('Dynamic Positive Ramp Test')
    else:
        f.suptitle('Dynamic Negative Ramp Test')
        
def dynOscPlot(f,axs,q,cmap):
    frange = q['range']
    tve    = q['tve']
    fe     = q['fe']
    rfe    = q['rfe']
    maxtve = q['limits'][0]
    maxfe  = q['limits'][1]
    maxrfe = q['limits'][2]
    amp    = q['inputs'][0]
    
    axs[0].plot(frange,tve,color=cmap)
    axs[0].hlines(maxtve,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[0].set(xlabel='Frequency (Hz)',ylabel='TVE (%)')
    axs[1].plot(frange,fe,color=cmap)
    axs[1].hlines(maxfe ,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[1].set(xlabel='Frequency (Hz)',ylabel='FE (Hz)')
    axs[2].plot(frange,rfe,color=cmap)
    axs[2].hlines(maxrfe,frange[0],frange[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    axs[2].set(xlabel='Frequency (Hz)',ylabel='RFE (Hz/s)')
    if(amp):
        f.suptitle('Dynamic Amplitude Modulation Test')
    else:
        f.suptitle('Dynamic Phase Modulation Test')
        
 
def ssOOBPlot(f,ax,q,cmap):
    fin = q['range'][0]
    fint = q['range'][1]
    tve    = q['tve']
    fe     = q['fe']
    rfe    = q['rfe']
    maxtve = q['limits'][0]
    maxfe  = q['limits'][1]
    maxrfe = q['limits'][2]
    """Split Interference Freq so there isnt a line connecting the left and right bounds"""
    if(len(fin) > 0):
        fintL = fint[:np.where(np.diff(fint) > 1)[0][0] + 1]
        fintR = fint[np.where(np.diff(fint) > 1)[0][0] + 1:] 
        for i in range(3):
            ax[i][0].plot(fintL,tve[i][:len(fintL)],color=cmap)
            ax[i][0].plot(fintR,tve[i][len(fintL):],color=cmap)
            ax[i][0].hlines(maxtve,fint[0],fint[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
            ax[i][0].set(xlabel='Interference (Hz)',ylabel='TVE (%) @'+str(fin[i])+'Hz')
        
            ax[i][1].plot(fintL,fe[i][:len(fintL)],color=cmap)
            ax[i][1].plot(fintR,fe[i][len(fintL):],color=cmap)
            ax[i][1].hlines(maxfe,fint[0],fint[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
            ax[i][1].set(xlabel='Interference (Hz)',ylabel='FE (Hz) @'+str(fin[i])+'Hz')
        
            ax[i][2].plot(fintL,rfe[i][:len(fintL)],color=cmap)
            ax[i][2].plot(fintR,rfe[i][len(fintL):],color=cmap)
            ax[i][2].set(xlabel='Interference (Hz)',ylabel='RFE (Hz/s) @'+str(fin[i])+'Hz')
                
            f.suptitle('Out-of-Bounds Interference Test')
        return 1
    else:
        return 0

def dynStepPlot(f,ax,q,cmap):
    T          = np.array(q['range'])
    tve        = np.array(q['tve'])
    fe         = np.array(q['fe'])
    rfe        = np.array(q['rfe'])
    maxtve     = q['limits'][0]
    maxfe      = q['limits'][1]
    maxrfe     = q['limits'][2]
    respTime   = q['limits'][3]
    frespTime  = q['limits'][4]
    dfrespTime = q['limits'][5]
    maxshoot   = q['limits'][6]
    mag        = q['inputs'][0]
    pos        = q['inputs'][1]
    Xm         = q['inputs'][2]
    kx         = q['inputs'][3]
    ka         = q['inputs'][4]
    Xp         = np.array(q['X'])
    
    N = np.size(T)
    # TVE
    ax[0,0].plot(T,tve,color=cmap)
    ax[0,0].hlines(maxtve,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    ax[0,0].vlines(T[np.where(tve > maxtve)[0][0]],-100,100,colors=cmap,linestyle='dashed', label='_nolegend_')
    ax[0,0].vlines(T[np.where(tve > maxtve)[0][0]]+respTime,-100,100,colors=cmap,linestyle='dashed', label='_nolegend_')
    ax[0,0].set(ylim=(0,2),xlim=(-0.3,0.6),xlabel='Time (s)',ylabel='TVE (%)')
       
    # FE
    ax[0,1].plot(T,fe,color=cmap)
    ax[0,1].hlines(maxfe,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    ax[0,1].hlines(-maxfe,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    if(np.size(np.where(abs(fe) > maxfe)[0]) > 0):
        ax[0,1].vlines(T[min(np.where(abs(fe) > maxfe)[0])],-100,100,colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[0,1].vlines(T[min(np.where(abs(fe) > maxfe)[0])]+frespTime,-100,100,colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[0,1].set(ylim=(-0.02,0.02),xlim=(-0.3,0.6),xlabel='Time (s)',ylabel='FE (Hz)')
       
    # RFE
    ax[1,0].plot(T,rfe,color=cmap)
    ax[1,0].hlines(maxrfe,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    ax[1,0].hlines(-maxrfe,T[0],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
    if(np.size(np.where(abs(rfe) > maxrfe)[0]) > 0):
        ax[1,0].vlines(T[min(np.where(abs(rfe) > maxfe)[0])],-100,100,colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,0].vlines(T[min(np.where(abs(rfe) > maxfe)[0])]+dfrespTime,-100,100,colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,0].set(ylim=(-0.6,0.6),xlim=(-0.3,1.0),xlabel='Time (s)',ylabel='RFE (Hz/s)')
       
    if(mag):
        B = math.sqrt(2)/2
        # Over/under shoot
        ax[1,1].plot(T,(Xp),color=cmap)
        ax[1,1].hlines(B*(Xm + Xm*kx + Xm*kx*maxshoot),T[int(N/2)],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].hlines(B*(Xm + Xm*kx - Xm*kx*maxshoot),T[int(N/2)],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].hlines(B*(Xm + Xm*kx*maxshoot),T[0],T[int(N/2)],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].hlines(B*(Xm - Xm*kx*maxshoot),T[0],T[int(N/2)],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].vlines(0,-100,100,colors=cmap, label='_nolegend_')
        if(pos):
            ax[1,1].set(ylim=(B*Xm-0.02*Xm,B*Xm+0.1*Xm),xlim=(-0.3,0.3),xlabel='Time (s)',ylabel='Magnitude')
            f.suptitle('Positive Magnitude Step Test')
        else:
            ax[1,1].set(ylim=(B*Xm-0.1*Xm,B*Xm+0.02*Xm),xlim=(-0.3,0.3),xlabel='Time (s)',ylabel='Magnitude')
            f.suptitle('Negative Magnitude Step Test')
    else:
        B = 180/math.pi
        # Over/under shoot
        ax[1,1].plot(T,(Xp)*B)
        ax[1,1].hlines(B*ka*(1+maxshoot),T[int(N/2)],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].hlines(B*ka*(1-maxshoot),T[int(N/2)],T[-1],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].hlines(B*ka*(+maxshoot),T[0],T[int(N/2)],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].hlines(B*ka*(-maxshoot),T[0],T[int(N/2)],colors=cmap,linestyle='dashed', label='_nolegend_')
        ax[1,1].vlines(0,-100,100,colors=cmap, label='_nolegend_')
        ax[1,1].set(ylim=(((pos==0)*B*ka)-5,((pos==1)*B*ka)+5),xlim=(-0.3,0.3),xlabel='Time (s)',ylabel='Phase Angle (degrees)')
        if(pos):
            f.suptitle('Positive Phase Step Test')
        else:
            f.suptitle('Negative Phase Step Test')
    


import json
def writeJson(data,name):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj.tolist())
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    with open(name+'.json', "w") as outfile:
        json.dump(data, outfile, cls=NpEncoder)
def readJson(name):
    with open(name+'.json', 'r') as f:
      data = json.load(f)
    return data

def TVE(X, Xref):
    Xt = X[~np.isnan(X)]; Xtr = Xref[~np.isnan(X)]
    PE = np.arctan2(np.real(Xt),np.imag(Xt)) - np.arctan2(np.real(Xtr),np.imag(Xtr));
    ME = (np.sqrt(pow(np.real(Xt),2) + pow(np.imag(Xt),2)) - np.sqrt(pow(np.real(Xtr),2) + pow(np.imag(Xtr),2))) /  np.sqrt(pow(np.real(Xtr),2) + pow(np.imag(Xtr),2))
    return np.sqrt(2*(1 + ME)*(1 - np.cos(PE)) + pow(ME,2)) * 100

def FE(F,Fref):
    return (F[~np.isnan(F)] - Fref[~np.isnan(F)])

def RFE(F,Fref):
    return (F[~np.isnan(F)] - Fref[~np.isnan(F)])