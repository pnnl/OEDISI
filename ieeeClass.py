# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:39:25 2022

Implementation of IEEE C37.118 (2018) Annex D standard PMU algorithms
Built to interface with "pmuClass"

- takes inputs of sampling rate (fs), nominal freq (f0), report rate (RR), and 
class (p or m)
- All filters described in Annex D are implemented (f0=[50,60] and all combinations of RR)

@author: Dylan Tarter
"""

import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

class ieeeClass:
    def __init__(self, fs, f0, RR, tclass, name = 'None'):
        self._fs = fs
        self._f0 = f0
        self._RR = RR
        self._tclass = tclass.lower() #force to lower case for later checks
        
        if(name == 'None'):
            self._name = 'IEEE '+self.tclass.upper()+'-Class'
        else:
            self._name = name
            
        """Build the filters before-hand to reduce computations on run"""
        if(tclass == 'p'):
            """From Annex D.6, equation D.5"""
            N = (fs/f0 - 1) * 2
            k = np.arange(-N/2,N/2+1)
            w = 1-2/(N+2)*abs(k)
        elif(self._tclass == 'm'):
            """From Annex D.7, Table D.1"""
            if(self._f0 == 50):
                if(self._RR == 10):
                    Ffr, N = 1.779, 806
                elif(self._RR == 25):
                    Ffr, N = 4.355, 338
                elif(self._RR == 50):
                    Ffr, N = 7.75, 142
                elif(self._RR == 100):
                    Ffr, N = 14.1, 66
                else:
                   print('[PMU/IEEE] M Class filter at 50Hz nominal only supports reports at (10,25,50,100)') 
            elif(self._f0 == 60):
                if(self._RR == 10):
                    Ffr, N = 1.78, 968
                elif(self._RR == 12):
                    Ffr, N = 2.125, 816
                elif(self._RR == 15):
                    Ffr, N = 2.64, 662
                elif(self._RR == 20):
                    Ffr, N = 3.5, 502
                elif(self._RR == 30):
                    Ffr, N = 5.02, 306
                elif(self._RR == 60):
                    Ffr, N = 8.19, 164
                elif(self._RR == 120):
                    Ffr, N = 16.25, 70    
                else:
                   print('[PMU/IEEE] M Class filter at 60Hz nominal only supports reports at (10,12,15,20,30,60,120)') 
            else:
                print('[PMU/IEEE] M Class filter only supports nominal frequencies of 60 or 50Hz')
                
            """From Annex D.7, equation D.7"""
            k = np.arange(-N/2,N/2+1)
            h = np.hamming(N+1) #hamming filter of order N+1
            w = np.sin(2*math.pi*(2*Ffr)/self._fs*k) / (2*math.pi*(2*Ffr)/self._fs*k) * h
            #w  = np.sinc(2*math.pi*(2*Ffr)/self._fs*k) * h
            w[int(N/2)] = 1
        else:
            print('[PMU/IEEE] Filter class not recognized. Try p or m.')
        
        """From D.2, Equation D.1 and D.2. Just coefficients, no summation accross x"""
        w0 = 2*math.pi*f0
        G = sum(w) # D.2
        E = np.exp(-1j*k/fs*w0) # DFT
        # filter coefficients to be used in the summation
        self._filter = math.sqrt(2)/G * np.exp(1j*N/2/fs*w0) * w * np.flip(E)
     
    def __name__(self):
        return self._name
    
    @property
    def fs(self):
        """Return the pmu's assumed sampling rate."""
        return self._fs
    @property
    def f0(self):
        """Return the pmu's assumed nominal frequency."""
        return self._f0
    @property
    def RR(self):
        """Return the pmu's report rate."""
        return self._RR
    @property
    def tclass(self):
        """Return the pmu's class type (p or m)."""
        return self._tclass   
    
    @property
    def plotFilter(self):
        
        """A debugger function to plot the filter's magnitude and angle."""
        ax1 = plt.subplot()
        l1, = ax1.plot(abs(self._filter), color='red')
        ax2 = ax1.twinx()
        l2, = ax2.plot(np.angle(self._filter), color='orange')
        plt.legend([l1, l2], ["Filter Magnitude", "Filter Angle/Phase"])
        plt.show()
    
    def run(self, t, x):
        """
        Called to estimate phasors of timestamps t and measurements x.
        Will output timestamps, phasors, frequency, and ROCOF at the report rate
        
        If x is 1D, it will output just a phasor, with NaN for frequency and ROCOF
        If x has 3 measurements it will assume 3 phase, and give 3 phasors + the positive sequence
        as well as frequency and ROCOF estimations
        If x has 3+3 measurements it will output 6 single phase phasors and 2 positive phase phasors
        in the order or [Xa,Xb,Xc,Xp,Ya,Yb,Yc,Yp]. And will only use X to calculate frequency and ROCOF.
        """
        if(x.size == x.shape[0]): # if .size and .shape[0] are same it is 1D
            X = self.__estimate(t,x,self._fs,self._f0,self._filter)
            F  = np.empty(np.size(x,0), dtype=complex) * np.nan # cannot calculate F w/o 3 Phases
            DF  = np.empty(np.size(x,0), dtype=complex) * np.nan# cannot calculat ROCOF w/o 3 Phases
        else:
            # builds an empty NaN array shaped like the input but with additional positive seq columns
            X  = np.empty((np.size(x,0),np.size(x,1)+math.floor(np.size(x,1)/3)), dtype=complex) * np.nan
            
            j = 0 # iterator for the output columns
            for i in range(np.size(x,1)): # for each input column write an output
                X[:,j] = self.__estimate(t,x[:,i],self._fs,self._f0,self._filter)
                j += 1;
                if((i+1) % 3 == 0): # for every 3 measurements, add a positive sequence calculation
                    X[:,j]  = ((0.5 - 1j*math.sqrt(3)/2)*X[:,j-3] + (0.5 + 1j*math.sqrt(3)/2)*X[:,j-2] - X[:,j-1]) / (1.5 - 1j*3*math.sqrt(3)/2)
                    j += 1 # make sure to skip past where the pos seq. is stored.
            
            # only use the first 3phase system to calculate frequency. others are assumed to be same frequency
            """Annex D.4 Equations D.3 and D.4"""
            F   = signal.lfilter([1, 0,-1],1,self.__unwrap(np.angle(X[:,3])))/(4*math.pi*1/self._fs) + self._f0
            DF  = signal.lfilter([1,-2, 1],1,self.__unwrap(np.angle(X[:,3])))/(2*math.pi/self._fs/self._fs)
            
            # account for group delay of the filters
            GrpDel = 1
            F  = np.concatenate((F [GrpDel:],np.empty(GrpDel)*np.nan))
            DF = np.concatenate((DF[GrpDel:],np.empty(GrpDel)*np.nan))
            
            """If P class, fix the magnitude using Annex D.6, Equation D.6"""
            if(self._tclass == 'p'):
                for i in range(np.size(X,1)):
                    XM = abs(X[:,i]);
                    XM = XM / np.sin(math.pi*(self._f0 + 1.625*(self._f0-F))/(2*self._f0))
                    X[:,i] = XM*np.cos(np.angle(X[:,i])) + 1j*XM*np.sin(np.angle(X[:,i]))
            
        return X,F,DF
    
    @staticmethod
    def __estimate(t,x,fs,f0,b):
        """From D.2, Equation D.1 and D.2. convolution with X"""
        w0  = 2*math.pi*f0;
        N = np.size(b) - 1;
        X = signal.lfilter(b,1,x); # this part does the summation/convolution
        
        X[:N] = np.NaN; # filter start-up
        XM = abs(X);
        XA = ((np.angle(X) - t*w0) + math.pi) % (2*math.pi) - math.pi #wrapToPi
        
        """From D.3, group delay"""
        GrpDel = int(N/2); #GroupDelay = N/2
        XM = np.concatenate((XM[GrpDel:],np.empty(GrpDel)*np.nan))
        XA = np.concatenate((XA[GrpDel:],np.empty(GrpDel)*np.nan))
    
        return XM*np.cos(XA) + 1j*XM*np.sin(XA) # angle to complex
    
    @staticmethod 
    def __unwrap(array): # unwrap from pi, ignoring NaN's
        array[~np.isnan(array)] = np.unwrap(array[~np.isnan(array)])
        return array