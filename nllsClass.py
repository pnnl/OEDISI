# -*- coding: utf-8 -*-
"""
Modified on Wed Dec 14 2022 by Kaustav Chatterjee
Created on Tue Aug  9 12:53:47 2022

Implementation of the Non-Linear Least Squares algorithm,
developed by James Follum, Kaustav Chatterjee, and Dylan Tarter.
Built to interface with "pmuClass"

This version does not have the periodogram approach. 
This version uses heuristic grid search for frequency estimation. 
For periodogram approach, refer to version nllsClass_v2.py

- takes inputs of sampling rate (fs), nominal freq (f0), report rate (RR), and 
class (p or m)
- pmu.run(t,x), t is ignored. x must be a window of data to generate 1 sample for.
- Outputs, phasor, frequency, ROCOF for given sample
- If 3 phase is provided, will generate the single phasors followed by a positive sequence phasor.

@authors: Dylan Tarter and Kaustav Chatterjee (PNNL)
"""

import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

class nllsClass:
    def __init__(self, fs, f0, RR, tclass, cycles, doPreFilter = 1, doROCOF = 1, name = 'None'):
        self._fs = fs
        self._f0 = f0
        self._RR = RR
        self._tclass = tclass.lower() #force to lower case for later checks
        self._cycles = cycles # number of cycles a window has (usually integer)
        self._doPreFilter = doPreFilter
        self._doROCOF     = doROCOF
        
        """Default Name"""
        if(name == 'None'):
            self._name = 'NLLS '+self.tclass.upper()+'-Class, '+ str(self.cycles)+' Cycles'
            if(not self._doPreFilter):
                name = name + ' (un-filtered)'
        else:
            self._name = name
            
        """Calculate preFilter coefficients before-hand"""
        if(self._tclass == 'p'):
            n = 31
            fo = np.array([0,0.1292,0.2708,1.0000])/2
            ao = np.array([1,0])
            w  = np.array([617.8376,1])
        elif(self._tclass == 'm'):
            n = 115
            fo = np.array([0,0.0625,0.1146,0.1354,0.1875,1.0000])/2
            ao = np.array([0,1,0])
            w  = np.array([1,34.7436,1])
        else:
            print('[PMU/NLLS] ERROR! Unrecognized class. Try p or m.')
        
        """Builds a Parks-McClellan for filtering out harmonics @ 960Hz, needs Matlab's firpmord for other fs"""
        self._preFilterCoef = signal.remez(n, fo, ao, weight=w)
     
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
    def cycles(self):
        """Return the pmu's cycle count per window"""
        return self._cycles
            
    def run(self,t,x):
        if(self._tclass == 'p'):
            fl,fh,e = 57.9, 62.1, 0.001 # define frequency limits on NLLS
        elif(self._tclass == 'm'):
            fl,fh,e = 54.9, 65.1, 0.001 # define frequency limits on NLLS
        else:
            print('[PMU/NLLS] ERROR! Unrecognized class. Try p or m.')
            
        # NLLS requires a fixed time vector where 0 is the report.
        # since the report is at center of the window, we center the fixed time
        # vector to -N/2 .. 0 .. N/2
        t  = np.arange(-(np.size(t)-1)/2,(np.size(t)-1)/2+1)/self._fs 
        
        
        """Only difference between 1 and 3Phase estimation is the estimation equation used
           because we can simplify the equations."""
        if(x.size == x.shape[0]): # if its a 1D array...
            if(self._doROCOF):
                FPP = self.__estimateFreq1Phase(t[ :-4],x[ :-4],fl,fh,e)
                FP  = self.__estimateFreq1Phase(t[1:-3],x[1:-3],fl,fh,e)
                F   = self.__estimateFreq1Phase(t[2:-2],x[2:-2],fl,fh,e)
                FN  = self.__estimateFreq1Phase(t[3:-1],x[3:-1],fl,fh,e)
                FNN = self.__estimateFreq1Phase(t[4:  ],x[4:  ],fl,fh,e)
                X  = self.__estimatePhasor1Phase(t,x,F)
            else:
                F   = self.__estimateFreq1Phase(t,x,fl,fh,e)
            X  = self.__estimatePhasor1Phase(t,x,F)
        else:                     # if its > 1D
            """Takes window and calculated 5 frequency estimations
            - each window is the same size just shifted 1 point forward
            - we tested 3 estimates but ROCOF could not pass al tests, so 5 are used.
            """
            if(self._doROCOF):
                FPP = self.__estimateFreq(t[ :-4],x[ :-4,:3],fl,fh,e)
                FP  = self.__estimateFreq(t[1:-3],x[1:-3,:3],fl,fh,e)
                F   = self.__estimateFreq(t[2:-2],x[2:-2,:3],fl,fh,e)
                FN  = self.__estimateFreq(t[3:-1],x[3:-1,:3],fl,fh,e)
                FNN = self.__estimateFreq(t[4:  ],x[4:  ,:3],fl,fh,e)
            else:
                F   = self.__estimateFreq(t,x[:,:3],fl,fh,e)
            # estimates the phasor of the 3phase element provided the frequency.
            X3  = self.__estimatePhasor(t,x[:,:3],F)
            # calculate the positive sequence phasor
            X  = ((0.5 - 1j*math.sqrt(3)/2)*X3[0] + (0.5 + 1j*math.sqrt(3)/2)*X3[1] - X3[2]) / (1.5 - 1j*3*math.sqrt(3)/2)
            # put 3phase and pos phasors together
            X = np.concatenate((X3,np.array([X])))
            
            # if there are more phasors, aka 3 current phasors, do more
            if(np.size(x,1) > 3):
                Xe3 = self.__estimatePhasor(t,x[:,3:],F) # calculate the phasors from same frequency
                # calcualte pos sequence of phasors
                Xe  = ((0.5 - 1j*math.sqrt(3)/2)*Xe3[0] + (0.5 + 1j*math.sqrt(3)/2)*Xe3[1] - Xe3[2]) / (1.5 - 1j*3*math.sqrt(3)/2)
                # add new phasors and pos seq. to the array of output phasors
                X = np.concatenate((X,Xe3,np.array([Xe])))
            
                
        if(self._doROCOF):
            F  = (FNN + FN + F + FP + FPP) / 5 # average of frequency estimates
            DF = (F-FPP)*self._fs/2  # backward ROCOF
            DF1 = (FN-FP)*self._fs/2 # center ROCOF
            DF2 = (FNN-F)*self._fs/2 # forward ROCOF
            DF = (DF + DF1 + DF2) / 3# average ROCOF
        else:
            DF = np.nan
            
        return X,F,DF
            
    def preFilter(self,t,x):
        """
        The tests NLLS cannot pass well is OOB and Harmonic Test. To fix this,
        a Band Pass Filter is simulated to happen before the estimation. This function
        applies the band-pass filter as a Parks-McClellan Filter
        """
        if(self._doPreFilter):
            L = self._preFilterCoef.shape[0]
            # if you do not filter on seperate channels it will break big time!
            if(x.size == x.shape[0]): #1 channel filtering
                x = signal.lfilter(self._preFilterCoef,1,x)
                x = np.concatenate((np.empty(L)*np.nan,x[L:]))
            else: # filter for all channels
                num_cols = np.size(x,1)
                for i in range(num_cols): # for each channel, filter it and apply start-up delay
                    x[:,i] = signal.lfilter(self._preFilterCoef,1,x[:,i])
                    x[:,i] = np.concatenate((np.empty((L))*np.nan,x[L:,i])) # startup delay
            gd = (L-1)/2 #overall group delay of filter
            x = x[int(gd):]

        return t,x

    
    @staticmethod
    def __estimateFreq(t,x,fl,fh,e):
        """
        The actual NLLS frequency estimation algorithm.
            - This is the Non-Linear part of the estimation, so it requires a 
            - cost function and some sort of gradient descent type algorithm as used here.
        """
        c = 0
        while(1):
            c += 1
            if c > 100:
                print('[NLLS] Error! Too many Itterations')
                break
            
            # move the high and low frequency
            ml = fl + (fh-fl)/3
            mh = fh - (fh-fl)/3
            
            # calculate the guess / cost function of lower frequency bound
            H = (np.vstack((np.cos(2*math.pi*ml*t),np.sin(2*math.pi*ml*t))))
            HH = np.matmul(np.linalg.pinv(H),H)
            Jl = np.matmul(np.matmul(np.transpose(x[:,0]),HH),x[:,0]) + np.matmul(np.matmul(np.transpose(x[:,1]),HH),x[:,1]) +      np.matmul(np.matmul(np.transpose(x[:,2]),HH),x[:,2])
            
            # calculate the guess / cost function of higher frequency bound
            H = (np.vstack((np.cos(2*math.pi*mh*t),np.sin(2*math.pi*mh*t))))
            HH = np.matmul(np.linalg.pinv(H),H)
            Jh = np.matmul(np.matmul(np.transpose(x[:,0]),HH),x[:,0]) + np.matmul(np.matmul(np.transpose(x[:,1]),HH),x[:,1]) + np.matmul(np.matmul(np.transpose(x[:,2]),HH),x[:,2])
        
            # if the cost function of lower is better than upper, adjust the lower
            # if cost of higher freq is better than lower, adjust higher
            # if they are equal, stop.
            if Jl < Jh:
                fl = ml
            elif Jh < Jl:
                fh = mh
            else:
                f0hat = (fl + fh) / 2
                break
            
            # if the estimated frequencies are close  to eachother, stop.
            if (fh - fl) < e:
                f0hat = (fl + fh) / 2
                break
            
        return f0hat

    @staticmethod
    def __estimatePhasor(t,x,f0hat):
        """
        NLLS Estimation of Phasors given frequency. 
            - This is technically a Linear Estimation.
        """
        H = (np.vstack((np.cos(2*math.pi*f0hat*t),np.sin(2*math.pi*f0hat*t))))
        Bhat = np.matmul(np.transpose(np.linalg.pinv(H)),x) # calculate the Beta vector which is [cos(theta),sin(theta)]
        Ahat = np.array(list(map(math.sqrt,sum(pow(Bhat,2)))))/math.sqrt(2) # calculate magnitude
        Phat = np.arctan2(-Bhat[1,:],Bhat[0,:]) # calculate theta
        return Ahat*np.cos(Phat) + 1j*Ahat*np.sin(Phat) # return complex phasor
    
    @staticmethod
    def __estimateFreq1Phase(t,x,fl,fh,e):
        """Same as other function but Jl and Jh have a simpler calculation"""
        c = 0
        while(1):
            c += 1
            if c > 100:
                print('[NLLS] Error! Too many Itterations')
                break
                
            ml = fl + (fh-fl)/3
            mh = fh - (fh-fl)/3
            
            H = (np.vstack((np.cos(2*math.pi*ml*t),np.sin(2*math.pi*ml*t))))
            HH = np.matmul(np.linalg.pinv(H),H)
            Jl = np.matmul(np.matmul(np.transpose(x),HH),x)
            
            H = (np.vstack((np.cos(2*math.pi*mh*t),np.sin(2*math.pi*mh*t))))
            HH = np.matmul(np.linalg.pinv(H),H)
            Jh = np.matmul(np.matmul(np.transpose(x),HH),x)
        
            if Jl < Jh:
                fl = ml
            elif Jh < Jl:
                fh = mh
            else:
                f0hat = (fl + fh) / 2
                break
                    
            if (fh - fl) < e:
                f0hat = (fl + fh) / 2
                break
            
        return f0hat
    
    @staticmethod
    def __estimatePhasor1Phase(t,x,f0hat):
        """Same as other function but Ahat has a simpler calculation"""
        H = (np.vstack((np.cos(2*math.pi*f0hat*t),np.sin(2*math.pi*f0hat*t))))
        Bhat = np.matmul(np.transpose(np.linalg.pinv(H)),x)
        Ahat = math.sqrt(np.sum(pow(Bhat,2)))/math.sqrt(2)
        Phat = np.arctan2(-Bhat[1],Bhat[0])
        return Ahat*np.cos(Phat) + 1j*Ahat*np.sin(Phat)