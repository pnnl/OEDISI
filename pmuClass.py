# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 14:32:21 2022

PMU Base Class for simulating PMU Estimators

- This class's purpose is to pass data to and from a PMU
estimator class. IEEE and NLLS are currently implemented.
- To use this, create the class with proper inputs, then pass
a timestamp and measurement array into the .run function.
    - fs (sampling rate) must be an integer multiple of both RR (report rate) and
    f0 (nominal frequency), otherwise strange behavior occurs.
- Valid inputs can be 1 phase, 3 phase, and 3 voltage + 3 current phase (stacked arrays)
- Frequency and ROCOF may or may not be calculated when only given 1 phase,
it depends on the estimator used.
- There are 2 modes of data parsing. Filtered and Windowed. 
    - Filtered will directly pass the measurements to the estimator, and expects
    many estimates to be passed back. The estimates are then dyadically downsampled.
    - Windowed will splice the measurements based on the estimator's cycles per window value.
    Each window is ensured to have an odd amount of points so that the report timestamp can be
    placed at the center of each window.
- Outputs are T,X,F,DF
    - T are report rate timestamps
    - X are phasor measurements at reports
        - For 1 phase, its 1 estimate
        - For 3 phase, the estimates are [Xa, Xb, Xc, Xp]
        - For 3+3 phase, the estimates are [Xa, Xb, Xc, Xp, Ya, Yb, Yc, Yp]
    - F are frequency estimates at reports
    - DF are rate of change of frequency (ROCOF) at reports
    
@author: Dylan Tarter
"""
import numpy as np
import math

class pmuClass:
    def __init__(self, fs, f0, RR, method, estimator):
        self._fs = fs
        self._f0 = f0
        self._RR = RR
        self._method = method.lower() #force to lower case for testing later
        self._estimator = estimator   #estimator must be a special class
        
        # tests to give warning when fs is not integer multiple of f0 or RR
        if(not fs/f0 == int(fs/f0)):
            print('[PMU WARNING] Sampling rate is not an integer multiple of the nominal frequency!')
            print('strange behavior or errors may occur. Please pick a proper sample rate.')
        if(not fs/RR == int(fs/RR)):
            print('[PMU WARNING] Sampling rate is not an integer multiple of the reporting rate!')
            print('strange behavior or errors may occur. Please pick a proper sample rate.')
        
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
    def method(self):
        """Return the pmu's method of handling data (filtered or windowed)."""
        return self._method
    @property
    def estimator(self):
        """Return the pmu's estimator class (the actual PMU algorithm used)."""
        return self._estimator
    
    def run(self, t, x):
        """
        If 1 phase is inputed, output 1 phase estimate. IEEE will output NaN for freq in that case,
        NLLS can guess freq.

        If 3 phases are inputted, output 4 phasors [Xa,Xb,Xc,Xp], and the frequency from them with ROCOF

        If 6 phases are inputted, output 8 phasors [Va,Vb,Vc,Vp,Ia,Ib,Ic,Ip] with freq. calculated from 
        the Voltage phasors, and frequency of it is passed to the current phasors. For IEEE, this does
        not do much except in P class it will mean the correctly is based on it.

        """
        t,x = self.__fixShape(t, x) # ensure that x is a vertical array
        
        if(not (x.size == x.shape[0] or np.size(x,1) == 3 or np.size(x,1) == 6)):
            print('[PMU] Warning! Inputted a strange amount of signals, undefined behavior may occurr.')
            print('please try to input either 1 phase, 3 phase, or 3 voltage phase + 3 current phase (6)')
        
        """Apply pre-filtering for harmonic rejection (if the pmu has it)"""
        # this was necessary for the NLLS to pass tests using a BPF
        preFilterFunc = getattr(self._estimator, "preFilter", None)
        if(callable(preFilterFunc)): 
            t,x = preFilterFunc(t,x)
        
        """Pick based on stored method"""
        if(self._method == 'filtered'):
            T,X,F,DF = self.__filtered(t, x)
        elif(self._method == 'windowed'):
            T,X,F,DF = self.__windowed(t, x)
        else:
            print('[PMU] ERROR, unrecognized data handling method. Try filtered or windowed.')
        
        return T,X,F,DF
    
    @staticmethod
    def __fixShape(t,x):
        #Ensures that inputs are column matrices
        if(x.shape[0] < (np.size(x)-x.shape[0])):
            x = np.transpose(x)
        return t,x
    
    def __filtered(self, t, x):
        """
           Data handling method that works with filter-based PMUs.
           These PMUs tend to operate on all the data and output at a rate of fs
           and then must be downsampled in some way to the reporting rate.
           
           IEEE C37.118 Part D (2018) standard PMU examples are of this type.
           Both classes work with basic dyadic downsampling, a filter may be needed
           for other PMU algorithms.
        """
        
        X, F, DF = self.estimator.run(t,x)
        return self.__decimate(t,X,F,DF)
   
    def __decimate(self,t,X,F,DF):
        """Dyadic Decimation of the estimated values and timestamps, synchronized to report rate"""
        RRidx = self.__getRRidx(t) # RRidx = Report Rate indices
        return t[RRidx], X[RRidx], F[RRidx], DF[RRidx] 
    
    def __windowed(self, t, x):
        """Data handling method that works with window-based PMUs.
           Each window is ensured to have a report rate timestamp that lies in the center
           of the window. Window size is determined by the estimator's "cycles" variable.
           Built specifically for James Follum & Dylan Tarter's Non-Linear Least Squares estimation method,
           but would also work on some DFT window-based methods.
        """
        RRidx = self.__getRRidx(t) # get indices of reports ahead of time
        
        W   = self._estimator.cycles*self._fs/self._f0 # calculate points in window
        if((W % 2) == 0): # ensure its an odd number of points so the report can go at center point
            W += 1
        wHalf = (W-1)/2 # half of the window
        
        """The windowPts array is a boolean of reports that will be inside the measurements. This will typically
        just put falses on the left and right edges of the array. Bigger the window, more falses / borders."""
        windowPts = (np.array(RRidx) <= (x.shape[0] - wHalf - 1)) & (np.array(RRidx) >= wHalf) 
        if(x.size == x.shape[0]): # test if its 1D array
            X  = np.empty(np.size(RRidx), dtype=complex) * np.nan
        else:
            #if its not 1D array, make phasor array have 3 estimates and 1 additional for the positive sequence
            #so if a 3 phase is passed, its 3+1 phasors. if 3+3 phases are passed, its 3+1+3+1 phasors.
            X  = np.empty((np.size(RRidx),np.size(x,1)+math.floor(np.size(x,1)/3)), dtype=complex) * np.nan

        F  = np.empty(np.size(RRidx)) * np.nan # 1 Frequency estimate per report
        DF = np.empty(np.size(RRidx)) * np.nan # 1 ROCOF estimate per report
        
        """For each report calculate the values"""
        for i in range(0,np.size(RRidx)):
            if(windowPts[i]): # if the window lies within the measurements
                xwindow = x[np.arange(int(RRidx[i]-wHalf),int(RRidx[i]+wHalf+1))] # grab the window
                if(np.sum(np.isnan(xwindow)) == 0): #ensure the window has no NaN's
                    X[i], F[i], DF[i] = self._estimator.run(np.arange(-wHalf,wHalf+1,1)/self._fs,xwindow)

        T = t[list(map(int, RRidx))] # output timestamps at RR
        
        return T,X,F,DF
    
    def __getRRidx(self,t):
        """get the indices of the timestamps that align with the report rate"""
        t0 = math.ceil(min(t)) # first timestamp should lie on a whole number inside the array
        t1 = t
        "Exception for if the first expected report is not found in the timestamps"
        if(self.__findNearest(t1,t0) == None):
            t1 = np.concatenate((t1,np.arange(t1[-1]+1/self._fs,t0,1/self._fs))) #if that whole number is not in the array, add points until you reach it.
            
        idx   = self.__findNearest(t1,t0) # find the location of the whole number idx
        idxR  = np.arange(idx,np.size(t1),self._fs/self._RR) # extrapolate indices after idx
        idxL  = np.arange(idx,-self._fs/self._RR,-self._fs/self._RR) #extrapolate indices before idx
        RRidx = np.concatenate((np.flip(idxL[1:]),idxR))  # combine indices into 1 array
        RRidx = RRidx[RRidx <= np.size(t)] #only pick indices that lie in the original timestamps
        return list(map(int, RRidx)) #export them as ints so they can be used later to index arrays
    
    @staticmethod
    def __findNearest(array, value):
        """find the index of the nearest 'value' in 'array'"""
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin() # find location of minimum
        if(abs(array[idx] - value) > 0.001): # return None index if 0.001 off of the search value
            return None
        return idx