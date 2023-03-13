# -*- coding: utf-8 -*-
"""
Modified on Wed Dec 14 2022 by Kaustav Chatterjee
Created on Fri Aug 19 08:07:11 2022

This file uses the PMUTestLib to run tests on pmuClass with estimators ieeeClass and nllsClass_v2
This version includes the periodogram approach

@author: Dylan Tarter and Kaustav Chatterjee
"""
from IPython import get_ipython
import matplotlib.pyplot as plt

get_ipython().magic('reset -sf')
get_ipython().magic('clear -sf')
plt.close('all')

from pmuClass import pmuClass
from ieeeClass import ieeeClass
from nllsClass_v2 import nllsClass_v2
import PMUTestLib as PMUTests

# Settings
# =============================================================================
fs      = 960
f0      = 60
RR      = 60
tclass  = 'p'
doPlots = 0
cycles  = 2
# =============================================================================

# Make the IEEE PMU
pmu = pmuClass(fs,f0,RR,'filtered',ieeeClass(fs,f0,RR,tclass,name='60255-118-1'))
# test the PMU and export results to IEEE dictionary
IEEE = PMUTests.run(pmu,doPlots,'all')
# Make the NLLS PMU
pmu = pmuClass(fs,f0,RR,'windowed',nllsClass_v2(fs,f0,RR,tclass,cycles,doPreFilter=1,doROCOF=0))
# test the PMU and export results to NLLS dictionary
NLLS = PMUTests.run(pmu,doPlots,'all')

# save results to json files
PMUTests.writeJson(IEEE,'IEEEM')
PMUTests.writeJson(NLLS,'NLLSM')

# read the json files
IEEE = PMUTests.readJson('IEEEM')
NLLS = PMUTests.readJson('NLLSM')

# plot results read from json files
PMUTests.plot([IEEE,NLLS])