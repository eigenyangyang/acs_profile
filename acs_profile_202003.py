#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process data measured by profiling AC-S system (without filter).


Before using this script, please 
1) install:
- Python 3.7x or higher version    
- Essential Python packages: numpy, scipy, pandas, matplotlib.
2) prepare the following files (see examples as well as the main program) and 
put them in the same directory as the one containing AC-S data files:
- config_PostProc.txt
- AC-S device file (e.g. acs219_07032019.dev)
- thresholdsforDespike.txt
- rmSpectraIndex.txt
- MatchStationLabel.txt
- Sullivan_etal_2006_instrumentspecific.xls
- temperature and salinity data file (e.g. PS121_phys_oce_extracted.txt)


Analysis method detailed in:
Liu et al., 2020. In prep.

@author: 
    Yangyang Liu (yangyang.liu@awi.de), March 2020.
    
"""

import glob, os, math, shutil, datetime
import numpy as np
import numpy.matlib, scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as inp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from pandas import Series, DataFrame
from dateutil import parser #Date parser

# =============================================================================
def readConfig(config):
    paramDict = {}
    try:
        with open(config) as f:
            for line in f:
                if line.startswith('#')==False and len(line)>1:
                    key = line.strip().split('=')[0]
                    try:
                        value = line.strip().split('=')[1].strip()
                    except:
                        value = None
                    
                    paramDict[key] = value
    except Exception as e:
        print(e)
    finally:
        return paramDict
# =============================================================================
def acs_spectral_unsmooth(wl_acs, a):  
    #Reference: Chase et al., 2013. Decomposition of in situ particulate absorption 
    #spectra. Methods in Oceanography, 7, 110-124.
    
    wl_filtercorr = np.arange(0.1, 799.0+0.1, 0.1)
    SIG1 = (-9.845*10**(-8)*wl_acs**3 + 1.639*10**(-4)*wl_acs**2 - 
            7.849*10**(-2)*wl_acs + 25.24)/2.3547 
    
    filtfunc = np.empty([len(wl_filtercorr), len(wl_acs)])
    for i in range(len(wl_acs)):
        #First term normalizes area under the curve to 1.            
        filtfunc[:,i] = (1/(np.sqrt(2 * np.pi) * SIG1[i])) * np.exp(
                    -0.5 * ( (wl_filtercorr - wl_acs[i]) / SIG1[i] )**2 ) 
       
    error = 10**(-10) #stop iterating when error in calculacsed a is less than this from measured a.
    numbits = 100 #number of iterations
#    Convolve the measurement with the fiter factors add the difference to
#    the measured spectrum to get the first corrected spectrum and iteracse
#    to find the spectrum thacs when convolved with the filters, will give the
#    measured spectrum. This is the corrected absorption spectrum "acs_corr".    
    
    minwavel, maxwavel = min(wl_acs), max(wl_acs)
    xixi = np.linspace(minwavel, maxwavel, int(maxwavel*10)-int(minwavel*10)+1,
                       endpoint=True)

    for ii in range(numbits):
        func = ius(wl_acs, a)  #The range of centwavel is 0.1 nm.
        yiyi = func(xixi)  #Spline the measured dacsa to every 0.1 nm.
        #We need data from 0 to 799 nm to multiply by filtfac.
        absspec = np.zeros(len(wl_filtercorr))
        absspec[int(minwavel*10) : int(maxwavel*10)+1] = yiyi
        absspec[:int(minwavel*10)] = absspec[int(minwavel*10)]
        absspec[int(maxwavel*10)+1 :] = 0
        measur2 = np.zeros([len(wl_filtercorr), len(wl_acs)])
        meassignal6 = np.zeros(len(wl_acs))         
        for i in range(len(wl_acs)):
            #for all filters
            #measured signal for every filter factor.
            measur2[:,i] = absspec * filtfunc[:,i] 
            #measured spectrum acs a wl_filtercorr i is the sum of whacs a filter measured acs
            #all wl_filtercorrs times 0.1 as we did it acs every nm.
            meassignal6[i] = 0.1 * sum(measur2[:,i]) 
        acs_unsmooth = a - meassignal6 + a
        if max((a-meassignal6)/a) <= error:
            break
    return  acs_unsmooth
# =============================================================================
class acsProfilePostProc:
    
    def __init__(self, config='config_PostProc.txt'):
        paramDict = readConfig(config)
        self.device = paramDict['device']
        self.qft = paramDict['qft']
        self.lwcc = paramDict['lwcc']
        self.tsg = paramDict['tsg']
        self.tscoeff = paramDict['tscoeff']
        self.matched_labels = paramDict['matched_labels']
        self.dirs_drift = paramDict['dirs_drift'].split(',')
        self.blanks = paramDict['blanks']
    
    #--------------------------------------------------------------------------
    def saveData(self, filename, dfvarname, headline=None, index=False, 
                 header=False, sep='\t'):
        
        with open(filename, 'w') as f:
            if headline is not None:
                f.write(headline)
            dfvarname.to_csv(f, index=index, header=header, sep=sep, 
                             encoding='utf-8') 
    #--------------------------------------------------------------------------
    def createDir(self, dirname, overwrite=False):
        
        if overwrite:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            else:
                shutil.rmtree(dirname) #removes all the existing directories!
                os.makedirs(dirname)
        else:
            if not os.path.exists(dirname):
                os.makedirs(dirname)  
    #--------------------------------------------------------------------------
    def scatterPlot(self, dfx, dfy, xlabel, ylabel, figname):
        fig = plt.figure(figsize=(8.5, 6.5))
        plt.plot()
        plt.xticks(fontsize=12)
        try:
            plt.yticks(np.arange(np.nanmin(dfy), np.nanmax(dfy)+1, 1), fontsize=12)
            plt.scatter(dfx, dfy, s=1.5)
            plt.xlabel(xlabel,fontsize=15)
            plt.ylabel(ylabel,fontsize=15)
            fig.savefig(figname, dpi=200)
        except:
            print('Error: Unable to find data!')
        plt.close(fig)         
    #--------------------------------------------------------------------------
    def plotSpectrum(self, dfx, dfy1, figname, ylabel, xlabel='Wavelength [$nm$]', 
                     dfy2=None, legend=None, title=None):
        fig = plt.figure(figsize=(8.5, 6.5))
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        ax.tick_params(axis="both", labelsize=10)
        ax.plot(dfx, dfy1, linewidth=1)
        if dfy2 is not None:
            ax.plot(dfx, dfy2, linewidth=1)
        if legend is not None:
            ax.legend(legend, loc='upper left', bbox_to_anchor=(1, 1),
                      fontsize='small')
        if title is not None:
            plt.title(title)
        ax.set_xlabel(xlabel,fontsize=12)
        ax.set_ylabel(ylabel,fontsize=12)
        fig.savefig(figname, dpi=200)
        plt.close(fig)
    #--------------------------------------------------------------------------
    def getWL(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                if '%' in line:
                    break
        wl = pd.Series(line[1:].split(' ')).astype('float')  

        wl_selected = [420, 440, 580, 675, 700, 730]
        wlpos = {}
        for i, wvl in enumerate(wl_selected):
            key = 'pos'+str(wvl)
            value = np.where(wl>=wvl)[0][0]
            wlpos[key] = value
            
        return wl, line, wlpos
    #--------------------------------------------------------------------------
    def rsd_median(self, array):
    
        '''
        This function calculates relative median standard deviation (i.e. relative 
        median coefficient of variation, median standard deviation/median) of a 
        Pandas DataFrame for several rows.
        
        Input: array - Numpy ndarray.
        
        Output: 
        relative median coefficient of variation on row axis.
        '''
        row = array.shape[0]
        array_median = np.nanmedian(array,axis=0)
        diff_squre=(array-array_median)**2 
        sumofsquare = np.nansum(diff_squre, axis=0)
        if row>1:
            sd_median = sumofsquare/(row-1)
        else:
            sd_median = sumofsquare/row
        rsd = np.sqrt(sd_median)/abs(array_median)
        return rsd
    #-------------------------------------------------------------------------- 
    def rd_wetview(self, folder):
        '''
        Extraction of Wetlabs AC-S data from the Wetview software output (.dat files),
        and match the ac data with depth data.
        blank=True: MQ water (blank) data measured by acs;
        blank=False: Seawater sample data measured by acs.
        '''    
        #output list of all ac-s data directories 
        pd.DataFrame(dir_acs_all).to_csv('listofFolders.txt', index=False, 
                                         header=False, sep='\t')
    
        filename=sorted(glob.glob(os.path.join(folder, 'acs*.dat')))[0]
        foo = filename.split('/')
        print(f'Importing {foo[-1]}')
                
        try:
            
            with open(filename, 'r') as f:
                time_start = f.readline()
                time_start = time_start.split('\t') 
                time_start = time_start[1]+'\t'+time_start[2]
                date = parser.parse(time_start)
                date = pd.Series(date)
                for num, line in enumerate(f,1):
                    if 'acquisition binsize' in line:
                        break   
                line = f.readlines()      
            line = line[0].split('\t')  
            
            wlc=list()
            wla=list()
            for wavelength in line:
                if wavelength.startswith(('C', 'c')):
                    wl = float(wavelength[1:])
                    wlc.append(wl)
      
                if wavelength.startswith(('A', 'a')):
                    wl = float(wavelength[1:])
                    wla.append(wl)
            if not len(wlc)==len(wla):
                print('Unexpected number of wavelength.\n')
                
            wl = pd.Series(wla)
            pos_440 = np.where(wl>=440)[0][0]
    
            data = pd.read_csv(filename, header=None, skiprows=num+2, sep='\t') 
            data_dt = data.iloc[:,0]
            data_c = data.iloc[:,1:len(wla)+1]
            data_a = data.iloc[:,len(wla)+1:2*len(wla)+1]
            func = inp.interp1d(wlc, data_c, 'linear', fill_value='extrapolate')
            data_c = func(wla) #Interpolate data_c on wavelength_a scale
            data_c = DataFrame(data_c)
            
            #Check if there is a jump in timestmap of wetview
            max_idx = data_dt.idxmax()
            if max_idx!= len(data_dt)-1:
                delta = np.median(data_dt.values[1:max_idx+1] - data_dt.values[0:max_idx])
                dt_tmp = []
                for i in range(len(data_dt.iloc[max_idx+1:])+1):
                    tmp = data_dt.values[max_idx] + delta *i
                    dt_tmp.append(tmp)
                data_dt = pd.Series(list(data_dt.values[:max_idx]) + dt_tmp)
            
            data_time = pd.Series()
            time_step = Series.tolist(data_dt)
            for j in range(len(time_step)):
                time_counter = date + datetime.timedelta(milliseconds=time_step[j])
                data_time = data_time.append(time_counter)
            data_time.index = range(len(data_time)) 
            data_t = data_time.apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        
            depthfile = sorted(glob.glob(os.path.join(folder, 'Port2*')))[0]
            depth = pd.read_csv(depthfile, header=None, sep=" ")
            length_diff = len(depth) - len(data)
            print (f'Length difference between data and depth file number is \
                    {str(length_diff)}')
            depth = depth.iloc[length_diff:]
            output = DataFrame(np.hstack([data_t.values[:,np.newaxis], depth.values,
                                data_a.values, data_c.values]))
            
            #plot ac data against depth
            y = data_a.iloc[:,pos_440].transpose()
            self.scatterPlot(depth, y, 'depth [$m$]', 'a [$m ^{-1}$]', os.path.join(folder,'raw_a440.png'))
            y = data_c.iloc[:,pos_440].transpose()
            self.scatterPlot(depth, y, 'depth [$m$]', 'c [$m ^{-1}$]',  os.path.join(folder,'raw_c440.png'))
        
            headline = '%'+ ' '.join([str(i) for i in wla])+'\n'
            self.saveData(filename.replace('.dat','_extracted.txt'), output, 
                     headline=headline)
            print(f'Extraction of {foo[-1]} Finished!')
        
        except FileNotFoundError:
            print(f'Error: Unable to open file: {filename}!!!')
    #--------------------------------------------------------------------------    
    def rmSpikesDepBin(self, folder, a_Thold, c_Thold, a_cvThold, c_cvThold):
    
        '''
        This function removes spikes of ac-s data. It combines 2 ways:
        1) ac data measurements whose data at 440nm are greater than the thresholds 
        a(c)_Thold are removed.
        2) ac data measured within 2 seconds whose relative median coefficient of 
        variations are greater than a(c)_cvThold are removed. Also, the 10 seconds 
        meaurements centered on these outliers are removed. The choices of the 2 
        thresholds relies on the understanding of the data. If the users are not 
        sure which values to set to them, then leave them as default, i.e. 
        a_cvThold=None, c_cvThold=None.
    
        Input: 
        a(c)_cvThold - threshold for the median coefficient of variation of 
        a(c) measurements in 2 seconds (8 measurements). e.g. 5.
        a(c)_Thold - threshold for a(c) data at 440nm, [m^-1]. e.g. 2.
        
        Output: despiked ac-s data saved as "*despiked.txt", in which the columns 
        are datetime, a and c, respectively.
        '''
        filename = sorted(glob.glob(os.path.join(folder, '*extracted.txt')))[0]
        foo = filename.split('/')
        print(f'Removing spikes of {foo[-1]}')
        
        nmeas_1s = 4 #4 measurements per second.
        nmeas_10s = 10*4 #10 seconds' measurements
        
        wl, line, wlpos = self.getWL(filename)    
        pos_NIR = np.where(wl>700)[0]
    
        try:
            #Read data file
            data = pd.read_csv(filename, header=None, sep='\t',comment='%')
            data_a = data.iloc[:,2:len(wl)+2]
            data_c = data.iloc[:,len(wl)+2:]
            data_t = data.iloc[:,0]
             
            pos = np.where((data_a.iloc[:,wlpos['pos440']]>a_Thold) | 
                   (data_c.iloc[:,wlpos['pos440']]>c_Thold) |
                   (data_a.iloc[:,0]<0) | (data_c.iloc[:,0]<0))[0]
            data_a.iloc[pos,:] = np.nan
            data_c.iloc[pos,:] = np.nan
             
            #Calculate median coefficient of variation for a and c    
            for row in range(len(data_t)):     
                if row - nmeas_1s >= 0 and row + nmeas_1s <= len(data_t)-1:
                    nn = np.int_(range(row - nmeas_1s,row + nmeas_1s + 1))
                elif row - nmeas_1s < 0:
                    nn = np.int_(range(0,row + nmeas_1s+1))
                else:
                    nn = np.int_(range(row - nmeas_1s,len(data_t)))           
                a_cv = self.rsd_median(data_a.iloc[nn,range(pos_NIR[1])].to_numpy())
                c_cv = self.rsd_median(data_c.iloc[nn,range(pos_NIR[1])].to_numpy())
                if any(a_cv>a_cvThold) or any(c_cv>c_cvThold):
                    data_a.iloc[nn,:] = np.nan
                    data_c.iloc[nn,:] = np.nan
                    if row - nmeas_10s >= 0 and row + nmeas_10s <= len(data_t):
                        data_a.iloc[row - nmeas_10s : row + nmeas_10s,:] = np.nan
                        data_c.iloc[row - nmeas_10s : row + nmeas_10s,:] = np.nan

            #plot ac data against depth
            y = data_a.iloc[:,wlpos['pos440']].transpose()
            self.scatterPlot(data.iloc[:,1], y, 'depth [$m$]', 'a [$m ^{-1}$]', os.path.join(folder,'despike_a440.png'))
            y = data_c.iloc[:,wlpos['pos440']].transpose()
            self.scatterPlot(data.iloc[:,1], y, 'depth [$m$]', 'c [$m ^{-1}$]', os.path.join(folder,'despike_c440.png'))
    
            #Write output
            output = DataFrame(np.hstack([data_t.values[:,np.newaxis], 
                                             data.values[:,1].reshape(len(data),1),
                                             data_a.values, data_c.values]))

            self.saveData(filename.replace('.txt','_despiked.txt'), output, 
                     headline=line)
            
            data = pd.read_csv(filename.replace('.txt','_despiked.txt'), 
                               header=None, sep='\t',comment='%')
            data.iloc[:,1] = round(data.iloc[:,1])
            pos = np.where(data.iloc[:,1]==max(data.iloc[:,1]))[0]
            
            if len(data)-pos[-1]<100:
                #if few data on upcast, then take the data on downcast
                data_new = data.iloc[:pos[-1]+1,1:]
            else: #take the data on upcast
                data_new = data.iloc[pos[0]:,1:]
            data_new = data_new.iloc[np.where(data_new.iloc[:,0]>2)[0],:]
            #1-m bin and sorting data in depth-ascending order.
            data_bin = data_new.groupby(pd.Grouper(key=1)).median().dropna()
            data_bin = data_bin.sort_index(ascending=True)
            #standard deviation of data
            data_sd = data_new.groupby(pd.Grouper(key=1)).agg(np.std, ddof=1)
            data_sd = data_sd.sort_index(ascending=True)
            
            #plot ac data against depth
            y = data_bin.iloc[:,wlpos['pos440']].transpose()
            self.scatterPlot(data_bin.index, y, 'depth [$m$]', 'a [$m ^{-1}$]', os.path.join(folder,'despikeBin_a440.png'))
            y = data_bin.iloc[:,len(wl)+wlpos['pos440']].transpose()
            self.scatterPlot(data_bin.index, y, 'depth [$m$]', 'c [$m ^{-1}$]', os.path.join(folder,'despikeBin_c440.png'))
    
            self.saveData(filename.replace('.txt','_despiked_onemeter.txt'), data_bin, 
                     headline=line, index=True)
            self.saveData(filename.replace('.txt','_despiked_onemeter_sd.txt'), data_sd, 
                     headline=line, index=True)            
            print(f'Spikes removal and 1-m bin of {foo[-1]} Finished!')     
            
        except FileNotFoundError:
            print(f'Error: Unable to open file: {filename}') 
    #--------------------------------------------------------------------------    
    def tempsalcorr(self, folder, plot=True):
        '''
        This function performs temperature and salinity correction for ac-s data.
        
        Input:
        filename - absolute path of ac-s file named as "acs_extracted_despiked_merged_1min.txt".
        dev - absolute path of ac-s device file (*.dev).
        tsg - absolute path of the temperature and salinity data file (*.txt). 
        Inside, first column: date time in the format of "yyyy-mm-dd HH:MM:SS"; 
        second column: temperature in degrees Celsius;
        third column: salinity. Separators between columns are commas(",").
        tscoeff - absolute path of temperature and salinity correction coefficients 
              (Sullivan et al, 2006) file (*.xls).
    
        Output:
        Temperature and salinity corrected ac-s data saved as 
        "acs_extracted_despiked_merged_1min_tscorr.txt", in which the columns 
        are datetime, a and c, respectively.
        '''
        
        filename = sorted(glob.glob(os.path.join(folder, '*onemeter.txt')))[0]
        foo = filename.split('/')
        print(f'Correcting temperature and salinity effects for {foo[-1]}')
           
        #read temparature value during factory calibration (tcal) from device file.
        #(salinity value during factory calibration is 0, i.e. scal=0)
        with open (self.device, 'r') as f:
            for num, line in enumerate(f,1):
                if 'tcal:' in line:
                    break  
            tcal = float(line.split(' ')[1])
        
        #read TS data
        with open (self.tsg, 'r') as f:
            for i, line in enumerate(f,1):
                if '*/' in line:
                    num = i
                    break
                else:
                    num = 0
        data_ts = pd.read_csv(self.tsg, skiprows=num, sep='\t') 
        data_ts.rename(columns = {[s for s in data_ts.columns if 'Depth' in s][0]:'Depth'}, inplace = True) 
        col_idx = [data_ts.columns.get_loc('Depth'),
                data_ts.columns.get_loc([s for s in data_ts.columns if 'Temp' in s][0]),
                data_ts.columns.get_loc([s for s in data_ts.columns if 'Sal' in s][0])]
        
        try:
            wl, line, wlpos = self.getWL(filename) 
            #temperature & salinity dependency coefficients (Sullivan et al, 2006).
            tscoeff = pd.read_excel(self.tscoeff) 
            func_temp = inp.interp1d(tscoeff.iloc[:,0], tscoeff.iloc[:,1], 
                                'linear', fill_value='extrapolate')
            psi_temp = func_temp(wl)
        
            func_sala = inp.interp1d(tscoeff.iloc[:,0], tscoeff.iloc[:,5], 
                                'linear', fill_value='extrapolate')
            psi_sala = func_sala(wl)
            
            func_salc = inp.interp1d(tscoeff.iloc[:,0], tscoeff.iloc[:,3], 
                                'linear', fill_value='extrapolate')
            psi_salc = func_salc(wl)
                      
            #match ac-s data with TS data
            data_acs = pd.read_csv(filename, header=None, sep='\t',comment='%') 
            data_acs.rename(columns = {0:'Depth'}, inplace = True) 
            
            acs_labels = pd.read_csv(self.matched_labels, sep='\t') 
            pos = np.where(acs_labels.iloc[:,0] == filename.split('/')[-1].replace(
                    '_extracted_despiked_onemeter.txt',''))[0]
            pos_ts = np.where(data_ts.iloc[:,0] == acs_labels.iloc[pos,1].to_list()[0])[0]
            data_ts = data_ts.iloc[pos_ts,col_idx]
            data_ts['Depth'] = round(data_ts['Depth'])
            data_ts = data_ts.groupby(pd.Grouper(key='Depth')).median().dropna()
            func_temp = inp.interp1d(data_ts.index, data_ts.iloc[:,0], 'nearest', 
                                     fill_value='extrapolate')
            data_temp = func_temp(data_acs['Depth'])
            func_sal = inp.interp1d(data_ts.index, data_ts.iloc[:,1], 'nearest', 
                                    fill_value='extrapolate')
            data_sal = func_sal(data_acs['Depth']) 
            temp_delta = data_temp - tcal
            
            a_tscorr = DataFrame()
            c_tscorr = DataFrame()
            for row in range(len(data_acs)):
                tmp_a = data_acs.values[row,1:len(wl)+1] - temp_delta[row
                                        ] * psi_temp - data_sal[row] * psi_sala
                a_tscorr = a_tscorr.append(Series(tmp_a), ignore_index=True)
                
                tmp_c = data_acs.values[row,len(wl)+1:] - temp_delta[row
                                        ] * psi_temp - data_sal[row] * psi_salc
                c_tscorr = c_tscorr.append(Series(tmp_c), ignore_index=True)
                
                del tmp_a, tmp_c

            #Write output
            output = DataFrame(np.hstack([data_acs['Depth'].values[:,np.newaxis],
                                a_tscorr.values, c_tscorr.values]))
            self.saveData(filename.replace('.txt','_tscorr.txt'), output, 
                     headline=line)    
            
            
            if plot:
                dir_fig = os.path.join(folder,'plots_tscorr')
                self.createDir(dir_fig, overwrite=True)
                for i in range(len(output)):   
                    namestr = os.path.join(dir_fig, str(int(output.iloc[i,0]))+'m.png')
                    dfy1 = a_tscorr.iloc[i].transpose()
                    dfy2 = c_tscorr.iloc[i].transpose()
                    legend = ['TS corrected a','TS corrected c']
                    self.plotSpectrum(wl, dfy1, namestr, ylabel='$[m^{-1}]$', 
                                      dfy2=dfy2,legend=legend)
                
            print(f'{foo[-1]} TS corrected!')
        except:
            print(f'{foo[-1]} not processed due to some reason!!!')
    #-------------------------------------------------------------------------- 
    def derive_possible_drift(self, folder, plot=True):
        
        filename = sorted(glob.glob(os.path.join(folder,'*tscorr.txt')))[0]        
        wl_acs, line, wlpos = self.getWL(filename)
        
        acs = pd.read_csv(filename, header=None, comment='%', sep='\t') 
        labels = pd.read_csv(self.matched_labels, sep='\t') 
        pos_acs = np.where(labels.iloc[:,0] == folder)[0]
        labelstr = [str(lb) for lb in labels.iloc[pos_acs,2]][0]
        
        acs_a = acs.iloc[:,1:len(wl_acs)+1]
        acs_c = acs.iloc[:,len(wl_acs)+1:]
        #plot ac data against depth
        y = acs_a.iloc[:,wlpos['pos440']].transpose()
        self.scatterPlot(acs.iloc[:,0], y, 'depth [$m$]', 'a [$m ^{-1}$]', os.path.join(folder,'tscorr_a440.png'))
        y = acs_c.iloc[:,wlpos['pos440']].transpose()
        self.scatterPlot(acs.iloc[:,0], y, 'depth [$m$]', 'c [$m ^{-1}$]', os.path.join(folder,'tscorr_c440.png'))
            
        
        #QFT-ICAM
        ap_qft = pd.read_csv(self.qft, sep='\t') 
        ap_qft = ap_qft.drop(ap_qft.columns[6], axis=1)
        wl_qft = [float(str(wl).replace('wl','')) for wl in ap_qft.columns[6:]]
        pos_qft = np.where(ap_qft.iloc[:,0] == labelstr)[0]
        ap_qft = ap_qft.iloc[pos_qft,5:]
        ap_qft.index = round(ap_qft.iloc[:,0])
          
        #LWCC
        ag_lwcc = pd.read_csv(self.lwcc, sep='\t') 
        wl_lwcc = [float(str(wl).replace('wl','')) for wl in ag_lwcc.columns[6:]]
        pos_lwcc = np.where(ag_lwcc.iloc[:,0] == labelstr)[0]
        ag_lwcc = ag_lwcc.iloc[pos_lwcc,5:]
        ag_lwcc.index = round(ag_lwcc.iloc[:,0])
        
        #interpolation
        func_qft = inp.interp1d(wl_qft, ap_qft.iloc[:,1:], 'linear', 
                                fill_value='extrapolate')
        ap_qft_interp = DataFrame(func_qft(wl_acs), index=ap_qft.index, 
                                  columns=range(len(wl_acs)))
        func_lwcc = inp.interp1d(wl_lwcc, ag_lwcc.iloc[:,1:], 'linear', 
                                fill_value='extrapolate')
        ag_lwcc_interp = DataFrame(func_lwcc(wl_acs), index=ag_lwcc.index, 
                                  columns=range(len(wl_acs)))
        at_discrete = ap_qft_interp + ag_lwcc_interp 
        at_discrete = at_discrete.dropna()
        
        if len(at_discrete)>0:
            depth_common = list(set(acs.iloc[:,0]) & set(at_discrete.index))
            pos0 = [np.where(acs.iloc[:,0]==dep)[0][0] for dep in depth_common]
            tmp = [i for i, j in enumerate(acs_a.iloc[pos0,wlpos['pos440']]) if j == np.nanmin(acs_a.iloc[pos0,wlpos['pos440']])] 
            refdepth = acs.iloc[pos0[tmp[0]],0]
    #        refdepth = max(depth_common)
            pos_acs = np.where(acs.iloc[:,0]==refdepth)[0]
            pos_discrete = np.where(at_discrete.index==refdepth)[0]
            pos_qft = np.where(ap_qft_interp.index==refdepth)[0]
            pos_lwcc = np.where(ag_lwcc_interp.index==refdepth)[0]
            
            a_drift = acs_a.values[pos_acs,:] - at_discrete.values[pos_discrete,:]
            c_drift = acs_c.values[pos_acs,:] - at_discrete.values[pos_discrete,:]
            acs_drift = DataFrame(np.vstack([a_drift, c_drift]), index=['a','c'])
            
            #calculate the ratio of LWCC ag(440) and QFT ap(440) at reference depth
            ratio = ag_lwcc_interp.values[pos_lwcc,wlpos['pos440']]/ap_qft_interp.values[pos_qft,wlpos['pos440']]
            
            #plot ac drift
            y1 = a_drift.transpose()
            y2 = c_drift.transpose()
            self.plotSpectrum(wl_acs, y1, os.path.join(folder,'possible_acs_drift.png'), ylabel='$[m^{-1}]$', 
                              dfy2=y2,legend=['drift a','drift c']) 
            
            #plot LWCC CDOM again QFT ap
            y1 = ap_qft_interp.iloc[pos_qft,:].transpose()
            y2 = ag_lwcc_interp.iloc[pos_lwcc,:].transpose()
            legend = ['QFT $a_p(\lambda)$; $a_p(440)$='+str(ap_qft_interp.values[pos_qft,wlpos['pos440']]),
                        'LWCC $a_g(\lambda)$; $a_g(440)$='+str(ag_lwcc_interp.values[pos_qft,wlpos['pos440']])]
            title = str(refdepth) + '$m:\ a_g(440)/a_p(440)$=' + str(ratio)
            
            self.plotSpectrum(wl_acs, y1, os.path.join(folder,'lwcc_vs_qft.png'), ylabel='absorption $[m^{-1}]$', 
                              dfy2=y2,legend=legend, title=title)
     
            with open(os.path.join(folder,'acs_derived_possible_drift.txt'), 'w') as f:
                f.write(line)
                f.write('%Reference depth: '+str(refdepth)+' m \n')
                acs_drift.to_csv(f, index=True, header=False, encoding='utf-8', sep='\t') 
    #--------------------------------------------------------------------------
    def calc_drift(self):

        data = DataFrame()
        if self.dirs_drift == ['']:
            print('Please fill the parameter "dirs_drift" in "config_PostProc.txt"!!')
        else:
            for f in self.dirs_drift:
                tmp = pd.read_csv(os.path.join(f, 'acs_derived_possible_drift.txt'), 
                                   comment='%', sep='\t', header=None)
                data = data.append(tmp)
        
            wl, line, wlpos = self.getWL(os.path.join(f, 'acs_derived_possible_drift.txt'))
            #take the median and standard deviation of repeated data
            data_median = data.groupby(pd.Grouper(key=0)).median() 
            data_sd = data.groupby(pd.Grouper(key=0)).agg(np.std, ddof=1)
            
            #plot ac drift and standard deviation
            y1 = data_median.loc['a',:].transpose()
            y2 = data_median.loc['c',:].transpose()
            self.plotSpectrum(wl, y1, 'acs_derived_drift_final.png', ylabel='$[m^{-1}]$', 
                                  dfy2=y2,legend=['drift a','drift c']) 
            
            y1 = data_sd.loc['a',:].transpose()
            y2 = data_sd.loc['c',:].transpose()
            legend = ['standard deviation of drift a','standard deviation of drift c']
            self.plotSpectrum(wl, y1, 'acs_derived_drift_final_sd.png', ylabel='$[m^{-1}]$', 
                                  dfy2=y2,legend=legend) 
        
            self.saveData('acs_derived_drift_final.txt', data_median, headline=line, 
                          index=True) 
            self.saveData('acs_derived_drift_final_sd.txt', data_sd, headline=line, 
                          index=True) 
    #--------------------------------------------------------------------------        
    def subtract_drift(self, folder, derived=True, plot=True):
        
        filename = sorted(glob.glob(os.path.join(folder, '*tscorr.txt')))[0]
        foo = filename.split('/')
        print(f'Subtracting drift/blanks for {foo[-1]}')
        wl, line, wlpos = self.getWL(filename)
            
        data = pd.read_csv(filename, header=None, comment='%', sep='\t')
#        with open(filename.replace('_extracted_despiked_onemeter_tscorr.txt',
#                                   '.dat'), 'r') as f:
#            date_data = f.readline()
#            date_data = date_data.split('\t')[1]
#            date_data = parser.parse(date_data)
                
        blank = DataFrame()
        blanks = [os.path.join(wd, self.blanks)]
        for bl in blanks:
            tmp = pd.read_csv(bl, header=None, comment='%', sep='\t')
            blank = blank.append(tmp, ignore_index=True)  
        pos_a = np.where(blank.iloc[:,0] == 'a')[0]
        blank_a = blank.iloc[pos_a,:]
        pos_c = np.where(blank.iloc[:,0] == 'c')[0]
        blank_c = blank.iloc[pos_c,:]
    
    #    pos_a = blank_a.index.get_loc(date_data, method='nearest')
    #    pos_c = blank_c.index.get_loc(date_data, method='nearest')
    #    a_blankcorr = DataFrame(data.values[:,1:len(wl)+1] - blank_a.values[pos_a,2:])
    #    c_blankcorr = DataFrame(data.values[:,len(wl)+1:] - blank_c.values[pos_c,2:])
         
        if not derived:
            blank_a.index = pd.to_datetime(blank_a.iloc[:,1])
            blank_c.index = pd.to_datetime(blank_c.iloc[:,1])
            blank_a = np.nanmin(blank_a.iloc[:,2:], axis=0)
            blank_c = np.nanmin(blank_c.iloc[:,2:], axis=0)
        else:
            blank_a = np.squeeze(blank_a.iloc[:,1:].transpose().to_numpy())
            blank_c = np.squeeze(blank_c.iloc[:,1:].transpose().to_numpy())
              
        a_blankcorr = DataFrame(data.values[:,1:len(wl)+1]- blank_a[np.newaxis,:])
        c_blankcorr = DataFrame(data.values[:,len(wl)+1:] - blank_c[np.newaxis,:])
            
        #Write output
        output = DataFrame(np.hstack([data.values[:,0][:,np.newaxis],
                            a_blankcorr.values, c_blankcorr.values]))
        self.saveData(filename.replace('.txt','_driftcorr.txt'), output, headline=line)
        
        if plot:
            #Create target directory for plots
            dir_fig = os.path.join(folder, 'plots_ts_driftcorr')
            self.createDir(dir_fig, overwrite=True)
            #plot spectra
            for i in range(len(output)):       
                namestr = os.path.join(dir_fig, str(int(output.iloc[i,0]))+'m.png') 
                dfy1 = a_blankcorr.iloc[i].transpose()
                dfy2 = c_blankcorr.iloc[i].transpose()
                legend = ['TS+Blank corrected a','TS+Blank corrected c']
                self.plotSpectrum(wl, dfy1, namestr, ylabel='$[m^{-1}]$', 
                                      dfy2=dfy2,legend=legend)
        print(f'Drift/Blank correction for {foo[-1]} Finished!')
    #--------------------------------------------------------------------------
    def residTempScatCorr(self, folder, plot=True):
        '''
        This function corrects the residual temperature and scattering errors of 
        ap and residual temperature of cp from ac-s inline system. 
        Residual temperature correction of ap and cp follows Slade et al. (2010).
        
        Input:
        filename - absolute path of ac-s file named as "acs_p.txt".
        tscoeff - absolute path of temperature and salinity correction coefficients 
              (Sullivan et al, 2006) file (*.xls).
        scatcorr - Scattering correction method. If scatcorr=None (by default), 
        it takes the proportional method from Zaneveld et al. (1994); 
        if scatcorr=1, scattering correction follows Roettgers et al. (2013).
        
        Output:
        The finally quality controlled a and c data of particulate materials saved as
        "acs_TSalScatCorr.txt", in which the columns are datetime, a and c, respectively.
        '''
        filename = sorted(glob.glob(os.path.join(folder, '*driftcorr.txt')))[0]
        foo = filename.split('/')
        print(f'Correcting residual temperature and scattering effect for \
              {foo[-1]} ...')
        
        wl, line, wlpos = self.getWL(filename)
        pos_NIR = np.where(wl>700)[0]
        
        #temperature correction coefficients (Sullivan et al. 2006).
        tscoeff = pd.read_excel(self.tscoeff) 
        func_temp = inp.interp1d(tscoeff.iloc[:,0], tscoeff.iloc[:,1], 
                            'linear', fill_value='extrapolate')
        psi_temp = func_temp(wl).squeeze()
    
        #ac-s data
        data_acs = pd.read_csv(filename, header=None, sep='\t', comment='%')    
        a = data_acs.to_numpy()[:,1:len(wl)+1].astype('float')
        depth = data_acs.iloc[:,0].reindex(range(len(a)))
        #==========================================================================
        def costf_tscat_slade_modify(delta_temp, a, psi_temp, pos_NIR, pos_ref):
        
            '''
            This function defines the cost function according to or adapted from
            Equation 6 in Slade et al. (2010) for residual temperature correction for 
            cp data and combined residual temperature and scattering correction for 
            ap data. 
            
            Input:
            Data type of all input arguments and parameters are numpy array.
            delta_temp - residual temperature value between water samples and TSG 
            measured temperature.
            a - TS corrected particulate absorption coefficients.
            psi_temp - temperature correction coefficients from Sullivan et al. 2006.
            pos_NIR - indexes of near infrared wavelengths for ap/cp spectra.
            pos_ref - index of reference wavelength for scattering correction of 
            absorption measurements (ac-9: 715 nm, ac-s: 730 nm).
            
            Output:
            Cost function according to Equation 6 in Slade et al. (2010).
            '''
    
            costf = sum( abs( a[pos_NIR] - psi_temp[pos_NIR] * delta_temp
                         - (a[pos_ref] - psi_temp[pos_ref] * delta_temp) ) )
            return costf
        #==========================================================================
        a_TSalScatCorr = np.empty((len(a), len(wl))) * np.nan
        delta_temp = np.zeros(len(a)) * np.nan
        fiterr = np.zeros(len(a)) * np.nan
        
        for i in range(len(a)):
            if all(np.isfinite(a[i,:])):
                delta_temp0 = 0
                # minimization routine (Equation 6 in Slade et al. 2010)
                delta_temp[i] = scipy.optimize.fmin(func=costf_tscat_slade_modify, 
                                    x0=delta_temp0, 
                                    args=(a[i,:], 
                                          psi_temp, 
                                          pos_NIR, 
                                          wlpos['pos730']), 
                                          xtol=1e-8, ftol=1e-8, 
                                          maxiter=20000000, maxfun=20000,
                                          disp=0)
     
                fiterr[i] = costf_tscat_slade_modify(delta_temp[i], a[i,:], 
                                          psi_temp, 
                                          pos_NIR, 
                                          wlpos['pos730'])
    
                #null-point scattering correction.
                a_TSalScatCorr[i,:] = a[i,:] - psi_temp*delta_temp[i] - (
                    (a[i,wlpos['pos730']] - psi_temp[wlpos['pos730']]*delta_temp[i]))
    
        #Write ac-s output 
        output= DataFrame(np.hstack([depth.values[:,np.newaxis], a_TSalScatCorr.astype('object')]))
        self.saveData(filename.replace('.txt','_scatcorr.txt'), output, headline=line)
    
        #plot spectra
        if plot:
            dir_fig = os.path.join(folder, 'plots_tsdrift_scatcorr')
            self.createDir(dir_fig, overwrite=True)
            for i in range(len(output)):       
                namestr = os.path.join(dir_fig, str(int(output.iloc[i,0]))+'m.png') 
                dfy1 = a_TSalScatCorr[i].transpose()
                self.plotSpectrum(wl, dfy1, namestr, ylabel='$a_t\ [m^{-1}]$')
                       
        print(f'Residual temperature correction for c data and combined \
              residual temperature and scattering correction for a data for \
              {foo[-1]} Finished!')
    
    #--------------------------------------------------------------------------
    def at_unsmooth(self, folder, plot=True):
        
        filename = sorted(glob.glob(os.path.join(folder, '*scatcorr.txt')))[0]
        wl, line, wlpos = self.getWL(filename)
        acs = pd.read_csv(filename, header=None, comment='%', sep='\t') 

        #unsmooth ac-s data
        print(f'unsmoothing {filename}...')
        acs_unsmooth = np.zeros([len(acs), len(wl)]) 
        for i in range(len(acs)):
            a = acs.iloc[i,1:]
            acs_unsmooth[i,:] = acs_spectral_unsmooth(wl,a)
            del a
        at_acs = DataFrame(acs_unsmooth, index=acs.iloc[:,0])
        self.saveData(filename.replace('.txt','_unsmooth.txt'), at_acs, headline=line, index=True)
        
        if plot:
            dir_fig = os.path.join(folder,'plots_unsmooth')
            self.createDir(dir_fig, overwrite=True)      
            at_acs.insert(0,'row_index',range(len(at_acs)))
            for i in range(len(at_acs)):       
                namestr = os.path.join(dir_fig, str(int(at_acs.index[i]))+'m.png') 
                dfy1 = at_acs.iloc[i,1:].transpose()
                legend = [str(at_acs['row_index'].iloc[i])]
                ylabel='$a_t\ [m^{-1}]$'
                self.plotSpectrum(wl, dfy1, namestr, ylabel=ylabel,legend=legend)
        print(f'at of {filename} unsmoothed!')    
    
    #--------------------------------------------------------------------------
    def rmSpectra(self, folder, row_index=None, plot=True):
        
        filename = sorted(glob.glob(os.path.join(folder,'*unsmooth.txt')))[0]
        wl, line, wlpos = self.getWL(filename)        
        at = pd.read_csv(filename, header=None, comment='%', sep='\t') 
          
        #step correction at ~580nm
        at_stepbefore = at.iloc[:,wlpos['pos580']+1]
        at_stepafter = at.iloc[:,wlpos['pos580']+2]
        step = (at_stepbefore - at_stepafter)/2
        
        for i in range(len(at)):
            at.iloc[i,1:wlpos['pos580']+2] = at.values[i,1:wlpos['pos580']+2] - step[i]
            at.iloc[i,wlpos['pos580']+2:] = at.values[i,wlpos['pos580']+2:] + step[i]
            if at.iloc[i,wlpos['pos440']+1] < at.iloc[i,wlpos['pos675']+1]: 
                at.iloc[i, :] = np.nan
        
        pos = np.unique(np.where(at.iloc[:,1+wlpos['pos420']: wlpos['pos675']+2]<-1e-3)[0])
        at.iloc[pos, :] = np.nan
        
            
    #    fig, ax = plt.subplots()
    #    ax.plot(wl, at.iloc[4,1:])
    #    ax.scatter(wl.iloc[pos_580+1], at.iloc[4,pos_580+2], marker='o', color="red")
    #    ax.scatter(wl.iloc[pos_580], at.iloc[4,pos_580+1], marker='o', color="blue") 
        
        if not math.isnan(row_index[0]):
            at.loc[row_index, :] = np.nan    
        
        at = at.dropna()
        self.saveData(filename.replace('.txt','_rmspec_final.txt'), at, headline=line)
        if plot:
            dir_fig = os.path.join(folder, 'plots_at_rmspec_final')
            self.createDir(dir_fig, overwrite=True)
            #plot spectra        
            for i in range(len(at)):       
                namestr = os.path.join(dir_fig, str(int(at.iloc[i,0]))+'m.png') 
                dfy1 = at.iloc[i,1:].transpose()
                self.plotSpectrum(wl, dfy1, namestr, ylabel='$a_t\ [m^{-1}]$', 
                                  legend=[str(at.index[i])])

    #--------------------------------------------------------------------------
    def compare_qftlwcc(self, folder):
    
        filename = sorted(glob.glob(os.path.join(folder, '*final.txt')))[0]
        wl_acs, line, wlpos = self.getWL(filename)  
        
        acs = pd.read_csv(filename, header=None, comment='%', sep='\t') 
        acs.index = acs.iloc[:,0]
        labels = pd.read_csv(self.matched_labels, sep='\t') 
        pos_acs = np.where(labels.iloc[:,0] == folder)[0]
        labelstr = [str(lb) for lb in labels.iloc[pos_acs,2]][0]
        
        #QFT-ICAM
        ap_qft = pd.read_csv(self.qft, sep='\t') 
        ap_qft = ap_qft.drop(ap_qft.columns[6], axis=1)
        wl_qft = [float(str(wl).replace('wl','')) for wl in ap_qft.columns[6:]]
        pos_qft = np.where(ap_qft.iloc[:,0] == labelstr)[0]
        ap_qft = ap_qft.iloc[pos_qft,5:]
        ap_qft.index = round(ap_qft.iloc[:,0])
        
        #LWCC
        ag_lwcc = pd.read_csv(self.lwcc, sep='\t') 
        wl_lwcc = [float(str(wl).replace('wl','')) for wl in ag_lwcc.columns[6:]]
        pos_lwcc = np.where(ag_lwcc.iloc[:,0] == labelstr)[0]
        ag_lwcc = ag_lwcc.iloc[pos_lwcc,5:]
        ag_lwcc.index = round(ag_lwcc.iloc[:,0])
        
        #interpolation
        func_qft = inp.interp1d(wl_qft, ap_qft.iloc[:,1:], 'linear', 
                                fill_value='extrapolate')
        ap_qft_interp = DataFrame(func_qft(wl_acs), index=ap_qft.index, 
                                  columns=range(len(wl_acs)))
        
        func_lwcc = inp.interp1d(wl_lwcc, ag_lwcc.iloc[:,1:], 'linear', 
                                fill_value='extrapolate')
        ag_lwcc_interp = DataFrame(func_lwcc(wl_acs), index=ag_lwcc.index, 
                                  columns=range(len(wl_acs)))
        at_discrete = ap_qft_interp + ag_lwcc_interp
        
        self.saveData(os.path.join(folder, 'at_qft_lwcc_match.txt'), at_discrete, headline=line, index=True)
        
        at_acs = DataFrame()
        #match ac-s with discrete samples
        for i in range(len(at_discrete)):
            pos = np.where(acs.index == at_discrete.index[i])[0]
            at_acs = at_acs.append(acs.iloc[pos,1:len(wl_acs)+1])
        
        pos = [np.where(at_discrete.index==idx)[0][0] for idx in at_acs.index]
        at_discrete = at_discrete.iloc[pos,:]
        
        self.saveData(os.path.join(folder, 'at_acs_match.txt'), at_acs, headline=line, index=True)
      
        dir_fig = os.path.join(folder, 'plots_acs_qft_lwcc')
        self.createDir(dir_fig, overwrite=True)
        #plot spectra
        for i in range(len(at_acs)):       
            namestr = os.path.join(dir_fig, str(int(at_acs.index[i]))+'m.png') 
            dfy1 = at_acs.iloc[i,:]
            dfy2 = at_discrete.iloc[i,:]
            legend = ['ac-s','QFT+LWCC']
            xlabel='Wavelength [$nm$]'
            ylabel = '$a_t\ [m^{-1}]$'
            title = str(int(at_acs.index[i]))+'m'
            self.plotSpectrum(wl_acs, dfy1, namestr, ylabel, xlabel, dfy2, legend,
                              title)
                           
        
#-----------------------------------------------------------------------------
# Main Program. 
#-----------------------------------------------------------------------------
if __name__ == '__main__': 

    #Set working directory
    wd = '/isibhv/projects/Phytooptic/yliu/Data/cruises/acs/profile/PS121/data'
    os.chdir(wd)
    
    process = acsProfilePostProc()  
    
    dir_acs_all = [f.path.split('/')[-1] for f in os.scandir(wd) if f.is_dir()]    
    dir_acs_all.sort()
    
    #Step1: extract data
    outStep1 = [process.rd_wetview(folder) for folder in dir_acs_all]
    
    #Step2: remove spikes and bin data into 1-meter interval
    thresholds = pd.read_csv('thresholdsforDespike.txt', comment='#', sep='\t')
    outStep2 = [process.rmSpikesDepBin(row.iloc[0], row.iloc[1], row.iloc[2],
                                        row.iloc[3], row.iloc[4]) for i, row in thresholds.iterrows()]
    
    #Step3: TS correction and deirve possible drift data from samples
    outStep3_1 = [process.tempsalcorr(folder) for folder in dir_acs_all]
    outStep3_2 = [process.derive_possible_drift(folder) for folder in dir_acs_all]
    
    #Step4: calculate isntrument drift
    process.calc_drift()
    
    #Step5: subtract instrument drift, correct residual temperature and 
    #scattering effect and unsmooth at spectra.
    outStep5_1 = [process.subtract_drift(folder) for folder in dir_acs_all]
    outStep5_2 = [process.residTempScatCorr(folder) for folder in dir_acs_all]
    outStep5_3 = [process.at_unsmooth(folder) for folder in dir_acs_all]
    
    #Step6:
    #Prepare your own 'rmSpectraIndex.txt' file (tab delimited) based on the 
    #plots generated from Step0 in the working directory!
    SpectraIndex = pd.read_csv('rmSpectraIndex.txt', comment='#', sep='\t')
    outStep6_1 = [process.rmSpectra(row.iloc[0], [float(idx) for idx in str(row.iloc[1]).split(',')]) \
                   for i, row in SpectraIndex.iterrows()]
    outStep6_2 = [process.compare_qftlwcc(folder) for folder in dir_acs_all]

