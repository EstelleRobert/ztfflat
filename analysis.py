""" Data analysis : get some part of concat dataframes, creation of binned columns and show residual map """

from ztfquery import io
from astropy.io import fits
import pandas
import warnings
import numpy as np
import matplotlib.pyplot as plt


class ApertureCatalog(object):

    def __init__(self, data = None, radius = None):
        if data is not None:
            self.set_data(data, radius)


    @classmethod
    def read_parquet(cls, filename, **kwargs):
        data = pandas.read_parquet(filename)
        #radius = pandas.read_parquet(filename)
        radius = np.linspace(1,15,20)
        this = cls(data, radius)
        return this
    
    def read_hdf(cls, filename, **kwargs):
        data = pandas.read_hdf(filename, 'catalog')
        radius = pandas.read_hdf(filename, 'radius')
        this = cls(data, radius)
        return this


    # -------- #
    #  SETTER  #
    # -------- #
    def set_data(self, data, radius = None, **kwargs):
        """ Set the data and the radius if given """

        self._data = data
        if radius is not None:
            self.set_radius(radius, **kwargs)


    def set_radius(self, radius, warn = True):
        """ Set the radius"""
        if not self.has_data():
            if warn:
                warnings.warn('Radius given but data not given.')
        else:
            if len(radius) != self.nfluxes:
                raise ValueError('Size of given radius and size of radius in dataframe do not match.')

        self._radius = np.asarray(radius)


    # -------- #
    #  GETTER  #
    # -------- #
    def get_data(self, isolated = True, flagged = False, gmag_range = None, colormag_range = None):
        """ Get the wanted part of the dataframe
        
        Parameters
        ---------

        isolated: [bool]
        if True it takes only isolated stars
            
        flagged: [bool]
        if False it takes only stars with flag = 0 
            
        gmag_range: [int interval] - optional -
        to take only a part of the data in gmag range
        
        colormag_range: [int interval] - optional -
        to take only a part of the data in gmag range
            
        stat: [str] -optional-
        how the aperture is done:
        -median: perform the median of flux values
        -mean: perform the mean of flux values
            
        
        **kwargs go to self.get_data()
        
        
        Returns
        -----
        pandas dataframe
        """

        # flagged = False : selects stars with flag = 0
        data = self.data.copy()
        
        if gmag_range is not None:
            data = data[data['gmag'].between(*gmag_range)]#select the stars with this mag range
            
        if colormag_range is not None:
            data = data[data['colormag'].between(*colormag_range)]
            
        # isolated stars
        if isolated == True:
            data = data[data['isolated']]
        elif isolated == False:
            data = data[~data['isolated']]

        if flagged is not None:
            columns = []
            for i in range (0,len(self.radius)):
                columns.append(f'f_{i}_f')
            data_col = data[columns]
            flag = data_col.sum(axis = 1) == 0
            if flagged == True:
                data = data[~flag]
            #return flag values that are True
        else :
                data = data[flag]

        return data


    def get_aperture_flux(self, name, norm = 'asym', rad = None, stat='median', **kwargs):
        """ Get the aperture flux
        
        Parameters
        ---------

        name: [str]
        image name in .fits
            
        norm: [str] -optional-
        how to normalise:
        -asym: normalisation to the last radius
        -radius: normalisation to a specific radius
            
        rad: [float] -optional-
        radius for the normalisation if norm = radius
            
        stat: [str] -optional-
        how the aperture is done:
        -median: perform the median of flux values
        -mean: perform the mean of flux values
            
        
        **kwargs go to self.get_data()
        
        
        Returns
        -----
        """
        flux = []
        data = self.get_data(**kwargs)   
        columns = []
        for i in range (0,len(self.radius)):
            columns.append(f'f_{i}')

        df_image = data.xs(name)
        data_col = df_image[columns]  
        sources = data_col.reset_index('Source')['Source'] #all the sources

        if norm == 'asym':
            norm_val = data_col[data_col.columns[-1]]
        elif norm == 'radius':
            idx = (np.abs(self.radius-rad)).argmin()
            norm_val = data_col[data_col.columns[idx]]
        else:
            raise NameError('No such option.')

        flux_aperture = data_col/norm_val.values[:,None]

        if stat is None:
            d = []
            for source in sources:
                flux = data_col.loc[source][columns]
                norm = data_col.loc[source][columns][-1]
                flux_aperture = flux/norm
                dic = {f'{source}': flux_aperture}
                d.append(pandas.DataFrame(dic))
                
            return pandas.concat(d, axis = 1)
        
        return getattr(flux_aperture, stat) (axis = 0) #transforms into flux_aperture.statvalue


    def create_col_para(self, variable = 'flux_rad',  flux_rad = 10, n = None, **kwargs):
        """Calulate mean, delta, ratio of flux i and make a column with nb of stars entries in the dataframe 
        
        Parameters
        ---------
        
        variable: [str]
        aperture or psfcat:
        -flux_rad: flux at at a specific radius 
        -psfcat
            
        flux_rad: [int]
        index of radius of interest
        
        n: [int]
        keep stars that appear greater than n times in the dataframe
            
        **kwargs go to self.get_data()
           
            
        Returns
        -------
        pandas dataframe with:
        column with mean value of each star with the rad given
        column with delta value: mean column - flux column at given radius
        column with ratio value: (flux-mean)/mean
        column with n values
        """
        
        data = self.get_data(**kwargs)
        if variable == 'flux_rad':
            data[f'f_{flux_rad}_mean'] = data.groupby(level = 1)[f'f_{flux_rad}'].transform('mean')
            data[f'f_{flux_rad}_delta'] = data[f'f_{flux_rad}_mean'] - data[f'f_{flux_rad}']
            data[f'f_{flux_rad}_ratio'] = (data[f'f_{flux_rad}'] - data[f'f_{flux_rad}_mean']) / data[f'f_{flux_rad}_mean'] 
            data['n_entries'] = data.groupby(level = 1)[f'f_{flux_rad}'].transform('count')

        elif variable == 'psfcat':
            data[f'{variable}_mean'] = data.groupby(level = 1)['flux'].transform('mean')
            data[f'{variable}_delta'] = data[f'{variable}_mean'] - data['flux']
            data[f'{variable}_ratio'] = (data['flux'] - data[f'{variable}_mean']) / data[f'{variable}_mean']
            data['n_entries'] = data.groupby(level = 1)['flux'].transform('count')
        
        elif variable == 'both':
            data[f'f_{flux_rad}_mean'] = data.groupby(level = 1)[f'f_{flux_rad}'].transform('mean')
            data[f'f_{flux_rad}_delta'] = data[f'f_{flux_rad}_mean'] - data[f'f_{flux_rad}']
            data[f'f_{flux_rad}_ratio'] = (data[f'f_{flux_rad}'] - data[f'f_{flux_rad}_mean']) / data[\
f'f_{flux_rad}_mean']
            data[f'psfcat_mean'] = data.groupby(level = 1)['flux'].transform('mean')
            data[f'psfcat_delta'] = data[f'psfcat_mean'] - data['flux']
            data[f'psfcat_ratio'] = (data['flux'] - data[f'psfcat_mean']) / data[f'psfcat_mean']
            #data['n_entries'] = data.groupby(level = 1)[f'f_{flux_rad}', 'psfcat'].transform('count')

        if n is None:
            return data
        else : 
            return data[data['n_entries']>n]    
    
    
    def create_col_bin_coord(self, variable = 'flux_rad', flux_rad = 10, n = None, bin_nb = 30, 
                             right1 = True, right2 = True, group_stat = 'median', group_count = 'count',
                             gmag = False, **kwargs):
        """ Create binned columns for u and v coordinates : add columns with u and v digit,
        add column with u,v digit, 
        add columns with flux mean, ratio and delta
        
        Parameters
        ---------
        
        variable: [str]
        aperture or psfcat:
        -flux_rad: flux at at a specific radius 
        -psfcat
            
        flux_rad: [int]
        index of radius of interest
            
        n: [int]
        keep stars that appear greater than n times in the dataframe
          
        bin_nb: [int] -optional-
        number of bins
        
        right1: [bool]
        right parameter in numpy digitize for u coordinates
            
        right2: [bool]
        right parameter in numpy digitize for v coordinates
        
        group_stat: [str]
        how to apply groupby
        -median
        -mean
        
        group_count; [str]
        how to apply groupby in u,v_digit
        -count: to have all stars
        -nunique: to have only unique stars
           
        gmag

        **kwargs go to self.get_data()
            
            
         Returns
        -------
        data: dataframe of entrance with digit columns added
        binned_value: a dataframe with f{variable} ratio, u digit and v digit
        binned_value_c: a dataframe with nb of stars per u,v_digit
        """
        data = self.create_col_para(variable = variable, flux_rad = flux_rad, n = None, **kwargs)        
        if gmag is False: 
            data = self.create_col_para(variable = variable, flux_rad = flux_rad, n = None, **kwargs)
            bin_u = np.linspace(np.min(data['u']), np.max(data['u']), bin_nb)
            bin_v = np.linspace(np.min(data['v']), np.max(data['v']), bin_nb)
            data['u_digit'] = np.digitize(data['u'], bin_u, right = right1) #for each value tell in which bin it is
            data['v_digit'] = np.digitize(data['v'], bin_v, right = right2)
            data['u,v_digit'] = data['u_digit'].astype(str)+ ',' + data['v_digit'].astype(str)
        
            if n is None:
                data = data
            else : 
                data = data[data['n_entries']>n]
            
            if variable == 'flux_rad':
                if group_stat == 'median':
                    binned_value = data.groupby('u,v_digit')[[f'f_{flux_rad}_ratio', 'u_digit', 'v_digit', 
                                                      'u', 'v', 'x', 'y']].median() #median of ratio per digit
                elif group_stat == 'mean':
                    binned_value = data.groupby('u,v_digit')[[f'f_{flux_rad}_ratio', 'u_digit', 'v_digit', 
                                                      'u', 'v', 'x', 'y']].mean() #mean of ratio per digit
            elif variable == 'psfcat':
                if group_stat == 'median':
                    binned_value = data.groupby('u,v_digit')[[f'{variable}_ratio', 'u_digit', 'v_digit',
                                                      'u', 'v', 'x', 'y']].median() 
                elif group_stat == 'mean':
                    binned_value = data.groupby('u,v_digit')[[f'f_{flux_rad}_ratio', 'u_digit', 'v_digit', 
                                                          'u', 'v', 'x', 'y']].mean()    
                else:
                    raise NameError('No such option')

            else:
                raise NameError('No such option')

            if group_count == 'nunique':
                binned_value_c = data.reset_index('Source').groupby('u,v_digit')['Source'].nunique()

            elif group_count == 'count':
                print('hereh')
                binned_value_c = data.reset_index('Source').groupby('u,v_digit')['Source'].count()

            else: 
                raise NameError('No such option.')
        
########## Centroids ##########
            
            dic = {}
            dic['centroid_u'] = bin_u
            dic['centroid_v'] = bin_v
            bin_df = pandas.DataFrame(dic)
            centro_bin_u = pandas.DataFrame(bin_df['centroid_u'].loc[binned_value['u_digit'].values])
            centro_bin_v = pandas.DataFrame(bin_df['centroid_v'].loc[binned_value['v_digit'].values])


            binned_value = binned_value.merge(centro_bin_u.set_index(binned_value.index), on = 'u,v_digit')

            binned_value = binned_value.merge(centro_bin_v.set_index(binned_value.index), on = 'u,v_digit')
##########################
        data = self.create_col_para(variable = variable, flux_rad = flux_rad, n = None, **kwargs)

        if gmag is True:
            data = self.create_col_para(variable = variable, flux_rad = flux_rad, n = None, **kwargs)
            bin_gmag = np.linspace(np.min(data['gmag']), np.max(data['gmag']), bin_nb)
            data['gmag_digit'] = np.digitize(data['gmag'], bin_gmag, right = right1) #for each value tell in which bin it is
            if n is None:
                data = data
            else :
                data = data[data['n_entries']>n]
            print('hello')
            if variable == 'flux_rad':
                if group_stat == 'median':
                    binned_value = data.groupby('gmag_digit')[[f'f_{flux_rad}_ratio',
                                                      'u', 'v', 'x', 'y']].median() #median of ratio per digit
                elif group_stat == 'mean':
                    binned_value = data.groupby('gmag_digit')[[f'f_{flux_rad}_ratio',
                                                        'u', 'v', 'x', 'y']].mean() #mean of ratio per digit
                else:
                    raise NameError('No such option')
                    
            elif variable == 'psfcat':
                if group_stat == 'median':
                    binned_value = data.groupby('gmag_digit')[[f'{variable}_ratio',
                                                      'u', 'v', 'x', 'y']].median()
                elif group_stat == 'mean':
                    binned_value = data.groupby('gmag_digit')[[f'f_{flux_rad}_ratio',
                                                          'u', 'v', 'x', 'y']].mean()
                else:
                    raise NameError('No such option')

            else:
                raise NameError('No such option')

            if group_count == 'nunique':
                binned_value_c = data.reset_index('Source').groupby('gmag_digit')['Source'].nunique()

            elif group_count == 'count':
                binned_value_c = data.reset_index('Source').groupby('gmag_digit')['Source'].count()
            else:
                raise NameError('No such option')



        return data, binned_value, binned_value_c
        
        
    def create_bin_1d(self, key, bin_nb = 30, vmin = '0.1', vmax = '99.9', right = True, **kwargs):
        """ Get bins for a specified column
        
        Parameters
        ---------
        key: [str]
        column name to be binned
            
        bin_nb: [float] -optional-
        number of bins
        
        vmin: [str or float] -optional-
        percent value
        
        vmax: [str or float] -optional-
        percent value
            
        right: [bool]
        right parameter in numpy digitize   
            
        flux_norm: [int] -optional-
        to wich radius the normalisation is done
           
        
        *kwargs go to self.get_data()
        
        
        Returns
        -------
        pandas serie of wanted bins
        """
        
        data = self.get_data(**kwargs)
        values = data[key]
        if type(vmin) == str:
            vmin = np.percentile(values, float(vmin))
        if type(vmax) == str:   
            vmax = np.percentile(values, float(vmax))
        
        bins = np.linspace(vmin, vmax, bin_nb)
        binning = np.digitize(values, bins, right = right) #for each value tell in which bin it is

        return pandas.Series(binning, name = f'bin_{key}', index = data.index), bins #take the same indices as in data
    

    def get_binned_1d_fstat(self, key, bin_nb = 30, vmin = '0.1', vmax = '99.9', right = True, group_stat = 'median', flux_norm = 19, **kwargs):
        """ Get bins for flux columns
        
        Parameters
        ---------
        key: [str]
        column name to be binned
            
        bin_nb: [float] -optional-
        number of bins
        
        vmin: [str or float] -optional-
        percent value
        
        vmax: [str or float] -optional-
        percent value
            
        right: [bool]
        right parameter in numpy digitize   
            
        group_stat: [str]
        how to apply groupby
        -median
        -mean

        flux_norm: [int] -optional-
        to wich radius the normalisation is done
            
        *kwargs go to self.get_data()
        
        Returns
        -------
        dataframe with for all bins, the stat value of all fluxes
        """
        
        binned, bins = self.create_bin_1d(key = key, bin_nb = bin_nb, vmin = vmin, vmax = vmax, right = right, **kwargs)
        data = self.get_data(**kwargs)
        columns = self.f_columns
        #dataframe with fluxes noralised to a given flux
        norm_data = data[columns]/data[columns[flux_norm]].values[:,None] 
        # put binned serie next to norm_data 
        norm_data_merge = norm_data.merge(binned, left_index = True, right_index = True) #left and right index => take the indices that match
        
        if group_stat == None:
            return norm_data_merge
        else:
            return getattr(norm_data_merge.groupby(f'bin_{key}'), group_stat)() #make for e.g. the median of all f_i inside bin i and return the value 


    
    # -------- #
    #   show   #
    # -------- # 
    def show_residual_map(self, variable = 'flux_rad', flux_rad = 10, 
                          n = None, bin_nb = 30, s= 70, right1 = True, right2 = False, 
                          vmax = None, vmin = None, group_stat = 'median', group_count = 'count', scprop={}, **kwargs):
        """ Parameters go into self.create_col_bin_coord() 
        
        variable: [str]
            object of the plot:
            -flux_rad: flux at at a specific radius 
            -psfcat for e.g.
            
        flux_rad: [int]
            index of radius of interest
            
        n: [int]
            keep stars that appear greater than n times in the dataframe
        
        bin_nb: [int]
            number of bins
        
        group_stat: [str]
        how to apply groupby
        -median
        -mean

        group_count: [str]
        how to apply groupby in u,v_digit
        -count: to have all stars
        -nunique: to have only unique stars

        **kwargs go to self.get_data()
        
        """
        
        df, binned_value, binned_value_c  = self.create_col_bin_coord(variable = variable, 
                                                                      flux_rad = flux_rad, n = n, 
                                                                      bin_nb = bin_nb, right1 = right1, 
                                                                      right2 = right2, group_stat = group_stat, group_count = group_count,
                                                                      **kwargs)
        fig = plt.figure(figsize=[6.5,5])
        ax = fig.add_subplot()
        
        if variable == 'flux_rad':
            c = binned_value[f'f_{flux_rad}_ratio']*100
            c_label = f'(f_{flux_rad} - f_{flux_rad}_mean) / f_{flux_rad}_mean [%]'
        else:
            c = binned_value[f'{variable}_ratio']*100
            c_label = f'({variable} - {variable}_mean) / {variable}_mean [%]'
        
        if group == 'median':
            sc = ax.scatter(binned_value['u_digit'], binned_value['v_digit'], 
                c = c, vmax = vmax, vmin = vmin, 
                marker = 's', s = s, **scprop)
            ax.set_title(f"{self.ccd_info.to_dict('records')}")
            
        elif group in ['count', 'nunique']:
            sc = ax.scatter(binned_value['u_digit'], binned_value['v_digit'], 
                c = binned_value_c, marker = 's', s = s, **scprop)
            c_label = f'{group}'
            

        c_bar = fig.colorbar(sc)
        c_bar.set_label(c_label)
        



    def show_correction_evolution(self, variable = 'time', unit = 'radius', radius = 10, flux_norm = 19,
                                  flux_ratio_num = 'f_6', cmap = 'coolwarm', **kwargs):
        """ Parameters go into self.get_aperture_flux()

        variable: [str]
        the x variable plotted for the aperture correction study:
        -time
        -gmag
        -colormag

        unit: [str] -optional-
        how the correction is done:
        -fwhm: returns the correction for the fwhm radius
        -index: returns the correction for the radius of given index
        -radius: returns the correction for the given radius

        radius: [float] -optional-
        radius value or index of radius where to calculate the correction
            
        flux_norm: [int] -optional-
        to wich radius the normalisation is done
         
         
        flux_ratio_num: [str] -optional-
        numerator in the ratio of fluxes to plot
       
        flux_ratio_deno: [str] -optional-
        denominator in the ratio of fluxes to plot
            
        **kwargs go to self.get_data()
        
        """
        
        img_names = self.image_names
        
        fig = plt.figure(figsize = (6,4))
        ax = fig.add_subplot()
        
        if variable == 'time':
            corr = self.measure_correction(unit = unit, radius = radius)  

            ax.xaxis.set_visible(False)
            ax.set_xlabel('time', fontsize = 'large')
            ax.plot(img_names, corr,'+')
            ax.set_ylabel('aperture correction', fontsize = 'large')
        
        else:
            df = self.get_binned_1d_fstat(key = variable, flux_norm = flux_norm, **kwargs)
            bin_nb = len(df)
            cmap = plt.cm.get_cmap(cmap)
            ax.set_title(f'isolated, no saturated, {variable} binned stars', fontsize = 'large')
            ax.grid()
            ax.set_xlabel('radius (pixels)', fontsize = 'large')
            ax.set_ylabel('aperture flux', fontsize = 'large')
            
            for i,r in (df.iterrows()):
                ax.plot(self.radius, r, color = cmap(float(i)/bin_nb))
            
            fig = plt.figure(figsize = (6,4))
            ax = fig.add_subplot()
            cmap = plt.cm.get_cmap(cmap)
            bin_1d, bins = self.create_bin_1d(key = variable, **kwargs)
            ax.scatter(bins, (df[flux_ratio_num]/df[f'f_{flux_norm}']).values[:-1])
            #histogram of how many stars are in which color
            binned = pandas.DataFrame(bin_1d).groupby(f"bin_{variable}").size()
            ax_t = ax.twinx()
            step = np.diff(bins)[0]
            ax_t.bar(bins, binned.values[:-1], facecolor = plt.cm.binary(0.5, 0.1), width = 0.8*step, edgecolor = '0.5')
            ax.set_xlabel(f'{variable}')
            ax.set_ylabel(f'{flux_ratio_num}/f_{flux_norm}')
            


    # -------- #
    # property #
    # -------- #
    @property
    def ccd_info(self):
        name = self.image_names[0] #all images in names are from the same ccd and same quadrant
        dic_ccd = {'CCDID' : [fits.getheader(io.get_file(name))['CCDID']], 'QID' : [fits.getheader(io.get_file(name))['QID']], 'RCID' : [fits.getheader(io.get_file(name))['RCID']]}
        self._ccd = pandas.DataFrame(dic_ccd, index = [0])
        return self._ccd
    
    
    @property
    def data(self):
        """"""
        if not hasattr(self, '_data'):
            return None
        return self._data
    
    @property
    def radius(self):
        """"""
        if not hasattr(self, '_radius'):
            return None
        return self._radius
    
    
    @property
    def image_names(self):
        
        return self.data.index.levels[0]
    
    
    def has_data(self):
        """"""
        return self.data is not None #return True when not None
    
    @property
    def nfluxes(self):
        """Return size of radius by looking at dataframe columns"""
        
        return len([col for col in self.data if (col.startswith('f_') & col.endswith('f') )])
    
    @property
    def f_columns(self):
        columns = []
        for i in range (0,len(self.radius)):
            columns.append(f'f_{i}')
            
        return columns
