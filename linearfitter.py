
import numpy as np
import pandas
from .io import *
from . import starflat_nicolas_zp as nr_sf
from scipy.interpolate import RectBivariateSpline

from ztfquery import fields

def from_dataframe_to_recarray(dataframe, year = 2019, filter_ = 'zg', month = 3, keys=None, store = False):
        """ Save a .npy reccaray from .parquet dataframe to be used in fit code

        Parameters
        ----------

        data: [dataframe]
        dataframe to convert

        filter: [str]
        -zg
        -zr
        -zi

        year: [int]
        -2018
        -2019
        -2021

        month: [int]

        Returns
        -------
        npy recarray
        """
        mapping = {v:int(v.split('_')[1]) for v in dataframe.index.get_level_values(0).unique()}
        dataframe = dataframe.rename(index=mapping, level=0)
        if keys is None:
            dataframe = dataframe[["Source","qid", "ccdid", 'rcid',
                                   "x", "y", "u", "v", "ra","dec",
                                   'x_ps1', 'y_ps1','u_ps1', 'v_ps1','ra_ps1', 'dec_ps1',
                                   'x_psfcat', 'y_psfcat', 'ra_psfcat', 'dec_psfcat',
                                   "f_10", "f_10_e", "f_10_f", "psfcat", "psfcat_e",
                                   'gmag', 'g_mag','e_gmag', 'g_magErr',
                                   'rpmag','rmag_ps1', 'e_rpmag', 'e_rmag_ps1',
                                   'bpmag', 'imag_ps1','e_bpmag','e_imag_ps1',
                                   'zmag_ps1', 'e_zmag_ps1', "isolated", "colormag"]].reset_index().rename({"level_0":"img_id"}, axis=1)
        else:
            dataframe = dataframe[keys].reset_index().rename({"level_0":"img_id"}, axis=1)
            
        dataframe_npy = dataframe.to_records(column_dtypes={'qid':np.int, 'isolated': np.int})

        if store:
            recarray_path = get_recarray_path(filter_ = filter_, year= year, month = month)

            np.save(recarray_path, dataframe_npy)

        return dataframe_npy



class StarFlatFitter():

    
    def __init__(self, npyfile, load=True):
        """ """
        self.npyfile = npyfile
        if load:
            self.load(npyfile)
        
    @classmethod
    def from_npyfile(cls, npyfile, load=True):
        """ """
        return cls(npyfile, load=load)
        # If rcids is None - build_from_filename
        #this.rcids = rcids
          
    def load(self, npyfile, psf_flux=False,  starflat = False, check_fits = False, superpix=10):
        """ """
        dp = nr_sf.load(npyfile, psf_flux = psf_flux, starflat = starflat, check_fits = check_fits)
        dp, m = nr_sf.pixellize(dp, nx = superpix, ny = superpix)
        model = nr_sf.starflat_model_zpexp(dp)
        _,_,solver,x = nr_sf.fit(dp,model)
        nr_sf.pars_to_starflats(m, model.params, plot=False)

        self.superpix = superpix
        self.dp  = dp
        self.model = model
        self.solver = solver
        self.metapixdata = pandas.DataFrame(m)

        
    def get_interpolator(self, rcid, kx=2, ky=2):
        """ """
        metapixdata_rcid = self.metapixdata[self.metapixdata['rcid'] == rcid]
        dzp = metapixdata_rcid["dzp"].values
        x = metapixdata_rcid["xc"].values
        y = metapixdata_rcid["yc"].values
       
        return RectBivariateSpline(np.unique(x), np.unique(y), dzp.reshape(self.superpix,self.superpix), kx=kx, ky=ky)


    def make_fits(self, psf_flux = False, save_fits = False, keys = None, store_result = True, **kwargs):
        """ """
        if psf_flux == True:
            flux = 'psfcat'
            flux_estimator = 'psf'        
        else:
            flux = 'f_10'
            flux_estimator = 'ap'
        
        array = np.load(self.npyfile, allow_pickle=True)
        rcids = fields.ccdid_qid_to_rcid(self.metapixdata['ccd'], self.metapixdata['qid'])
        self.metapixdata['rcid'] = rcids
        df_data = pandas.DataFrame(array)
        
        #For each quadrant
        df_data['dzp_corr'] = 0
        for i in range(0,64):
            ccdid,qid = fields.rcid_to_ccdid_qid(i)
            idx = df_data.rcid == i
            f = self.get_interpolator(rcid = i, **kwargs)

            correction = f(df_data['x'][idx], df_data['y'][idx], grid = False)
            df_data.loc[idx,'dzp_corr'] = correction

            if save_fits:
                X,Y = np.mgrid[0:3072,0:3080] # DOUBLE CHECKED !
                dzp_grid = f(X,Y, grid = False)
                fits_path_ztf = get_fits_path(self.npyfile, ccdid=ccdid, qid=qid, flux_estimator=flux_estimator)
                fits.writeto(fits_path_ztf, dzp_grid[::-1,::-1].T) # DOUBLE CHECKED !

        df_data[f'{flux}_corr'] = df_data[f'{flux}']*10**(+0.4*df_data['dzp_corr']) #put correction from fit into originate dataframe

        if keys is None:
            keys = ["qid", "ccdid", "rcid","x", "y", "u", "v", "ra", "dec", "f_10", "f_10_e",
                    "f_10_f", "psfcat", "psfcat_e", "isolated", "colormag", 'Source', 'img_id',
                    f'{flux}_corr','dzp_corr']

        data_path_fit = get_path_fitresult(self.npyfile, self.superpix, extension = 'npy', flux_estimator = flux_estimator)
        if store_result:
            data = df_data[keys].reset_index().rename({"level_0":"img_id"}, axis=1) #faire en sorte d'utiliser une seule fonction de tranformation dataframe -> recarray
            data_path_fit = get_path_fitresult(self.npyfile, self.superpix, extension = 'npy', flux_estimator = flux_estimator)
            data_path_parquet = get_path_fitresult(self.npyfile,  self.superpix, extension = 'parquet', flux_estimator = flux_estimator)
            np.save(data_path_fit, data.to_records(index=False))
            data.to_parquet(data_path_parquet)

        self._corr_cat_path = data_path_fit

        return data

    def plot_starflat(self, npyfile=None, starflat_boucle = False, ccd=None, subtract_gains = False, vmin = -0.005, vmax = 0.005, hexbin = True):
        """ """
        if starflat_boucle == False:
            nr_sf.plot_model(self.dp, self.model, self.solver, ccd=ccd, subtract_gains=subtract_gains, vmin = vmin, vmax = vmax, hexbin = hexbin)
        else:
            self.load(self._corr_cat_path, psf_flux=False,  starflat = True, check_fits = False, superpix=self.superpix)
            return nr_sf.plot_model(self.dp, self.model, self.solver, ccd=ccd, subtract_gains=subtract_gains, vmin = vmin, vmax = vmax, hexbin = hexbin)
            
        
    @property
    def corr_cat_path(self):
        """ """
        if hasattr(self,"_corr_cat_path"):
            return self._corr_cat_path

        return self.make_fits()
