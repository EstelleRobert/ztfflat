import os

import numpy as np
import pandas
import dask.dataframe as dd
from ztfimg import aperture
from ztfquery import fields
import dask
from ztfimg import science


from .io import *


def get_columns(sciimg, incl_radec=False):
    """ For a starflat sciimg.fits file, build ra, dec, rcid and filename columns
    
    Parameters
    ----------
    
    filename: [str]
    starflat sciimg.fits file
    
    Returns
    -------
    ra, dec, rcid, filename values
    """
    if type(sciimg):
        filename = sciimg
        if incl_radec:
            img = science.ScienceQuadrant.from_filename(filename)
    else:
        img = sciimg
        filename = img.filename

    ccdid = int(filename.split('_')[4].replace('c',''))
    qid = int(filename.split('_')[6].replace('q',''))
    rcid = fields.ccdid_qid_to_rcid(ccdid, qid)
    if incl_radec:
        ra = img.get_center("radec")[0]
        dec = img.get_center("radec")[1]
    else:
        ra, dec = np.NaN, np.NaN
    
    return ra, dec, rcid, filename



class Starflat():

    def __init__(self, scimgs=None, load_datafile=False):
        """ """
        self.set_scimgs(scimgs)
        if load_datafile:
            self.load_datafile()
        return None

    
    # -------- #
    # I/O      #
    # -------- #
    @classmethod
    def from_filenames(cls, starflat_filenames, load_img=False, filtername=None, year=None, month=None,**kwargs):
        """ """
        if load_img:
            scimgs = [science.ScienceQuadrant.from_filename(filename) 
                  for filename in starflat_filenames]
        else:
            scimgs = None

        this = cls(scimgs, **kwargs)
        this._filenames = starflat_filenames
        this._filtername = os.path.basename(starflat_filenames[0]).split("_")[3] if filtername is None else filtername
        this._year = int(os.path.basename(starflat_filenames[0]).split("_")[1][:4]) if year is None else year
        this._month = int(os.path.basename(starflat_filenames[0]).split("_")[1][4:6]) if month is None else month
        return this
        
    @classmethod
    def from_date_and_filter(cls,  filter_, year, month, **kwargs):
        """ """
        filenames = get_starflat_files(filter_=filter_, year=year, month=month)
        
        return cls.from_filenames(filenames, filtername=filter_, year=year, month=month, **kwargs)
        
    # -------- #
    # SETTER   #
    # -------- #
    def set_scimgs(self, scimgs):
        """ """
        self._scimgs = scimgs
        
    def set_datafile(self, datafile):
        """ """
        self._datafile = datafile

    # -------- #
    # GETTER   #
    # -------- #

    def load_datafile(self, incl_radec=False):
        """ """
        datafile = self.get_datafile(incl_radec=incl_radec)
        #self._datafile = datafile
        self.set_datafile( datafile )

    def get_datafile(self, incl_radec = False, filenames=None):
        """ Build a dataframe with ra, dec, rcid, filename columns for all starflat sciimf.fits

        Parameters
        ----------

        filenames: [str]
        starflat sciimg.fits files

        Returns
        -------
        pandas dataframe
        """
        if filenames is None:
            if self.scimgs is not None and incl_radec:
                _loop_over = self.scimgs
            else:
                _loop_over = self.filenames
        else:
            _loop_over = filenames

        results = [dask.delayed(get_columns)(sciimg, incl_radec=incl_radec) for sciimg in _loop_over]
        resultvalues = dask.delayed(list)(results).compute()
        return pandas.DataFrame(resultvalues, columns=["ra","dec","rcid", 'filename'])


    # ------------- #
    #  BUILDER     #
    # ------------ #
    def build(self, radius=None, sep_limit = 20, rcids=None, store = False, **apkwargs):
        """
        radius:
            if None: linspace(1,15,20)
        """
        if radius is None:
            radius = np.linscape(1,15,20)
            
        if rcids is None:
            rcids = np.arange(64)
        else:
            rcids = np.atleast_1d(rcids)

        aper_fileout = self.build_concat_cat(radius, sep_limit=sep_limit, rcids=rcids, **apkwargs)
        dataframe_from_fileout, recarray_from_fileout = self.buid_merged_catalog(filter_ = self.filtername,
                                                                                  year = self.year,
                                                                                   month = self.month, store = store,
                                                                                **kwargs) 
        starflat_fitter = StarFlatFitter.from_aperturefiles(aper_fileout, load=True, rcids=rcids)
        # load -> fit and build interopator
        

    def build_concat_cat(self, radius,  rcids = None, sep_limit = 20,
                         calibrators = 'gaia', extra = ['ps1','psfcat'], store = True,  **apkwargs):
        """ For each rcid, build a concat dataframe for all fits files

        Parameters
        ----------

        radius: [list of float]
        aperture radius (same for all images)

        rcids: [int]
        rcids values of interest

        sep_limit: [int] -optional-
        no neighbors in the sep_limit radius around the considered star (arcsec)

        calibrators: [str]
        -gaia

        extra: [list of str]
        -psfcat
        -ps1

        store: [bool]
        if store: build pathname to store concat dataframe

        Returns
        -------
        list of concat pandas dataframes (delayed)
        if store: list of all concat dataframes (delayed), list of pathnames to store them
        """
        cats = []
        file_out_list = []

        if rcids is None:
            rcids = np.arange(64)
        else:
            rcids = np.atleast_1d(rcids)

        dir_path =  os.path.join(DIRPATH_CONCAT, f'{self.year}/{self.month:02}')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        # load
        rcid_filenames = self.datafile.groupby("rcid")["filename"].apply(list)
        for rcid in rcids:
            files = rcid_filenames.loc[rcid]
            ap = aperture.AperturePhotometry.from_filenames(files)
            # -
            cat = ap.build_apcatalog(radius, calibrators = calibrators, extra = extra, isolation = sep_limit)
            # -
            ccdid, qid = fields.rcid_to_ccdid_qid(rcid)
            file_out = os.path.join(dir_path, f'{self.year}{self.month:02}_{self.filtername}_c{ccdid:02}_o_q{qid}_apconcat.parquet')
            # -
            cats.append(cat)

            if store:
                file_out_list.append(file_out)

        if store:
            cats_values = dask.delayed(list)(cats).compute()
            for cat, file_out in zip(cats_values, file_out_list):
                cat.to_parquet(file_out)
            return file_out
        else:
            return cats, file_out
        

    def build_merged_catalog(self, as_npy = False,  year = 2019, filter_ = 'zg', month = 3, store = True, **kwargs):
        """ Build the dataframe to perform the fit for the focal plan

        Parameters
        ----------

        client: [daks client]
        mandatory

        filter: [str]
        -zg
        -zr
        -zi

        year: [int]
        -2018
        -2019
        -2021

        month: [int]

        store: [bool]
        if True, store the final dataframe

        Returns
        -------
        usable dataframe (focal plan) for the fit
        """
        dfs = get_concat_df(filter_ = filter_, year = year, month = month)
        df_list = []

        for df in dfs:
            dataframe = dask.delayed(self.build_dataframe)(df = df, **kwargs)
            df_list.append(dataframe)

        df_parquet = pandas.concat(dask.delayed(list)(df_list).compute())

        if store:
            parquet_path = get_parquet_path(filter_ = filter_, year = year, month = month)
            dir_path_parquet = parquet_path.split('_')[0][:-2]
                
            df_parquet.to_parquet(parquet_path)

        if not as_npy:
            return df_parquet

        from .linearfitter import from_dataframe_to_recarray
        df_npy = from_dataframe_to_recarray(dataframe = df_parquet, year = year, filter_ = filter_,  month = month, store = store, **kwargs)
        return df_npy

        
    def build_dataframe(self, df, keys = None):
        """ Build the dataframe to perform the fit for one concat dataframe (columns selections)

        Parameters
        ----------

        df: [dataframe]
        concat dataframe file to be transformed into a usable one for the fit

        Returns
        -------
        pandas dataframe with wanted columns for the fit
        """

        basic_keys = ['Source','x', 'y', 'u', 'v', 'x_ps1', 'y_ps1', 'x_psfcat', 'y_psfcat', 'u_ps1', 'v_ps1',
                'ra', 'dec', 'ra_ps1', 'dec_ps1', 'ra_psfcat', 'dec_psfcat',
                'f_10', 'f_10_e','f_10_f',
                'gmag', 'g_mag', 'e_gmag', 'g_magErr', 'rpmag', 'e_rpmag', 'bpmag', 'e_bpmag', 'colormag',
                'isolated']

        df_concat = pandas.read_parquet(df)

        info = df.split('_')
        ccdid = int(info[3].replace('c', ''))
        qid = int( info[5].replace('q', ''))

        dic = {}
        dic['ccdid'] = ccdid
        dic['qid'] = qid
        dic['rmag_ps1'] = df_concat['r_mag']
        dic['imag_ps1'] = df_concat['i_mag']
        dic['zmag_ps1'] = df_concat['z_mag']        
        dic['e_rmag_ps1'] = df_concat['r_magErr']
        dic['e_imag_ps1'] = df_concat['i_magErr']
        dic['e_zmag_ps1'] = df_concat['z_magErr']
        dic['psfcat'] = df_concat['flux']
        dic['psfcat_e'] = df_concat['sigflux']
        
        dataframe = pandas.DataFrame(dic, index = df_concat.index)
        dataframe[basic_keys] = df_concat[basic_keys].values
        rcid = fields.ccdid_qid_to_rcid(dataframe['ccdid'], dataframe['qid'])
        dataframe['rcid'] = rcid
        # In good order
        if keys is None:
            keys = ['Source', 'ccdid', 'qid', 'rcid',
                    'x','y', 'u', 'v', 'x_ps1', 'y_ps1', 'u_ps1', 'v_ps1',
                    'ra', 'dec','ra_ps1', 'dec_ps1', 'x_psfcat', 'y_psfcat', 'ra_psfcat', 'dec_psfcat',
                    'f_10', 'f_10_e', 'f_10_f', 'psfcat', 'psfcat_e',
                    'gmag', 'g_mag','e_gmag', 'g_magErr',
                    'rpmag','rmag_ps1', 'e_rpmag', 'e_rmag_ps1',
                    'bpmag', 'imag_ps1','e_bpmag','e_imag_ps1',
                    'zmag_ps1', 'e_zmag_ps1',
                    'colormag', 'isolated']
            
        ordered_dataframe = dataframe[keys]

        return ordered_dataframe


        
    @property
    def scimgs(self):
        """ """
        return self._scimgs

    @property
    def datafile(self):
        """ """
        if hasattr(self,"_datafile"):
            return self._datafile
        return self.get_datafile()

    @property
    def filenames(self):
        """ """
        if hasattr(self,"_filenames"):
            return self._filenames

        return [scimg.filename for scimg in self.scimgs]

    @property
    def filtername(self):
        """ """
        if hasattr(self,"_filtername"):
            return self._filtername

        return self.scimgs[0].filtername
    
    @property
    def year(self):
        """ """
        if hasattr(self,"_year"):
            return self._year

            return self.scimgs[0].year
    
    @property
    def month(self):
        """ """
        if hasattr(self,"_month"):
            return self._month

            return self.scimgs[0].month

#############################
#
#
#
##############################



#import starflat_nicolas_table.py as nr_sf
