""" Script to build concat dataframes if not built and to perform starflat analysis """

# This is dev
# this is test
import os
from glob import glob
import numpy as np
import pandas
import dask.dataframe as dd
from ztfimg import aperture
from ztfquery import fields
from ztfflat import analysis
import dask

STARFLAT_PATH = "/sps/ztf/data/storage/starflat/"
DATAFILE_PATH = os.path.join(STARFLAT_PATH, "datafiles")

class Starflat():

    def __init__(self):
        
        return None




    @staticmethod
    def get_rcid_datafile(rcid, filter = 'zg', year = 2018,  load=True):
        """ Get datafile.h5 for a rcid value, a filter and a year from starflat files in STARFLAT_PATH

        Parameters
        ----------

        rcid: [int]
        rcid value of interest

        filter: [str]
        -zg
        -zr
        -zi

        year: [int]
        -2018
        -2019

        load: [bool]
        if True load missing files

        Returns
        -------
        file.parquet
        """

        if year == 2018:
            y = '20180221'
        elif year == 2019:
            y = '20190331'
        else:
            raise ValueError('no such year')
        
        if ((year == 2018) & (filter in['zr','zi'])):
            rcfile = []
            raise ValueError('No such filter+year combinaison')
        else:
            rcfile = os.path.join(DATAFILE_PATH, f"starflat_{y}_{filter}_rcid{rcid}.h5")

        if load:
            return pandas.read_hdf(rcfile)

        return rcfile



        

    def build_concat_cat(self, radius, rcids = None, filter = 'zi', year = 2019,  sep_limit = 20, store = True):
        """ For each rcid, build a concat dataframe for all fits files 
        
        Parameters
        ----------

        radius: [list of float]
        aperture radius (same for all images)

        rcids: [int]
        rcids values of interest

        filter: [str]
        -zg
        -zr
        -zi

        year: [int]
        -2018
        -2019

        sep_limit: [int] -optional-
        no neighbors in the sep_limit radius around the considered star (arcsec)

        store: [bool]
        if store: build pathname to store concat dataframe

        Returns
        -------
        list of concat pandas dataframes (delayed)
        if store: list of all concat dataframes (delayed), list of pathnames to store them        
        """
        
        dir_path = os.path.join(os.getenv('ZTFDATA'), 'storage/starflat/concat_df')
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        files = []   
        cats = []
        file_out_list = []
        if rcids is None:
            rcids = np.arange(0,64)
        else:
            rcids = np.atleast_1d(rcids)

        for rcid in rcids:
            file = self.get_rcid_datafile(rcid, filter = filter, year = year, load = False)
            ap = aperture.AperturePhotometry.from_datafile(file)
            cat = ap.build_apcatalog(radius, calibrators=['gaia', 'ps1'], extracat=['psfcat'], isolation = sep_limit)
            ccdid = fields.rcid_to_ccdid_qid(rcid)[0]
            qid = fields.rcid_to_ccdid_qid(rcid)[1]
            file_out = os.path.join(dir_path, f'{year}_{filter}_c{ccdid:02}_o_q{qid}_concat.parquet')
            cats.append(cat)
            if store:
                file_out_list.append(file_out)

        if store:
            return cats, file_out_list
        else:
            return cats



    def get_concat_df(self, filter = 'zg', year = 2019):
        """ Glob the wanted concat dataframes files

        Parameters
        ----------

        filter: [str]
        -zg
        -zr
        -zi

        year: [int]
        -2018
        -2019

        Returns
        -------
        list of all concat dataframes files
        """
        
        dfs = []
        if ((year != 2018) and (year != 2019)):
            raise ValueError('no such year')
        
        if filter not in ['zi','zr','zg']:
            raise ValueError('no such filter')

        if ((year == 2018) & (filter in['zr','zi'])):
            raise ValueError('No such filter+year combinaison')
        else:
            dfs = glob( os.path.join('/sps/ztf/data/storage/starflat/concat_df', f"{year}_{filter}*.parquet"))
        
        return dfs


    
    def build_dataframe(self, df):
        """ Build the dataframe to perform the fit for one concat dataframe

        Parameters
        ----------

        df: [dataframe]
        concat dataframe file to be transformed into a usable one for the fit
        
        Returns
        -------
        pandas dataframe usable to perform the fit
        """

        keys = ['Source','x', 'y', 'u', 'v', 'x_ps1', 'y_ps1', 'u_ps1', 'v_ps1', 'ra', 'dec', 'ra_ps1', 'dec_ps1',
        'f_10', 'f_10_e','f_10_f',
        'gmag', 'gmag_ps1', 'e_gmag', 'e_gmag_ps1', 'rpmag', 'e_rpmag', 'bpmag', 'e_bpmag', 'colormag',
       'isolated']

        df_concat = pandas.read_parquet(df)
        ap = analysis.ApertureCatalog.read_parquet(df)
        dataf = apOB.create_col_para(variable = 'both')
        info = df.split('_')
        ccdid = int(info[3].replace('c','')) 
        qid = int( info[5].replace('q', ''))
        dic = {}
        dic['ccdid'] = ccdid
        dic['qid'] = qid
    
        dic['Source'] = df_concat['Source']
        dic['rmag_ps1'] = df_concat['rmag']
        dic['imag_ps1'] = df_concat['imag']
        dic['zmag_ps1'] = df_concat['zmag']

        dic['e_rmag_ps1'] = df_concat['e_rmag']
        dic['e_imag_ps1'] = df_concat['e_imag']
        dic['e_zmag_ps1'] = df_concat['e_zmag']

        dic['psfcat'] = df_concat['flux']
        dic['psfcat_e'] = df_concat['sigflux']
        dic['psfcat_ratio'] = dataf['psfcat_ratio']
        dic['f_10_raOBtio'] = dataf['f_10_ratio']
    
        dataframe = pandas.DataFrame(dic, index = df_concat.index)
        dataframe[keys] = df_concat[keys].values
        # In good ordrer:
        dataframe_final = dataframe[['Source', 'ccdid', 'qid',
                                     'x','y', 'u', 'v', 'x_ps1', 'y_ps1', 'u_ps1', 'v_ps1', 'ra', 'dec','ra_ps1', 'dec_ps1',
                                     'f_10', 'f_10_e', 'f_10_ratio','f_10_f', 'psfcat', 'psfcat_e', 'psfcat_ratio', 
                                     'gmag', 'gmag_ps1','e_gmag', 'e_gmag_ps1',
                                     'rpmag','rmag_ps1', 'e_rpmag', 'e_rmag_ps1',
                                     'bpmag', 'imag_ps1','e_bpmag','e_imag_ps1',
                                     'zmag_ps1', 'e_zmag_ps1',
                                     'colormag', 'isolated']]
    
        return dataframe_final
    

    def build_dataframe_fit(self, client, year = 2019, filter = 'zi', store = True):
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

        store: [bool]
        if True, store the final dataframe

        Returns
        -------
        usable dataframe (focal plan) for the fit 
        """
        dfs = self.get_concat_df(filter = filter, year = year)
        df_list = []
        for df in dfs:
            dataframe_final = dask.delayed(self.build_dataframe)(df)
            df_list.append(dataframe_final)
        
        df = pandas.concat(client.gather(client.compute(df_list)))
        if store is True:
            dir_path = '/pbs/home/e/erobert/libraries/ztfflat'
            df.to_parquet(os.path.join(dir_path, f'df_matched_{year}_{filter}.parquet'))

        return df
