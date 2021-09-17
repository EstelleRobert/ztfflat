""" Script to build concat dataframes if not built and to perform starflat analysis """

# This is dev
# this is test
import os
import glob
import numpy as np
import pandas
import dask.dataframe as dd
from ztfimg import aperture
from ztfquery import fields

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
