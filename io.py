
import os
import numpy as np
import pandas
from glob import glob

#__all__ = ["get_starflat_files"]
STARFLAT_PATH = "/sps/ztf/data/storage/starflat/"
DATAFILE_PATH = os.path.join(STARFLAT_PATH, "datafiles")
DIRPATH_CONCAT = os.path.join(os.getenv('ZTFDATA'), 'storage/starflat/concat_df')


DIRPATH_STARFLAT_FILES = '/sps/ztf/data/storage/starflat/processed_starflats.csv'
DIRPATH_STARFLAT_FILES_2021 =  '/sps/ztf/data/storage/starflat/starflat_nov2021.txt'
def get_starflat_files(filter_, year, month=None):
    """ Get starflat sciimg.fits regarding year and filter

    Parameters
    ----------
    
    file_path: [str]
    where the starflat sciimg.fits are
    
    filter: [str]
    -zg
    -zr
    -zi
    
    year: [int]
    -2018
    -2019
    -2021
    
    Returns
    -------
    list of wanted sciimg.fits
    """
    
    filters = {'zg': 1, 'zr':2, 'zi':3}
    filter_chosen = filters[filter_]

    if ((year == 2019) or (year == 2018)):
        file_path = DIRPATH_STARFLAT_FILES
        files = pandas.read_csv(file_path)
        dataframe = files.loc[files['fid'] == filter_chosen]
        if year == 2019:
            dataframe_reduced = dataframe.loc[dataframe['filename'].str.startswith('/ztf/archive/sci/2019')]
        elif year == 2018:
            dataframe_reduced = dataframe.loc[dataframe['filename'].str.startswith('/ztf/archive/sci/2018')]
        else:
            raise ValueError(f"year {year} no available")

        filenames = list(dataframe_reduced['filename'])
        return filenames
        
    elif year == 2021:
        file_path = DIRPATH_STARFLAT_FILES_2021
        files = open(file_path).read().splitlines()
        zr_filenames = []
        zg_filenames = []
        for l in files:
            if '_zg_' in l:
                zg_filenames.append(l)
            else:
                zr_filenames.append(l)

        if filter_ == 'zg':
            return zg_filenames

        elif filter_ == 'zr':
            return zr_filenames
        else:
            raise ValueError(f"filter {filter_} not implemented only zg and zr")


DIRPATH_DATAFILE = '/sps/ztf/data/storage/starflat/datafiles'
def get_rcid_starflat_datafile(rcid, filter_, year, month):
    """ """
    dir_path = DIRPATH_DATAFILE
    store_path = os.path.join(dir_path, f'{year}', f'{month:02}')

    return os.path.join(store_path, f'starflat_{year}{month:02}_{filter_}_rcid{rcid}.h5')


def get_concat_df(filter_ = 'zg', year = 2019, month = 3):
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
    -2021

    month: [int]

    Returns
    -------
    list of all concat dataframes files
    """

    dfs = []
    dir_path = os.path.join(DIRPATH_CONCAT, f'{year}/{month:02}')

    if year not in [2018,2019,2021]:
        raise ValueError('No such year')

    if filter_ not in ['zi','zr','zg']:
        raise ValueError('No  such filter')

    if ((year == 2018) & (filter_ in['zr','zi'])):
        raise ValueError('No such filter+year combinaison')

    dfs = glob( os.path.join(dir_path, f"{year}{month:02}_{filter_}*_apconcat.parquet"))

    return dfs

DIRPATH_RECARRAY = '/sps/ztf/users/erobert/'
def get_recarray_path(filter_ = 'zg', year = 2019, month =3):
    dir_path = os.path.join(DIRPATH_RECARRAY, f'{year}', f'{month:02}')
    recarray_path = os.path.join(dir_path, f'df_fit_{year}{month:02}_{filter_}_concat.npy')
    
    return recarray_path


DIRPATH_PARQUET = '/sps/ztf/users/erobert/'
def get_parquet_path(filter_ = 'zg', year = 2019, month =3):
    dir_path = os.path.join(DIRPATH_PARQUET, f'{year}', f'{month:02}')
    parquet_path = os.path.join(dir_path, f'df_{year}{month:02}_{filter_}_concat.parquet')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    return parquet_path







# ==================== #
#
#  IO for FITTER       #
#
# ==================== #

DIRPATH_FITS = '/sps/ztf/data/ztfin2p3/cal/starflat/'
def get_fits_path(file_, ccdid = None, qid = None, flux_estimator = 'ap'):
    """ """
    year = int(file_.split('_')[2][:4])
    month = int(file_.split('_')[2][4:6])
    filter_ = file_.split('_')[3]

    dir_path = os.path.join(DIRPATH_FITS, f'{year}_ap_new_radius', f'{month:02}')
    if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
    dir_path_fits = os.path.join(dir_path, f'ztfin2p3_{year}{month:02}_000000_{filter_}_c{ccdid:02}_q{qid}_{flux_estimator}starflat.fits')
    
    return dir_path_fits


DIRPATH_FITRESULT = '/sps/ztf/users/erobert/starfits/'
def get_path_fitresult(file_, superpix, extension = 'parquet', flux_estimator = 'ap'):
    year = int(file_.split('_')[2][:4])
    month = int(file_.split('_')[2][4:6])
    filter_ = file_.split('_')[3]
    name = f'starflat_{year}{month:02}_{filter_}_sup{superpix}_{flux_estimator}.npy'

    dir_path = os.path.join(DIRPATH_FITRESULT, f'{year}', f'{month:02}')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    if extension == 'parquet':
        dir_path_file = os.path.join(dir_path, name.replace('.npy', '.parquet'))
    elif extension == 'npy':
        dir_path_file = os.path.join(dir_path, name)

    return dir_path_file
