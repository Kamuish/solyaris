import numpy    as     np
import pandas   as     pd
import pickle
from   tqdm     import tqdm
import warnings
warnings.filterwarnings('ignore')

from typing import Literal

from .compute_berv_and_herv_corrections                     import compute_berv_and_herv_corrections
from .compute_differential_extinction_and_vsini_corrections import compute_differential_extinction_and_vsini_corrections
from .compute_extinction_and_quality                        import compute_extinction_and_quality
from .compute_limb_darkening_coefficient                    import compute_limb_darkening_coefficient
from .compute_solar_coordinates                             import compute_solar_coordinates
from .extract_header_keywords                               import extract_header_keywords
from .extract_order_by_order_rv                             import extract_order_by_order_rv

class SOLYARIS:

    def __init__(self, instrument:Literal['espresso', 'expres', 'harps', 'harps-n', 'kpf, ''neid', 'nirps'], id:str=None, verbose:bool=True):
        """Create SOLYARIS object.

        Parameters
        ----------
        instrument : Literal['espresso', 'expres', 'harps', 'harps-n', 'kpf', 'neid', 'nirps']
            Instrument name.
        id : str, optional
            SOLYARIS object ID, used as the name when saving the object as a pickled file or the data table as a CSV file. By default the instrument name.
        verbose : bool, optional
            Whether to print the name of the steps and progress bars. By default True.
        """

        # Instrument and SOLYARIS object ID
        self.instrument = instrument
        if id is not None:
            self.id = id
        else:
            self.id = instrument

        # Verbose
        self.verbose = verbose

        # ESPRESSO
        if self.instrument == 'espresso':
            self.Norder     = 170
            self.Npixel     = 9111
            self.obscode    = 309
            self.obsname    = 'Paranal Observatory'

        # EXPRES
        if self.instrument == 'expres':
            self.Norder     = 0
            self.Npixel     = 0
            self.obscode    = 0
            self.obsname    = 'X'

        # HARPS
        if self.instrument == 'harps':
            self.Norder     = 0
            self.Npixel     = 0
            self.obscode    = 0
            self.obsname    = 'X'

        # HARPS-N
        if self.instrument == 'harps-n':
            self.Norder     = 69
            self.Npixel     = 4096
            self.obscode    = 950
            self.obsname    = 'Roque de los Muchachos'

        # KPF
        if self.instrument == 'kpf':
            self.Norder     = 0
            self.Npixel     = 0
            self.obscode    = 0
            self.obsname    = 'X'

        # NEID
        if self.instrument == 'neid':
            self.Norder     = 122
            self.Npixel     = 9216
            self.obscode    = 695
            self.obsname    = 'Kitt Peak National Observatory'

        # NIRPS
        if self.instrument == 'nirps':
            self.Norder     = 0
            self.Npixel     = 0
            self.obscode    = 0
            self.obsname    = 'X'

    def add_data(self, files:list[str]=None):
        """Add data.

        Parameters
        ----------
        files : list[str], optional
            List of paths to FITS files. Must be CCF_A files (ESPRESSO, HARPS, HARPS-N, NIRPS) or L2 files (EXPRES, NEID). By default None.

        Returns
        -------
        None
            Saves files in self.files and nr. of files in self.Nfile.
        """

        # Create 'files' and 'Nfile' properties
        self.files = files
        self.Nfile = len(files)

        return None

    def extract_header_keywords(self):
        """Extract header keywords.

        Returns
        -------
        None
            Saves a pandas.DataFrame with the extracted header keywords in self.table.
        """

        # Print action
        if self.verbose:
            print('Extracting header keywords.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Loop files
        for i in iterable:

            # Extract header keywords
            header_dict = extract_header_keywords(self.files[i], self.instrument, self.Norder)

            # Initiate DataFrame
            if i == 0:
                df = pd.DataFrame(columns=header_dict.keys())

            # Populate DataFrame
            df.loc[i] = header_dict
        
        # Create 'table' property
        self.table = df

        return None

    def extract_order_by_order_rv(self):
        """Extract order-by-order RV.

        Returns
        -------
        None
            Saves order-by-order RV values and errors as 'vrad_val_ORDER' and 'vrad_err_ORDER' columns in self.table,
            where ORDER is the order index starting from 1.
            Saves the summed CCF RV values and errors as 'vrad_val_sum' and 'vrad_err_sum'.
        """

        # Print action
        if self.verbose:
            print('Extracting order-by-order RV.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Empty arrays
        vrad_val = np.empty((self.Nfile, self.Norder+1), dtype=float)
        vrad_err = np.empty((self.Nfile, self.Norder+1), dtype=float)

        # Loop files
        for i in iterable:

            # Extract order-by-order RV
            vrad_val[i], vrad_err[i] = extract_order_by_order_rv(self.files[i])

        # Loop orders
        for i in range(self.Norder):

            # Insert output columns
            self.table[f'vrad_val_{i+1}'] = vrad_val[:,i]
            self.table[f'vrad_err_{i+1}'] = vrad_err[:,i]

        # Insert output columns
        self.table['vrad_val_sum'] = vrad_val[:,self.Norder]
        self.table['vrad_err_sum'] = vrad_err[:,self.Norder]

        return None

    def compute_berv_and_herv_corrections(self):
        """Compute BERV (Barycentric-Earth RV) and HERV (Heliocentric RV) corrections.

        Returns
        -------
        None
            Saves BERV and HERV corrections as 'berv_val' and 'herv_val' columns in self.table. Renames BERV correction from header as 'berv_drs'.
        """

        # Print action
        if self.verbose:
            print('Computing BERV and HERV corrections.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Empty arrays
        berv_val = np.empty(self.Nfile, dtype=float)
        herv_val = np.empty(self.Nfile, dtype=float)

        # Loop files
        for i in iterable:

            # Compute BERV and HERV corrections
            berv_val[i], herv_val[i] = compute_berv_and_herv_corrections(self.obsname, self.table.date_obs.values[i], self.table.exp_time.values[i], self.table.photocen.values[i])

        # Rename header columns
        self.table.rename(columns={'berv_val': 'berv_drs'}, inplace=True)

        # Insert output columns
        self.table.insert(self.table.columns.get_loc('berv_drs')+1, 'berv_val', berv_val)
        self.table.insert(self.table.columns.get_loc('berv_drs')+2, 'herv_val', herv_val)

        return None

    def compute_solar_coordinates(self):
        """Compute solar coordinates.

        Returns
        -------
        None
            Saves RA, Dec and airmass as 'alph_val', 'delt_val' and 'airm_val' columns in self.table. Renames corresponding variables from header as 'alph_drs', 'delt_drs' and 'airm_drs'. 
        """

        # Print action
        if self.verbose:
            print('Computing solar coordinates.')
            print(f'-> Processing {self.Nfile} files ...')

        # Iterable
        iterable = tqdm(range(self.Nfile)) if self.verbose else range(self.Nfile)

        # Empty arrays
        alph_val = np.empty(self.Nfile, dtype=float)
        delt_val = np.empty(self.Nfile, dtype=float)
        airm_val = np.empty(self.Nfile, dtype=float)

        # Loop files
        for i in iterable:

            # Compute solar coordinates
            alph_val[i], delt_val[i], airm_val[i] = compute_solar_coordinates(self.obsname, self.table.date_obs.values[i], self.table.exp_time.values[i], self.table.photocen.values[i])

        # Rename header columns
        self.table.rename(columns={'alph_val': 'alph_drs'}, inplace=True)
        self.table.rename(columns={'delt_val': 'delt_drs'}, inplace=True)
        self.table.rename(columns={'airm_val': 'airm_drs'}, inplace=True)

        # Insert output columns
        self.table.insert(self.table.columns.get_loc('alph_drs')+1, 'alph_val', alph_val)
        self.table.insert(self.table.columns.get_loc('delt_drs')+1, 'delt_val', delt_val)
        self.table.insert(self.table.columns.get_loc('airm_drs')+1, 'airm_val', airm_val)

        return None

    def compute_extinction_and_quality(self, mode:Literal['all', 'one']='all', ref_order:int=None, Npt_min:int=10, plot_results:bool=False):
        """Compute extinction and quality.

        Parameters
        ----------
        mode : Literal['all', 'one'], optional
            Whether to process all orders ('all') or only one reference order ('one'), specified by the ref_order keyword. By default 'all'.
        ref_order : int, optional
            Reference order index (starting from 1) used if mode is 'one'. By default None.
        Npt_min : int, optional
            Minimum nr. of points required for a day to be processed. By default 10.
        plot_results : bool, optional
            Whether to plot the results. By default False.

        Returns
        -------
        None
            Saves extinction, quality and associated MCMC parameters as 'extinction', 'quality', 'c0', 'c0_hat', 'c1', 'jit', 'mu', 'mu_hat', 'sig' and 'Q' columns in self.table,
            with suffix '_ORDER' where ORDER is the ORDER index starting from 1.
        """

        # Print action
        if self.verbose:
            print('Computing extinction and quality.')

        # Nr. of orders in total
        Norder = self.Norder

        # Orders for mode with all orders
        if mode == 'all': orders = np.arange(1,Norder+1)

        # Orders for mode with one order
        if mode == 'one': orders = np.array([ref_order])

        # Nr. of orders to iterate
        Norder = len(orders)

        # Loop orders
        for order in orders:

            # For ESPRESSO, only process every other order
            if (self.instrument == 'espresso') & (mode == 'all'):

                # Skip orders with odd index
                if (order%2) == 1:

                    # Print action
                    if self.verbose:
                        print(f'-> Skipping order {order} out of {Norder} ...')

                    # Skip iteration
                    continue

            # Print action
            if self.verbose & (mode == 'all'):
                print(f'-> Processing order {order} out of {Norder} ...')
            if self.verbose & (mode == 'one'):
                print(f'-> Processing order {order} ...')

            # Empty arrays
            c0         = np.empty(self.Nfile, dtype=float)
            c0_hat     = np.empty(self.Nfile, dtype=float)
            c1         = np.empty(self.Nfile, dtype=float)
            jit        = np.empty(self.Nfile, dtype=float)
            mu         = np.empty(self.Nfile, dtype=float)
            mu_hat     = np.empty(self.Nfile, dtype=float)
            sig        = np.empty(self.Nfile, dtype=float)
            Q          = np.empty(self.Nfile, dtype=float)
            extinction = np.empty(self.Nfile, dtype=float)
            quality    = np.empty(self.Nfile, dtype=float)

            # Extract and compute required variables
            time_jdb = self.table[ 'time_jdb'   ].to_numpy(copy=True)
            airm_val = self.table[ 'airm_val'   ].to_numpy(copy=True)
            snrx_val = self.table[f'snr_{order}'].to_numpy(copy=True)
            time_jdn = np.floor(time_jdb + 0.5).astype(int)

            # Instrument modes
            ins_modes = np.unique(self.table.ins_mode.values)

            # Loop instrument modes
            for ins_mode in ins_modes:

                # Index of mode
                i_mode = self.table.ins_mode.values == ins_mode

                # Days of mode
                days = np.unique(time_jdn[i_mode])
                Nday = len(days)

                # Nr. of files of mode
                Nfile = np.sum(i_mode)

                # Print action
                if self.verbose:
                    print(f'--> Processing {Nfile} files over {Nday} days for instrument mode {ins_mode} ...')

                # Iterable
                iterable = tqdm(range(Nday)) if self.verbose else range(Nday)

                # Loop days
                for i in iterable:

                    # Index of day
                    i_day = time_jdn == days[i]

                    # Total index
                    ii = i_mode & i_day

                    # Compute quality
                    c0[ii], c0_hat[ii], c1[ii], jit[ii], mu[ii], mu_hat[ii], sig[ii], Q[ii], extinction[ii], quality[ii] = compute_extinction_and_quality(time_jdb[ii], time_jdn[ii], airm_val[ii], snrx_val[ii], Npt_min, plot_results, ins_mode, order)

            # Insert output columns
            self.table[f'c0_{order}'        ] = c0
            self.table[f'c0_hat_{order}'    ] = c0_hat
            self.table[f'c1_{order}'        ] = c1
            self.table[f'jit_{order}'       ] = jit
            self.table[f'mu_{order}'        ] = mu
            self.table[f'mu_hat_{order}'    ] = mu_hat
            self.table[f'sig_{order}'       ] = sig
            self.table[f'Q_{order}'         ] = Q
            self.table[f'extinction_{order}'] = extinction
            self.table[f'quality_{order}'   ] = quality

        return None

    def compute_differential_extinction_and_vsini_corrections(self, mode:Literal['all', 'one']='all', ref_order:int=None):
        """Compute differential extinction and vsini corrections.

        Parameters
        ----------
        mode : Literal['all', 'one'], optional
            Whether to process all orders ('all') or only one reference order ('one'), specified by the ref_order keyword. By default 'all'.
        ref_order : int, optional
            Reference order index (starting from 1) used if mode is 'one'. By default None.

        Returns
        -------
        None
            Saves differential extinction correction as 'corr_extinction_ORDER' columns in self.table,
            where ORDER is the order index starting from 1.
            Saves vsini correction as 'corr_vsini' column in self.table.
        """

        # Print action
        if self.verbose:
           print('Computing differential extinction and vsini corrections.')

        # Nr. of files
        Nfile = self.Nfile

        # Nr. of orders in total
        Norder = self.Norder

        # Orders for mode with all orders
        if mode == 'all': orders = np.arange(1,Norder+1)

        # Orders for mode with one order
        if mode == 'one': orders = np.array([ref_order])

        # Nr. of orders to iterate
        Norder = len(orders)

        # Extract Barycentric Julian Date
        time_jdb = self.table.time_jdb.to_numpy(copy=True)

        # Empty array
        extinction = np.empty((Norder,Nfile), dtype=float)

        # Loop orders
        for i, order in enumerate(orders):

            # Extract extinction
            # For ESPRESSO, duplicate extinction of orders with even order index
            if (self.instrument == 'espresso') & (mode == 'all'):
                extinction[i] = self.table[f'extinction_{((order // 2) + (order % 2)) * 2}'].to_numpy(copy=True)
            else:
                extinction[i] = self.table[f'extinction_{order}'].to_numpy(copy=True)

        # Compute limb darkening coefficient(s)
        ldc = compute_limb_darkening_coefficient(self.instrument, mode, ref_order)

        # Compute differential extinction and vsini corrections
        hour_angle, corr_extinction, corr_vsini = compute_differential_extinction_and_vsini_corrections(self.obsname, self.obscode, time_jdb, extinction, ldc)

        # Insert output columns
        self.table.insert(self.table.columns.get_loc('airm_val')+1, 'hour_angle', hour_angle)
        self.table.insert(self.table.columns.get_loc('fwhm_err')+1, 'corr_vsini', corr_vsini)

        # Loop orders
        for i, order in enumerate(orders):

            # Insert output columns
            self.table.insert(self.table.columns.get_loc(f'vrad_err_{order}')+1, f'corr_extinction_{order}', corr_extinction[i])

        return None

    def export_table(self):
        """Export table.

        Returns
        -------
        None
            Saves self.table as a CSV file with filename self.id.
        """

        self.table.to_csv(self.id+'.csv', index=False)

        return None

def save(solyaris):
    """Save SOLYARIS object as a pickled file.

    Parameters
    ----------
    solyaris : SOLYARIS object
        SOLYARIS object to be saved as a pickled file.

    Returns
    -------
    None
        Saves SOLYARIS object as a pickled file with filename self.id and extension .solyaris.
    """

    return pickle.dump(solyaris, open(solyaris.id+'.solyaris', 'wb'))

def load(solyaris):
    """Load SOLYARIS object from a pickled file.

    Parameters
    ----------
    solyaris : SOLYARIS file
        SOLYARIS pickled file to be loaded.

    Returns
    -------
    SOLYARIS object
        SOLYARIS object loaded from the SOLYARIS file.
    """

    return pickle.load(open(solyaris, 'rb'))