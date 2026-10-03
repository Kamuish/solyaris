from   astropy.io        import fits
from   bisect            import bisect_left
from   iCCF              import Mask
import numpy             as     np
from   PyAstronomy       import pyasl
from   scipy.interpolate import interp1d

def extract_ccf(path:str) -> tuple[np.array]:
    """Extract Cross-Correlation Function (CCF).

    Parameters
    ----------
    path : str
        Path to FITS file. Must be a CCF_A file (ESPRESSO, HARPS, HARPS-N, NIRPS) or an L2 file (EXPRES, KPF, NEID).

    Returns
    -------
    tuple[np.array]
        Velocity grid, CCF values and CCF errors.
    """

    # Instrument
    file = path.split('/')[-1]
    if file.startswith('r.ESPRE'):
        instrument = 'espresso'
    if file.startswith('r.HARPS'):
        instrument = 'harps'
    if file.startswith('r.HARPN'):
        instrument = 'harps-n'
    if file.startswith('r.NIRPS'):
        instrument = 'nirps'
    if file.startswith('expres'):
        instrument = 'expres'
    if file.startswith('KP'):
        instrument = 'kpf'
    if file.startswith('neid'):
        instrument = 'neid'

    # ESPRESSO, HARPS, HARPS-N, NIRPS
    if (instrument == 'espresso') | (instrument == 'harps') | (instrument == 'harps-n') | (instrument == 'nirps'):

        # Organization listed after 'HIERARCH' in header keywords
        if (instrument == 'espresso') | (instrument == 'harps') | (instrument == 'nirps'):
            ORG = 'ESO'
        if (instrument == 'harps-n'):
            ORG = 'TNG'

        # Load FITS file
        with fits.open(path) as hdul:

            # Header
            header = hdul[0].header

            # Extract CCF
            ccf_val = hdul[1].data
            ccf_err = hdul[2].data
            vstart  = header[f'HIERARCH {ORG} RV START']
            vstep   = header[f'HIERARCH {ORG} RV STEP' ]
            vgrid   = vstart + vstep * np.arange(ccf_val.shape[1])

    # EXPRES
    if instrument == 'expres':

        # Load FITS file
        with fits.open(path) as hdul:

            # Nr. of orders
            Norder = hdul[2].data.shape[0]

            # Velocity grid
            vgrid = np.array(hdul[1].data['V_grid'], dtype=float)
            Ngrid = len(vgrid)

            # NaN arrays
            ccf_val = np.empty((Norder+1,Ngrid))*np.nan
            ccf_err = np.empty((Norder+1,Ngrid))*np.nan

            # Loop orders
            for i in range(Norder):

                # Extract order-by-order CCF
                ccf_val[i] = hdul[2].data[i][1]
                ccf_val[i] = hdul[2].data[i][2]

            # Extract order-summed CCF
            ccf_val[-1] = hdul[1].data['ccf']
            ccf_err[-1] = hdul[1].data['e_ccf']

    # KPF
    if instrument == 'kpf':

        # TODO
        return None

    # NEID
    if instrument == 'neid':

        # Load FITS file
        with fits.open(path) as hdul:

            # Nr. of orders
            Norder = hdul[1].data.shape[0]

            # Velocity grid
            vstart = -20.
            vstop  =  20.
            vstep  =   1.
            vgrid  = np.arange(vstart, vstop+vstep/2, vstep)
            Ngrid  = len(vgrid)

            # Extract S2D spectrum
            ll      = hdul[ 7].data
            flux    = hdul[ 1].data
            error   = hdul[ 4].data ** (1/2)
            blaze   = hdul[15].data
            RV_table = vgrid
            mask = Mask('G2', 'ESPRESSO')
            berv = np.empty(Norder)
            for i in range(Norder):
                berv[i] = hdul[0].header[f'SSBRV{52+i:0>3}']
            bervmax = 32.
            mask_width = vstep

            # Quality
            quality = np.zeros_like(ll)

            # Compute dll
            dll = np.gradient(ll, axis=1)

            # NaN arrays
            ccf_val = np.empty((Norder+1,Ngrid))*np.nan
            ccf_err = np.empty((Norder+1,Ngrid))*np.nan

            # Loop orders
            for i in range(Norder):

                # Check that all wavelengths are valid
                if not np.all(ll[i] > 0):
                    continue

                # Vacuum-to-air transformation and BERV-correction of wavelengths
                c = 299792.458
                ll[i]  = pyasl.vactoair2(ll[i])
                ll[i] *= (1. + berv[i]/c)

                # Interpolate NaN pixels
                idx_val = np.isfinite(flux[i]) & np.isfinite(error[i]) & np.isfinite(blaze[i])
                idx_nan = ~idx_val
                flux [i,idx_nan] = interp1d(ll[i,idx_val], flux [i,idx_val], kind='linear', assume_sorted=True, bounds_error=False)(ll[i,idx_nan])
                error[i,idx_nan] = interp1d(ll[i,idx_val], error[i,idx_val], kind='linear', assume_sorted=True, bounds_error=False)(ll[i,idx_nan])
                blaze[i,idx_nan] = interp1d(ll[i,idx_val], blaze[i,idx_val], kind='linear', assume_sorted=True, bounds_error=False)(ll[i,idx_nan])

                # Compute CCF
                ccf_val[i], ccf_err[i], _ = espdr_compute_CCF_fast(ll[i], dll[i], flux[i], error[i], blaze[i], quality[i], RV_table, mask, berv[i], bervmax, mask_width)

                # Weight CCF
                ccf_weight = hdul[12].header[f'CCFWT{52+i:0>3}']
                if ccf_weight is None:
                    ccf_weight = np.nan
                ccf_val[i] *= ccf_weight
                ccf_err[i] *= ccf_weight

            # Combine CCF
            ccf_val[-1] = np.nansum(ccf_val[:-1], axis=0)
            ccf_err[-1] = np.sqrt(np.nansum(ccf_err[:-1]**2, axis=0))

    return vgrid, ccf_val, ccf_err

# Copied from iCCF
# https://github.com/j-faria/iCCF/blob/main/iCCF/meta.py
def espdr_compute_CCF_fast(ll, dll, flux, error, blaze, quality, RV_table, mask, berv, bervmax, mask_width=0.5):

    c = 299792.458

    nx_s2d = flux.size
    # ny_s2d = 1  #! since this function computes only one order
    n_mask = mask.size
    nx_ccf = len(RV_table)

    ccf_flux = np.zeros_like(RV_table)
    ccf_error = np.zeros_like(RV_table)
    ccf_quality = np.zeros_like(RV_table)

    dll2 = dll / 2.0  # cpl_image_divide_scalar_create(dll,2.);
    ll2 = ll - dll2  # cpl_image_subtract_create(ll,dll2);

    #? this mimics the pipeline (note that cpl_image_get indices start at 1)
    imin, imax = 1, nx_s2d
    while(imin < nx_s2d and quality[imin-1] != 0):
        imin += 1
    while(imax > 1 and quality[imax-1] != 0):
        imax -= 1

    if imin >= imax:
        return
    #? note that cpl_image_get indices start at 1, hence the "-1"s
    llmin = ll[imin + 1 - 1] / (1. + berv / c) * (1. + bervmax / c) / (1. + RV_table[0] / c)
    llmax = ll[imax - 1 - 1] / (1. + berv / c) * (1. - bervmax / c) / (1. + RV_table[nx_ccf - 1] / c)

    imin, imax = 0, n_mask - 1

    #? turns out cpl_table_get indices start at 0...
    while (imin < n_mask and mask['lambda'][imin] < (llmin + 0.5 * mask_width / c * llmin)):
        imin += 1
    while (imax >= 0     and mask['lambda'][imax] > (llmax - 0.5 * mask_width / c * llmax)):
        imax -= 1

    for i in range(imin, imax + 1):
        #? cpl_array_get indices also start at 0
        llcenter = mask['lambda'][i] * (1. + RV_table[nx_ccf // 2] / c)

        # index_center = 1
        # while(ll[index_center-1] < llcenter): index_center += 1
        # my attempt to speed it up
        # index_center = np.where(ll < llcenter)[0][-1] + 1
        index_center = bisect_left(ll, llcenter) + 1

        contrast = mask['contrast'][i]
        w = contrast * contrast

        for j in range(0, nx_ccf):
            llcenter = mask['lambda'][i] * (1. + RV_table[j] / c)
            llstart = llcenter - 0.5 * mask_width / c * llcenter
            llstop = llcenter + 0.5 * mask_width / c * llcenter

            # index1 = 1
            # while(ll2[index1-1] < llstart): index1 += 1
            index1 = bisect_left(ll2, llstart) + 1

            # index2 = index1
            # while (ll2[index2-1] < llcenter): index2 += 1
            index2 = bisect_left(ll2, llcenter) + 1

            # index3 = index2
            # while (ll2[index3-1] < llstop): index3 += 1;
            index3 = bisect_left(ll2, llstop) + 1

            k = j

            for index in range(index1, index3):
                ccf_flux[k] += w * flux[index-1] / blaze[index-1] * blaze[index_center-1]  # noqa: E501

            ccf_flux[k] += w * flux[index1-1-1] * (ll2[index1-1]-llstart) / dll[index1-1-1] / blaze[index1-1-1] * blaze[index_center-1]
            ccf_flux[k] -= w * flux[index3-1-1] * (ll2[index3-1]-llstop) / dll[index3-1-1] / blaze[index3-1-1] * blaze[index_center-1]

            ccf_error[k] += w * w * error[index2 - 1 - 1] * error[index2 - 1 - 1]

            ccf_quality[k] += quality[index2 - 1 - 1]

    # my_error = cpl_image_power(*CCF_error_RE,0.5);
    ccf_error = np.sqrt(ccf_error)

    return ccf_flux, ccf_error, ccf_quality