import iCCF
import numpy as np

from .extract_ccf import extract_ccf

def extract_order_by_order_rv(file_path:str) -> tuple[np.array]:
    """Extract order-by-order RV.

    Parameters
    ----------
    file_path : str
        Path to FITS file.

    Returns
    -------
    tuple[np.array]
        Order-by-order RV values and errors. The last index contains the order-summed RV value and error.
    """

    # Extract order-by-order CCF
    vgrid, ccf_val, ccf_err = extract_ccf(file_path)

    # Nr. of orders
    Norder = ccf_val.shape[0]

    # NaN arrays
    vrad_val = np.empty(Norder, dtype=float)*np.nan
    vrad_err = np.empty(Norder, dtype=float)*np.nan

    # Loop orders
    for i in range(Norder):

        # Check that CCF is finite
        if not np.all(np.isfinite(ccf_val[i]) & np.isfinite(ccf_err[i])):
            continue

        # Extract order-by-order RV
        iccf = iCCF.Indicators(vgrid, ccf_val[i], ccf_err[i])
        vrad_val[i] = iccf.RV
        vrad_err[i] = iccf.RVerror

    return vrad_val, vrad_err