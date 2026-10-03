import astropy.constants as     ac
from   astropy.time      import Time
import barycorrpy

def compute_berv_and_herv_corrections(obsname:str, date_obs:str, exp_time:float, photocen:float) -> tuple[float]:
    """Compute BERV (Barycentric-Earth RV) and HERV (Heliocentric RV) corrections.

    Parameters
    ----------
    obsname : str
        Observatory name from those available in astropy.coordinates.EarthLocation.get_site_names().
    date_obs : str
        Observation date in UTC for the start of the exposure.
    exp_time : float
        Exposure time in seconds.
    photocen : float
        Photometric center of exposure as a fraction of the total integration time.

    Returns
    -------
    tuple[float]
        BERV and HERV corrections.
    """

    # Unit conversions
    sec_to_day = 1/(60*60*24) # seconds to days
    mps_to_kmps = 1e-3        # m/s to km/s

    # Compute gravitational redshift of the Sun
    gr_sun = (ac.G.value * ac.M_sun.value) / (ac.R_sun.value * ac.c.value)

    # Compute exposure photometric center in JD UTC
    jd_utc_cen = Time(date_obs, format='isot').jd + exp_time * photocen * sec_to_day

    # Compute BERV and HERV
    berv = barycorrpy.get_BC_vel(JDUTC=jd_utc_cen, obsname=obsname, SolSystemTarget='Sun'                 )[0][0]
    herv = barycorrpy.get_BC_vel(JDUTC=jd_utc_cen, obsname=obsname, SolSystemTarget='Sun', predictive=True)[0][0]*(-1) - berv + gr_sun

    # Convert from m/s to km/s
    berv *= mps_to_kmps
    herv *= mps_to_kmps

    return berv, herv