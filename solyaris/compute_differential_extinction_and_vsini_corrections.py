from   astropy.coordinates import EarthLocation
from   astropy.time        import Time
import numpy               as     np
import subprocess

def compute_differential_extinction_and_vsini_corrections(obsname, obscode, time_jdb, extinction, ldc):

    ### UNITS & CONSTANTS

    # Unit conversions
    deg_to_arcsec = 3600.0                  # degrees to arcseconds
    deg_to_rad    = np.pi/180               # degrees to radians
    d_to_s        = 24.0*60.0*60.0          # days    to seconds
    yr_to_s       = 365.256*d_to_s          # years   to seconds
    AU_to_km      = 149597870.7             # AUs     to kilometers

    # Constants
    c             = 299792.458              # [km/s]     speed of light
    eccearth      = 0.0167                  # [unitless] Earth orbit eccentricity
    epsearth      = 23.439291111*deg_to_rad # [rad]      Earth ecliptic obliquity
    sunpolera     = 286.13*deg_to_rad       # [rad]      solar rotation pole RA
    sunpoledec    = 63.87*deg_to_rad        # [rad]      solar rotation pole DEC
    protsun       = 25.38*d_to_s            # [s]        solar rotation period
    rsun          = 696000.0                # [km]       solar radius
    ASU           = 2.9722e-6               # Snodgrass & Ulrich (1990) sidereal differential rotation law
    BSU           = -0.484e-6               # Snodgrass & Ulrich (1990) sidereal differential rotation law
    CSU           = -0.361e-6               # Snodgrass & Ulrich (1990) sidereal differential rotation law

    ### DOWNLOAD JPL DATA

    # Files output
    file_jpl_sun_obs_obs = 'jpl_sun-obs_obs.txt'
    file_jpl_sun_obs     = 'jpl_sun-obs.txt'
    file_jpl_ssb_obs     = 'jpl_ssb-obs.txt'
    file_jpl_ssb_sun     = 'jpl_ssb-sun.txt'

    # Time
    time_start = Time(np.min(time_jdb)-1, format='jd').isot[:10]
    time_stop  = Time(np.max(time_jdb)+1, format='jd').isot[:10]
    time_step  = '1%20h' # 1 hr

    # Latitude and longitude of observatory
    obs_lat = EarthLocation.of_site(obsname).lat.degree * deg_to_rad
    obs_lon = EarthLocation.of_site(obsname).lon.degree * deg_to_rad

    # Get Sun-Observatory ephemeris in observer format
    target = '10'
    center = obscode
    url = f"https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='{target}'&MAKE_EPHEM='YES'&CENTER='{center}'&TABLE_TYPE='OBSERVER'&START_TIME='{time_start}'&STOP_TIME='{time_stop}'&STEP_SIZE='{time_step}'&CSV_FORMAT='YES'&QUANTITIES='1,13,17,30'&CAL_FORMAT='JD'&ANG_FORMAT='DEG'"
    file = file_jpl_sun_obs_obs
    subprocess.check_call(['curl', '-k', url, '-o', file, '-s'])

    # Get Sun-Observatory ephemeris
    target = '10'
    center = obscode
    url = f"https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='{target}'&MAKE_EPHEM='YES'&CENTER='{center}'&TABLE_TYPE='VECTORS'&OUT_UNITS='KM-S'&START_TIME='{time_start}'&STOP_TIME='{time_stop}'&STEP_SIZE='{time_step}'&CSV_FORMAT='YES'"
    file = file_jpl_sun_obs
    subprocess.check_call(['curl', '-k', url, '-o', file, '-s'])

    # Get SSB-Observatory ephemeris (SSB: Solar System Barycenter)
    target = '0'
    center = obscode
    url = f"https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='{target}'&MAKE_EPHEM='YES'&CENTER='{center}'&TABLE_TYPE='VECTORS'&OUT_UNITS='KM-S'&START_TIME='{time_start}'&STOP_TIME='{time_stop}'&STEP_SIZE='{time_step}'&CSV_FORMAT='YES'"
    file = file_jpl_ssb_obs
    subprocess.check_call(['curl', '-k', url, '-o', file, '-s'])

    # Get SSB-Sun ephemeris (SSB: Solar System Barycenter)
    target = '0'
    center = '@sun'
    url = f"https://ssd.jpl.nasa.gov/horizons_batch.cgi?batch=1&COMMAND='{target}'&MAKE_EPHEM='YES'&CENTER='{center}'&TABLE_TYPE='VECTORS'&OUT_UNITS='KM-S'&START_TIME='{time_start}'&STOP_TIME='{time_stop}'&STEP_SIZE='{time_step}'&CSV_FORMAT='YES'"
    file = file_jpl_ssb_sun
    subprocess.check_call(['curl', '-k', url, '-o', file, '-s'])

    # Read data
    jdb_jpl, ra, dec, angdia, nppa, tdelta                 = np.genfromtxt(file_jpl_sun_obs_obs, skip_header=60, skip_footer=95, delimiter=',', unpack=True, usecols=(0,3,4,5,6,8))
    jdb_jpl, x_sun_obs, y_sun_obs, z_sun_obs, dist_sun_obs = np.genfromtxt(file_jpl_sun_obs    , skip_header=50, skip_footer=64, delimiter=',', unpack=True, usecols=(0,2,3,4,9)  )

    # Delete files
    subprocess.check_call(f'rm {file_jpl_sun_obs_obs}'.split(' '))
    subprocess.check_call(f'rm {file_jpl_sun_obs}'    .split(' '))
    subprocess.check_call(f'rm {file_jpl_ssb_obs}'    .split(' '))
    subprocess.check_call(f'rm {file_jpl_ssb_sun}'    .split(' '))

    # Scale data
    ra     *= deg_to_rad                 # Sun RA
    dec    *= deg_to_rad                 # Sun DEC
    angdia *= deg_to_rad / deg_to_arcsec # angular diameter
    nppa   *= deg_to_rad                 # North Pole position angle
    tdelta /= d_to_s                     # difference between TT and UT1

    ### CALCULATIONS

    # Calculate solar angular velocity
    wsun = 2.*np.pi/protsun

    # Calculate Earth angular velocity wearth = dtheta/dt
    # Using Kepler second law: pi a b / P = dA / dt = 0.5 r^2 dtheta/dt
    wearth = 2.*np.pi*AU_to_km*AU_to_km*np.sqrt(1.-eccearth*eccearth)/yr_to_s/dist_sun_obs/dist_sun_obs

    # Calculate solar rotation pole in ecliptic coordinates
    sinsunpolelat  = np.sin(sunpoledec)*np.cos(epsearth) - np.cos(sunpoledec)*np.sin(epsearth)*np.sin(sunpolera)
    cossunpolelat  = np.sqrt(1.-sinsunpolelat*sinsunpolelat)
    sinsunpolelong = (np.sin(sunpoledec)*np.sin(epsearth) + np.cos(sunpoledec)*np.cos(epsearth)*np.sin(sunpolera))/cossunpolelat
    cossunpolelong = np.cos(sunpoledec)*np.cos(sunpolera)/cossunpolelat

    # Transform to Cartesian ecliptic coordinates
    xsunpole = cossunpolelat*cossunpolelong
    ysunpole = cossunpolelat*sinsunpolelong
    zsunpole = sinsunpolelat

    # Get inclination angle between Sun centre and pole (via dot product)
    cosi = -(x_sun_obs*xsunpole + y_sun_obs*ysunpole + z_sun_obs*zsunpole)/dist_sun_obs
    sini = np.sqrt(1.-cosi*cosi)

    ### CORRECTION | DIFFERENTIAL EXTINCTION

    def func_corr_extinction(jdb, extinction, ldc, ra=ra, dec=dec, angdia=angdia, nppa=nppa, tdelta=tdelta, wearth=wearth, cosi=cosi):

        # Light travel time delay between the Sun and the observatory
        tdelay_sun_obs = dist_sun_obs / c / d_to_s

        # Time at observatory corrected for light travel time delay
        jdb_obs = jdb + np.interp(jdb, jdb_jpl, tdelay_sun_obs)

        # Interpolate queried data onto the observed time stamps
        ra     = np.interp(jdb_obs, jdb_jpl, ra    )
        ra     = np.unwrap(ra) % (2*np.pi)
        dec    = np.interp(jdb_obs, jdb_jpl, dec   )
        angdia = np.interp(jdb_obs, jdb_jpl, angdia)
        nppa   = np.interp(jdb_obs, jdb_jpl, nppa  )
        tdelta = np.interp(jdb_obs, jdb_jpl, tdelta)
        wearth = np.interp(jdb_obs, jdb_jpl, wearth)
        cosi   = np.interp(jdb_obs, jdb_jpl, cosi  )

        # GMST procedure adapted from ERFA/gmst06
        # based on SOFA version "20160503_a"
        # Capitaine, N., Wallace, P.T. & Chapront, J., 2005, A&A 432, 355
        dayfrac = (jdb_obs) % 1
        era00   = 2.0 * np.pi * (dayfrac + 0.7790572732640 + 0.00273781191135448 * (jdb_obs - 2450000.0 - 1545.0))                             # [rad]    Earth rotation angle
        ttjd    = jdb_obs + tdelta
        bigt    = (ttjd - 2450000.0 - 1545.0) / 36525.0
        gmst    = 0.014506 + (4612.156534 + (1.3915817 + (-0.00000044 + (-0.000029956 + (-0.0000000368) * bigt) * bigt) * bigt) * bigt) * bigt # [arcsec] Greenwich mean sidereal time
        gmst    = (gmst / 3600.0 / 180.0 * np.pi + era00) % (2.0 * np.pi)                                                                      # [rad]

        # Get hour angle of the Sun
        # See https://en.wikipedia.org/wiki/Sidereal_time
        # LMST = GMST + longitude as defined on wikipedia
        lmst       = gmst + obs_lon                               # [rad] local mean sidereal time
        hour_angle = (lmst - ra + np.pi) % (2.0 * np.pi) - np.pi  # [rad] hour angle

        # Get airmass
        cosz = np.sin(dec) * np.sin(obs_lat) + np.cos(dec) * np.cos(obs_lat) * np.cos(hour_angle)
        sinz = np.sqrt(1.0 - cosz * cosz)
        air  = 1.0 / cosz

        # Get cospp and sinpp (position angle corrected for North Pole position angle)
        sinp  = np.sin(hour_angle) * np.cos(obs_lat) / sinz
        cosp  = np.sqrt(1.0 - sinp * sinp)
        sinpa = np.sin(nppa)
        cospa = np.cos(nppa)
        sinpp = sinp * cospa - cosp * sinpa  # pp = p - pa, apply trigonometric rules
        cospp = cosp * cospa + sinp * sinpa

        # Get eps
        zangle = np.arccos(cosz)                              # zenith angle
        zp     = zangle + angdia / 2.0                        # zenith angle bottom
        zm     = zangle - angdia / 2.0                        # zenith angle top
        airp   = 1.0 / np.cos(zp)                             # airmass at bottom
        airm   = 1.0 / np.cos(zm)                             # airmass at top
        airyp  = airp * (1.0 - 0.0012 * (airp * airp - 1.0))  # correction for high airmass following Young (1994)
        airym  = airm * (1.0 - 0.0012 * (airm * airm - 1.0))  # correction for high airmass following Young (1994)
        dair   = airyp - airym                                # difference in airmass between bottom and top (dair > 0)
        dmag   = extinction * dair / 2.0                      # difference in magnitude due to airmass and extinction (dmag > 0)
        eps    = dmag * np.log(10.0) / 2.5
        
        # Get velocity correction due to differential extinction and parallactic angle
        corr_extinction = - (
            eps * rsun * sinpp
            * (
                ((ASU - wearth) * (7 * ldc - 15)) / (20 * (3 - ldc))
                + (BSU * (19 * ldc - 35 + cosi * cosi * (3 * ldc - 35))) / (280 * (3 - ldc))
                + (CSU * (187 * ldc - 315 - cosi**2.0 * (630 - 118 * ldc) - 105 * cosi**4.0 * (ldc - 1)))
                / (6720 * (3 - ldc))
            )
        )

        return hour_angle, corr_extinction

    ### CORRECTION | VISNI

    def func_corr_vsini(jdb, wearth=wearth, sini=sini):

        # Interpolate queried data onto the observed time stamps
        wearth = np.interp(jdb, jdb_jpl, wearth)
        sini   = np.interp(jdb, jdb_jpl, sini  )

        # vsini correction term
        corr_vsini = rsun*np.sqrt(wsun**2 - ((wsun-wearth)*sini)**2)

        return corr_vsini

    ### CORRECTIONS

    # Nr. of orders and files
    Norder, Nfile = extinction.shape

    # Empty array
    corr_extinction = np.empty((Norder,Nfile), dtype=float)

    # Loop orders
    for i in range(Norder):

        # Differential extinction correction
        hour_angle, corr_extinction[i] = func_corr_extinction(time_jdb, extinction[i], ldc[i])

    # vsini correction
    corr_vsini = func_corr_vsini(time_jdb)

    return hour_angle, corr_extinction, corr_vsini