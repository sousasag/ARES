from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

sys.path.insert(0, '/home/sousasag/Programas/GIT_projects/ARES/ares_cython') 
#sys.path.insert(0, '/mnt/e/Linux_stuff/Programas/GIT_Projects/SPECPAR3/codes/ARES/ares_cython') 
import ares_module as ares 

def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()

specfits = "/home/sousasag/Programas/GIT_projects/SPECPAR3/codes/ARES/sun_harps_ganymede.fits"


def read_spectra(specfits):
    hdulist    = fits.open(specfits)
    img_header = hdulist[0].header  
    img_data   = hdulist[0].data
    crval1     = hdulist[0].header['CRVAL1'] 
    cdelta1    = hdulist[0].header['CDELT1'] 
    npoints    = hdulist[0].header['NAXIS1'] 
    ll   = np.arange(0,npoints)*cdelta1+crval1
    flux = np.array(img_data, dtype = 'double')
    hdulist.close()
    return ll, flux 

def read_mine(mineopt):
    filein = open(mineopt,'r')
    options = filein.readlines()
    filein.close()
    specfits    = options[0].split("'")[1]
    readlinedat = options[1].split("'")[1]
    fileout     = options[2].split("'")[1]
    space       = float(options[6].split("=")[-1].strip())
    rejt        = options[7].split("=")[-1].strip()
    lineresol   = float(options[8].split("=")[-1].strip())
    rvmask      = options[11].split("'")[1]
    return specfits, readlinedat, fileout, space, rejt, lineresol, rvmask


### Main program:
def main():
    specfits, readlinedat, fileout, space, rejt_str, distline, rvmask = read_mine("mine.opt")
    try:
        ll, flux = read_spectra(specfits)
    except:
        ll, flux = np.loadtxt(specfits,unpack=True)
        ll = np.array(ll, dtype = 'double')
        flux = np.array(flux, dtype = 'double')


    ll = ares.correct_lambda_rvpy(ll.copy(), flux, rvmask) 

    rejt = ares.get_rejtpy(rejt_str, ll, flux)

    if len(sys.argv) < 2:
        lines = np.loadtxt(readlinedat, unpack=True, skiprows=2, usecols=(0,), ndmin=1)
        print(lines)
        line = lines[0]
    else:
        line = float(sys.argv[1])

    print("Measuring line: ", line)

    ew, error_ew, info_line = ares.get_medida_interactive(ll, flux, line, space, rejt, distline, rvmask)

    strline = "%8.3f   %d   %.5f   %.5f   %.5f   %.5f   %.5f   %.5f   %7.2f   %d\n" % info_line 
    print(strline)
    fileo = open(fileout, "a")
    fileo.write(strline)
    fileo.close()

if __name__ == "__main__":
    main()




