from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys

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

specfits, readlinedat, fileout, space, rejt_str, distline, rvmask = read_mine("mine.opt")
ll, flux = read_spectra(specfits)
ll = ares.correct_lambda_rvpy(ll.copy(), flux, rvmask) 

rejt = ares.get_rejtpy(rejt_str, ll, flux)

if len(sys.argv) < 2:
    lines = np.loadtxt(readlinedat, unpack=True, skiprows=2, usecols=(0,), ndmin=1)
    print(lines)
    line = lines[0]
else:
    line = float(sys.argv[1])

print("Measuring line: ", line)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ll,flux)
ax.set_xlim(line-space, line+space)

tellme('Zoom in then hit SPACE!')
zoom_ok = False
while not zoom_ok:
    zoom_ok = plt.waitforbuttonpress(timeout=-1)

while True:
    pts = []
    while len(pts) < 2:
        tellme('Select left & right continuum points')
        pts = np.asarray(plt.ginput(2, timeout=-1))
        if len(pts) < 2:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second
    print(pts)
    ax.plot(pts[:,0],pts[:,1],c='k')
    tellme('Hit space to accept, mouse click to repeat!')
    if plt.waitforbuttonpress():
        break
cont=pts

tellme("Mark the Gauss centers! Middle click to stop")
pts = np.asarray(plt.ginput(-1, timeout=-1))
print(pts)
ngauss = len(pts)

ax.scatter(pts[:,0],pts[:,1],marker='o')
gauss=pts

print("Cont:", cont)
print("Gauss:", gauss)

# normalize to line
x_coords = (cont[0][0],cont[1][0])
y_coords = (cont[0][1],cont[1][1])
A = np.vstack([x_coords,np.ones(len(x_coords))]).T 
m, c = np.linalg.lstsq(A, y_coords,rcond=None)[0]
fluxn = flux/(m*ll+c)

#Get local norm spec:
i1 = np.where(ll > cont[0,0])[0][0]
i2 = np.where(ll > cont[1,0])[0][0]
print(cont[0,0], ll[i1])
print(cont[1,0], ll[i2])

ll_l,flux_l = ares.getMedida_local_spec(ll, fluxn, i1, i2)


#Original acoef
acoef = np.zeros(ngauss*3)
sigma_const = 400.
for i in range(ngauss):
# We may need to play a bit with the constrains
    acoef[i*3 + 1] = sigma_const
    acoef[i*3 + 2] = gauss[i][0]
    ig = np.where(ll > gauss[i][0])[0][0]
    acoef[i*3] = (fluxn[ig] - 1)
print(acoef)

gauss = flux_l*0+1-rejt
init = ares.get_yfit(ll_l,acoef)


(acoef, acoef_er, status) = ares.fitngausspy(ll_l, flux_l, gauss, acoef)
for i in np.arange(0,len(acoef),3):
    print("::acoef[%2i]:  %.5f acoef[%2i]:  %9.5f acoef[%2i]:  %7.2f \n" %(i, acoef[i]+1., i+1, acoef[i+1], i+2, acoef[i+2]))
bestfit = ares.get_yfit(ll_l,acoef)

ew, error_ew, info_line = ares.getMedida_compile_ew_original(acoef, acoef_er, ll_l, flux_l, line, distline)
#print(out.fit_report(min_correl=0.5))
print ("-----------------------\n")
print ("line: %8.4f" % (line))
print ("EW: %8.4f" % (ew))
print ("Error EW: %8.4f" % (error_ew))
#print ("News: %2i" % (news))
#print ("Ngauss: %2i" % (len(out.params)/5))
print ("-----------------------\n")


#plt.plot(ll_l, flux_l)
plt.plot(ll_l, (init+1)*(m*ll_l+c), 'k--')
plt.plot(ll_l, (bestfit+1)*(m*ll_l+c), 'r-')
plt.axvline(line)

tellme("Complete!! You may close the plot...")
plt.show()
#(lineo, ngauss, line_depth, line_sigma, ew, error_ew, line_depth_f, line_sigma_f, line_center_f, news) = info_line
print(info_line)

