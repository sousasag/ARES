""" Example of wrapping ares c functions using Cython. """


# import both numpy and the Cython declarations for numpy
import numpy as np
from numpy import sqrt, pi, exp, linspace, loadtxt
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt

cimport numpy as np

cdef extern from "areslib.h":
  void fitngauss(double* t, double* y, double* sigma, long nvec, double* acoef, double* acoef_er, int para, int* status)
  void get_inner_data(double* x, double* y, long nx, double linha, double rejt, int* xi1, int* xi2)
  void zeroscenterfind(double* y, double* iy, double* dy, double* ddy, int n, int* center, int* ncenter, double rejt)
  void smooth(double* vec, long n, int w, double* svec)
  int continuum_det5 (double* x, double* y, double* ynorm, long nxele, 
                      double* res, double tree, int plots_flag)
  void deriv(double *x, double *y, double *dy, long n)
  void clean_zero_gaps(double* flux, long npoints)
  void getMedida(double* xpixels, double* pixels, long npoints, 
                 float linha, double space, 
                 double tree, int* plots_flag2, 
                 double smoothder, double distlinha, 
                 int ilinha, double* aponta, 
                 double lambdai, double lambdaf,
                 int cont_flag, int max_fit_lines)

cdef extern from "filesio.h":
  long find_pixel_line(double* xpixels, long npoints, double linha)


cdef extern from "sn_rejt_estimator.h":
  double get_rejt(char* tree, double* ll, double* flux, long npoints)

cdef extern from "rvcor.h":
  void correct_lambda(double* ll, int npoints, double vrad)
  double get_rv(double *ll, double* flux, long npoints, char* rvmask)


# #---------------------------------------------------------------------
# #---------------------------------------------------------------------

def get_rv_py(np.ndarray[double, ndim=1, mode="c"] ll not None,
                        np.ndarray[double, ndim=1, mode="c"] flux not None,
                        rvmask):
  """
  Use the rvmask from ARES to [compute and] correct the ll in radial velocity
  """
  rv = get_rv(<double *> np.PyArray_DATA(ll),
              <double *> np.PyArray_DATA(flux),
              ll.shape[0], rvmask)
  return rv


#---------------------------------------------------------------------
#---------------------------------------------------------------------

def correct_lambda_rvpy(np.ndarray[double, ndim=1, mode="c"] ll not None,
                        np.ndarray[double, ndim=1, mode="c"] flux not None,
                        rvmask):
  """
  Use the rvmask from ARES to [compute and] correct the ll in radial velocity
  """
  rv = get_rv(<double *> np.PyArray_DATA(ll),
              <double *> np.PyArray_DATA(flux),
              ll.shape[0], rvmask)
  print ("RV to correct: ", rv)
  correct_lambda(<double *> np.PyArray_DATA(ll), ll.shape[0], rv)
  return ll


# #---------------------------------------------------------------------
# #---------------------------------------------------------------------


def get_rejtpy(tree, 
               np.ndarray[double, ndim=1, mode="c"] ll not None,
               np.ndarray[double, ndim=1, mode="c"] flux not None):
  """
  Getting the rejt value from the inputed string in config.file
  """
  return get_rejt(tree,
                  <double *> np.PyArray_DATA(ll),
                  <double *> np.PyArray_DATA(flux), 
                  ll.shape[0])


#---------------------------------------------------------------------
#---------------------------------------------------------------------


def fitngausspy(np.ndarray[double, ndim=1, mode="c"] t not None,
                np.ndarray[double, ndim=1, mode="c"] y not None,
                np.ndarray[double, ndim=1, mode="c"] sigma not None,
                np.ndarray[double, ndim=1, mode="c"] acoef not None):
  """
  Fitting N gaussians
  """
  
  acoef_er = np.zeros(acoef.shape[0], dtype = 'double')
  status = np.zeros(1,dtype=int)
  fitngauss(<double *> np.PyArray_DATA(t),
            <double *> np.PyArray_DATA(y),
            <double *> np.PyArray_DATA(sigma),
            t.shape[0],
            <double *> np.PyArray_DATA(acoef),
            <double *> np.PyArray_DATA(acoef_er),
            acoef.shape[0],
            <int *> np.PyArray_DATA(status))
  return (acoef, acoef_er, status)



#---------------------------------------------------------------------
#---------------------------------------------------------------------

def get_inner_datapy(np.ndarray[double, ndim=1, mode="c"] x not None,
                     np.ndarray[double, ndim=1, mode="c"] y not None,
                     linha, rejt):
  """
  Identify absorption line centers in spectra, return the indexes
  """
  
  xi1 = np.zeros(1,dtype=int)
  xi2 = np.zeros(1,dtype=int)
 
  get_inner_data(<double *> np.PyArray_DATA(x),
                 <double *> np.PyArray_DATA(y),
                  y.shape[0], linha, rejt,
                  <int *> np.PyArray_DATA(xi1),
                  <int *> np.PyArray_DATA(xi2))

  return (xi1[0],xi2[0])

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def zeroscenterfindpy(np.ndarray[double, ndim=1, mode="c"] y not None,
                      np.ndarray[double, ndim=1, mode="c"] iy not None,
                      np.ndarray[double, ndim=1, mode="c"] dy not None,
                      np.ndarray[double, ndim=1, mode="c"] ddy not None, rejt):
  """
  Identify absorption line centers in spectra, return the indexes
  """
  center_n = np.zeros(y.shape[0],dtype=int)
  ncenter_n = np.zeros(1,dtype=int)
  cdef np.ndarray[int, ndim=1, mode="c"] center_c
  cdef np.ndarray[int, ndim=1, mode="c"] ncenter_c
  center_c = np.ascontiguousarray(center_n, dtype=np.dtype("i"))
  ncenter_c = np.ascontiguousarray(ncenter_n, dtype=np.dtype("i"))

  zeroscenterfind(<double *> np.PyArray_DATA(y),
                  <double *> np.PyArray_DATA(iy),
                  <double *> np.PyArray_DATA(dy),
                  <double *> np.PyArray_DATA(ddy),
                  y.shape[0],
                  <int*> np.PyArray_DATA(center_c),
                  <int*> np.PyArray_DATA(ncenter_c),
                  rejt)
  centerout = center_c[0:ncenter_c[0]].copy()
  return centerout


#---------------------------------------------------------------------
#---------------------------------------------------------------------

def smoothpy(np.ndarray[double, ndim=1, mode="c"] vec not None, w):
  """
  Compute the smooth using ARES smooth function
  """

  svec = np.zeros(vec.shape[0])
  smooth(<double *> np.PyArray_DATA(vec),
         vec.shape[0], w, 
        <double *> np.PyArray_DATA(svec))

  return svec

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def continuum_det5py(np.ndarray[double, ndim=1, mode="c"] x not None,
                     np.ndarray[double, ndim=1, mode="c"] y not None,
                     tree):
  """
  Compute the local continuum using ARES function the a given x,y data.
  """
  
  ynorm = np.zeros(x.shape[0],dtype=float)
  res = np.zeros(4,dtype=float)
  flag = continuum_det5(<double *> np.PyArray_DATA(x),
                        <double *> np.PyArray_DATA(y),
                        <double *> np.PyArray_DATA(ynorm),
                        x.shape[0],
                        <double *> np.PyArray_DATA(res),
                        tree, 0)
  if flag == 1:
    return ynorm, res
  else:
    return [-1], [-1]

  

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def derivpy(np.ndarray[double, ndim=1, mode="c"] x not None,
            np.ndarray[double, ndim=1, mode="c"] y not None):
  """
  Compute the derivative of the a given x,y data.
  """
  dy = np.zeros(x.shape[0])
  deriv(<double *> np.PyArray_DATA(x),
        <double *> np.PyArray_DATA(y),
        <double *> np.PyArray_DATA(dy),
        x.shape[0])

  return dy

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def find_pixel_linepy(np.ndarray[double, ndim=1, mode="c"] xpixels not None,
                      linha):
  """
  return index of value
  """
  return find_pixel_line(<double *> np.PyArray_DATA(xpixels), xpixels.shape[0], linha)

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def clean_zero_gaps_py(np.ndarray[double, ndim=1, mode="c"] flux not None):
  """
  Cleans the zeros in the flux array
  """
  clean_zero_gaps(<double *> np.PyArray_DATA(flux), flux.shape[0])
  return flux

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def getMedidapy(np.ndarray[double, ndim=1, mode="c"] xpixels not None,
                np.ndarray[double, ndim=1, mode="c"] pixels not None,
                linha, space, tree, smoothder, distlinha, lambdai, lambdaf, cont_flag=0, max_fit_lines=-1):
  """
  Gets the values for a given line that are outputed in the ascii file

  """

  import os.path
  if os.path.isfile('logARES.txt'):
    os.remove('logARES.txt')
  aponta = np.zeros(9,dtype=float)
  getMedida(<double *> np.PyArray_DATA(xpixels),
            <double *> np.PyArray_DATA(pixels),
            xpixels.shape[0],
            linha, space, tree,
            <int* > np.PyArray_DATA(np.array([0])),
            smoothder, distlinha, 
            0,<double *> np.PyArray_DATA(aponta),
            lambdai, lambdaf,
            cont_flag, max_fit_lines)
  return aponta

#---------------------------------------------------------------------
#---------------------------------------------------------------------

def lmfit_ngauss_constrains(x,y, params, constrains):
  """
  Fit n gaussians using the lmfit module
  """

  #params = params[0]
  #constrains = constrains[0]
  mods = []
  prefixes = []

  for i in range(0, len(params), 3):
    pref = "g%02i_" % (i/3)
    gauss_i = GaussianModel(prefix=pref)

    if i == 0:
      pars = gauss_i.guess(y, x=x)
    else:
      pars.update(gauss_i.make_params())
    A = params[i]
    limA = constrains[i]
    l_cen = params[i+1]
    limL = constrains[i+1]
    sigma = params[i+2]
    limS = constrains[i+2]

    pars[pref+'amplitude'].set(A, min=limA[0], max=limA[1])
    #pars[pref+'amplitude'].set(A)
    pars[pref+'center'].set(l_cen, min=limL[0], max=limL[1])
    #pars[pref+'center'].set(l_cen)
    pars[pref+'sigma'].set(sigma, min=limS[0], max=limS[1])

    mods.append(gauss_i)
    prefixes.append(pref)

  mod = mods[0]

  if len(mods) > 1:
    for m in mods[1:]:
      mod += m

  init = mod.eval(pars, x=x)
  out = mod.fit(y, pars, x=x)
  return mod, out, init


#---------------------------------------------------------------------
#---------------------------------------------------------------------


def find_lines(x,y, smoothder, line, tree, distline, rejt=0.98):
  """
  Find the number of total lines to be fitted by ARES for a given line
  """
  iy  = y.copy()
  y   = smoothpy(derivpy(x,y), smoothder)
  dy  = smoothpy(derivpy(x,y), smoothder)
  ddy = smoothpy(derivpy(x,dy), smoothder)

  (i1, i2) = get_inner_datapy(x,iy, line, tree)

  x = x[i1:i2]
  y = y[i1:i2]
  iy = iy[i1:i2]
  dy = dy[i1:i2]
  ddy = ddy[i1:i2]

  center = zeroscenterfindpy(y,iy,dy,ddy,rejt)
  if len(center) < 1:
    return (np.array([]),np.array([]),-1,-1)
  center2 = [center[0]]
  for i in range(1,len(center)):
    if abs(x[center[i]] - x[center[i-1]]) >= distline:
      center2.append(center[i])

  center = np.array(center2)
  return (x[center],iy[center], i1, i2)



#---------------------------------------------------------------------
#---------------------------------------------------------------------



def getMedida_pyfit(ll, flux, line, space, rejt, smoothder, distline, plots_flag):
  """
  Get the EW, and EW error for a given line using the ARES functions 
  with exception for the fitting procedure. 
  In this case it is used the lmfit module.
  """

#  line = 4741.53
#  space = 3
  tree = rejt
#  distline = 0.1

  nx1test = find_pixel_linepy(ll, line-space)
  nx2test = find_pixel_linepy(ll, line+space)

  x = ll[nx1test:nx2test].copy()
  y = flux[nx1test:nx2test].copy()

  ynorm,res = continuum_det5py(x,y,tree)
  lc, ld, i1, i2 =  find_lines(x,ynorm, 4, line, tree, distline)
  if lc.shape[0] < 1:
    print(line, "line not found")
    return (-1,-1, -1)

  acoef = np.zeros(lc.shape[0]*3)
  constrains = []
  sigma_const = 0.05
  for i in range(lc.shape[0]):
# We may need to play a bit with the constrains
    acoef[i*3] = (ld[i] - 1) * (sigma_const*sqrt(2*pi))
    constrains.append((-2.506*sigma_const,0.0))
    acoef[i*3 + 1] = lc[i]
    constrains.append((lc[i]-5*distline,lc[i]+5*distline))
    acoef[i*3 + 2] = sigma_const
    constrains.append((0.01,0.2))

  x = x[i1:i2]
  y = ynorm[i1:i2] - 1
  npoints = x.shape[0]
  xl = np.zeros(npoints*3)
  yl = np.zeros(npoints*3)
  xl[0:npoints] = x-(x[-1]-x[0])
  xl[npoints:2*npoints] = x
  xl[npoints*2:] = x+(x[-1]-x[0])
  yl[0:npoints] = 0
  yl[npoints:2*npoints] = y
  yl[npoints*2:] = 0
  
  x = xl 
  y = yl

  #np.savetxt('test.out', (x,y))
  
  #mod, out, init = lmfit_ngauss(x,y, acoef)
  mod, out, init = lmfit_ngauss_constrains(x,y, acoef, constrains)
  
  values = out.best_values
  params_fit = out.params
  ew = 0.
  error_ew = 0.
  news = 0
  line_is_found = 0
  for i in range(0,len(out.params)/5):
    pref = "g%02i_" % (i)
    if abs(line - values[pref+'center']) < distline:
      ew+=values[pref+'amplitude']*1000*-1
      #Deal with stderr null values -> 20% error
      if params_fit[pref+'amplitude'].stderr is None:
        error_ew += ew*0.2
      else:
        error_ew += params_fit[pref+'amplitude'].stderr*1000
      line_depth    = params_fit[pref+'amplitude']/(params_fit[pref+'sigma']*sqrt(2*pi))*-1
      line_sigma    = params_fit[pref+'fwhm']
      line_depth_f  = params_fit[pref+'amplitude']
      line_sigma_f  = params_fit[pref+'sigma']
      line_center_f = params_fit[pref+'center']
      news += 1
      line_is_found += 1
  if line_is_found:
  
    if plots_flag:

      print(out.fit_report(min_correl=0.5))

      print ("-----------------------\n")
      print ("line: %6.2f" % (line))
      print ("EW: %6.2f" % (ew))
      print ("Error EW: %6.2f" % (error_ew))
      print ("News: %2i" % (news))
      print ("Ngauss: %2i" % (len(out.params)/5))

      print ("-----------------------\n")

      plt.plot(x, y)
      plt.plot(x, init, 'k--')
      plt.plot(x, out.best_fit, 'r-')
      plt.axvline(line)
      plt.show()

    ngauss = len(out.params)/5



    info_line = (line, ngauss, line_depth, line_sigma, ew, error_ew, line_depth_f, line_sigma_f, line_center_f, news)
    return (ew, error_ew, info_line)

  else:
    #return -1,-1,(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)    
    print (line, "line not found -2")
    return (-1,-1, -2)



# #def getMedida_pyfit(ll, flux, line, space, rejt, smoothder, distline, plots_flag):




def getMedida_lines(ll, flux, line, space, rejt, distline, smoothder=4):
  """
  For a given line it gets the local spectrum and identify the lines for the fitting
  """
  nx1test = find_pixel_linepy(ll, line-space)
  nx2test = find_pixel_linepy(ll, line+space)

  x = ll[nx1test:nx2test].copy()
  y = flux[nx1test:nx2test].copy()
  ynorm,res = continuum_det5py(x,y,rejt)
  if len(ynorm) <= 1:
    return np.array([]),np.array([]),-1,-1,x,y
  lc, ld, i1, i2 =  find_lines(x, ynorm, smoothder, line, rejt, distline)
  return lc, ld, i1, i2, x, ynorm

def getMedida_coefs(x, ynorm, lc, ld, distline):
  """
  From the list of lines to be fitted it gets the guessing parameters for the fit
  """
  acoef = np.zeros(lc.shape[0]*3)
  constrains = []
  sigma_const = 0.05*2
  for i in range(lc.shape[0]):
# We may need to play a bit with the constrains
    acoef[i*3] = (ld[i] - 1) * (sigma_const*sqrt(2*pi))
    constrains.append((-2.506*sigma_const,0.0))
    acoef[i*3 + 1] = lc[i]
    constrains.append((lc[i]-5*distline,lc[i]+5*distline))
    acoef[i*3 + 2] = sigma_const
    constrains.append((0.01,0.2))
  return acoef, constrains


def getMedida_coefs_original(x, ynorm, lc, ld, distline):
  """
  From the list of lines to be fitted it gets the guessing parameters for the fit
  """
  acoef = np.zeros(lc.shape[0]*3)
  sigma_const = 400.
  for i in range(lc.shape[0]):
# We may need to play a bit with the constrains
    acoef[i*3] = (ld[i] - 1)
    acoef[i*3 + 1] = sigma_const
    acoef[i*3 + 2] = lc[i]
  return acoef



def getMedida_local_spec(x, ynorm, i1, i2):
  """
  It prepares the local spectrum so it passes to 1 all 
  the flux that does not count for the flux
  """
  x = x[i1:i2]
  y = ynorm[i1:i2] - 1
  npoints = x.shape[0]
  xl = np.zeros(npoints*3)
  yl = np.zeros(npoints*3)
  xl[0:npoints] = x-(x[-1]-x[0])
  xl[npoints:2*npoints] = x
  xl[npoints*2:] = x+(x[-1]-x[0])
  yl[0:npoints] = 0
  yl[npoints:2*npoints] = y
  yl[npoints*2:] = 0
  
  x = xl 
  y = yl

  return x,y

def getMedida_fitgauss(x, y, acoef, constrains):
  """
  calls the function for the lmfit ngauss fit
  """
#  mod, out, init = lmfit_ngauss(x,y, acoef)
  mod, out, init = lmfit_ngauss_constrains(x,y, acoef, constrains)
  return mod, out, init


def getMedida_compile_ew(out, x, y, init, line, distline, plots_flag = False):
  """
  From the n gauss fitting it computes the EW and its error for the output
  """

# nparam_gauss depends on the lmfit version (input model gauss that changed its parameter model)
#  nparam_gauss = 4
  nparam_gauss = 5
  values = out.best_values
  params_fit = out.params
  ew = 0.
  error_ew = 0.
  news = 0
  line_is_found = 0
  for i in range(0,len(out.params)/nparam_gauss):
    pref = "g%02i_" % (i)
    if abs(line - values[pref+'center']) < distline:
      ew+=values[pref+'amplitude']*1000*-1
      error_ew += params_fit[pref+'amplitude'].stderr*1000
      line_depth    = params_fit[pref+'amplitude']/(params_fit[pref+'sigma']*sqrt(2*pi))*-1
      line_sigma    = params_fit[pref+'fwhm']
      line_depth_f  = params_fit[pref+'amplitude']
      line_sigma_f  = params_fit[pref+'sigma']
      line_center_f = params_fit[pref+'center']
      news += 1
      line_is_found += 1
      
  if line_is_found:
  
    if plots_flag:

      print(out.fit_report(min_correl=0.5))

      print ("-----------------------\n")
      print ("line: %6.2f" % (line))
      print ("EW: %6.2f" % (ew))
      print ("Error EW: %6.2f" % (error_ew))
      print ("News: %2i" % (news))
      print ("Ngauss: %2i" % (len(out.params)/nparam_gauss))

      print ("-----------------------\n")

      plt.plot(x, y)
      plt.plot(x, init, 'k--')
      plt.plot(x, out.best_fit, 'r-')
      plt.axvline(line)
      plt.show()

    ngauss = len(out.params)/nparam_gauss
    info_line = (line, ngauss, line_depth, line_sigma, ew, error_ew, line_depth_f, line_sigma_f, line_center_f, news)
    return ew, error_ew, info_line
  else:
    #return -1,-1,(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)    
    print (line, "line not found -2")
    return (-1,-1, -2)





def getMedida_compile_ew_original(acoef, acoef_er, x, y, line, distline, plots_flag = False):
  """
  From the n gauss fitting it computes the EW and its error for the output
  """
  ew = 0.
  error_ew = 0.
  news = 0
  line_is_found = 0
  for i in range(0,len(acoef)/3):
    if abs(line - acoef[3*i+2]) < distline:
      ew+=acoef[3*i]*np.sqrt(np.pi/acoef[3*i+1])
      error_ew += ew*ew * (acoef_er[3*i]*acoef_er[3*i]/acoef[3*i]/acoef[3*i] + (0.5*0.5*acoef_er[3*i+1]*acoef_er[3*i+1]/acoef[3*i+1]/acoef[3*i+1]));
      line_depth    = -acoef[3*i]
      line_sigma    = 2.*np.sqrt(np.log(2)/acoef[3*i+1]);
      line_depth_f  = -acoef[3*i]
      line_sigma_f  = acoef[3*i+1]
      line_center_f = acoef[3*i+2]
      news += 1
      line_is_found += 1
  ew = ew*(-1000.)
  error_ew = np.sqrt(error_ew) * 1000.
      
  if line_is_found:
  
    if plots_flag:

      print(acoef)

      print ("-----------------------\n")
      print ("line: %6.2f" % (line))
      print ("EW: %6.2f" % (ew))
      print ("Error EW: %6.2f" % (error_ew))
      print ("News: %2i" % (news))
      print ("Ngauss: %2i" % (len(acoef)/3))

      print ("-----------------------\n")
 

      bestfit = np.ones(x.shape)
      for j in range(0,len(acoef)/3):
        bestfit += acoef[j*3]* np.exp (- acoef[j*3+1] * (x-acoef[j*3+2]) * (x-acoef[j*3+2]) )


      plt.plot(x, y)
      plt.plot(x, bestfit, 'r-')
      plt.axvline(line)
      plt.show()

    ngauss = len(acoef)/3
    info_line = (line, ngauss, line_depth, line_sigma, ew, error_ew, line_depth_f, line_sigma_f, line_center_f, news)
    return ew, error_ew, info_line
  else:
    #return -1,-1,(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1)    
    print (line, "line not found -2")
    return (-1,-1, -2)


def getMedida_pyfit_sep(ll, flux, line, space, rejt, smoothder, distline, plots_flag):
  """
  Uses the functions to get the EW measurement as in ARES with the difference of the gaussian fitting using lmfit here.
  """
  lc, ld, i1, i2, x, ynorm = getMedida_lines(ll, flux, line, space, rejt, distline)

  if lc.shape[0] < 1:
    print (line, "line not found")
    return (-1,-1, -1)    

  acoef, constrains = getMedida_coefs(x, ynorm, lc, ld, distline)

  x,y = getMedida_local_spec(x, ynorm, i1, i2)

  #np.savetxt('test.out', (x,y)) 
  
  mod, out, init = getMedida_fitgauss(x,y, acoef, constrains)
  ew, error_ew, info_line = getMedida_compile_ew(out, x, y, init, line, distline)

  return ew, error_ew, info_line





def find_fit_guess(ll, flux, line, space, rejt, smoothder, distline, plots_flag):
  """
  Get the fit parameters
  """
  nx1test = find_pixel_linepy(ll, line-space)
  nx2test = find_pixel_linepy(ll, line+space)

  x = ll[nx1test:nx2test].copy()
  y = flux[nx1test:nx2test].copy()

  ynorm,res = continuum_det5py(x,y,rejt)

  lc, ld, i1, i2 =  find_lines(x,ynorm, 4, line, rejt, distline)
  if lc.shape[0] < 1:
    print (line, "line not found")
    return (-1,-1, -1)


  return x, ynorm, lc, ld, i1, i2  

def interface_fit_ngauss(x, y, line, distline, acoef, lmfit_flag = True, constrains = None, gauss = None):
  """
  Interface to homogeneize the input and output around the fitting of gaussians
  """
  if lmfit_flag:
    mod, out, init = getMedida_fitgauss(x,y, acoef, constrains)
    ew, error_ew, info_line = getMedida_compile_ew(out, x, y, init, line, distline)
  else:
    (acoef, acoef_er, status) = fitngausspy(x, y, gauss, acoef)
    ew, error_ew, info_line = getMedida_compile_ew_original(acoef, acoef_er, x, y, line, distline)
  return ew, error_ew, info_line


def interface_fit_ngauss_acoef(x, y, line, distline, acoef, lmfit_flag = True, constrains = None, gauss = None):
  """
  Interface to homogeneize the input and output around the fitting of gaussians
  """
  if lmfit_flag:
    mod, out, init = getMedida_fitgauss(x,y, acoef, constrains)
    ew, error_ew, info_line = getMedida_compile_ew(out, x, y, init, line, distline)
  else:
    (acoef, acoef_er, status) = fitngausspy(x, y, gauss, acoef)
    ew, error_ew, info_line = getMedida_compile_ew_original(acoef, acoef_er, x, y, line, distline)
  return ew, error_ew, info_line



def getMedida_differential_EWs(spec1, spec2, rejt1, rejt2, line, space, smoothder, distline, plots_flag):
  """
  Get the EW of a line in 2 similar spectra. forcing the same kind of fit, in what regards number of lines to fit.
  """
  pass


  

