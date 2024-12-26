import unittest
import sys
import os
import numpy as np
sys.path.insert(0,os.path.dirname(repr(__file__).replace("'",""))[:-len("tests")])
import ares_module as ares # type: ignore
import matplotlib.pyplot as plt

def gaussian(x ,a ,c ,sig):
    return a*np.exp(-(x-c)**2/(2*sig**2))


class TestARESpy(unittest.TestCase):

    def test_read_spectrum(self):
        spectrum='../../sun_harps_ganymede.fits'
        ll, flux = ares.read_spec(spectrum)
        self.assertEqual(len(flux), 313025, "Should be 313025")

    def test_correct_rv(self):
        llr=np.array([5000.1,5001,5002,6000,7010])
        flux=llr*0+1
        llrv=llr*(40/299792.458+1)
        llrvcor = ares.correct_lambda_rvpy(llrv.copy(), flux, "0,40")
        self.assertListEqual(list(llr), list(llrvcor), "Should be equal")
    
    def test_getrejt(self):
        spectrum='../../sun_harps_ganymede.fits'
        ll, flux = ares.read_spec(spectrum)
        llcor = ares.correct_lambda_rvpy(ll.copy(), flux, "0,0.1")
        rejt = ares.get_rejtpy("3;5764,5766,6047,6053,6068,6076", llcor, flux)
        self.assertAlmostEqual(rejt, 0.997048,6, "Should be 0.997048")


    def test_getpixelline(self):
        line_test = 6005.01
        ll = np.linspace(6000,6100,10001)
        index = ares.find_pixel_linepy(ll, line_test)
        self.assertEqual(line_test,ll[index],"Should be "+str(line_test))

    def test_continuum_run(self):
        spectrum='../../sun_harps_ganymede.fits'
        ll, flux = ares.read_spec(spectrum)
        locind = np.where((ll > 6002 ) & (ll <6008))
        x=ll[locind]
        y=flux[locind]
        ynorm,res = ares.continuum_det5py(x,y,0.99)
        ch = int(len(y)/4)
        meanloc = np.mean(np.sort(ynorm)[-ch:])
        print("Mean local normalized continuum: ", meanloc)
        self.assertAlmostEqual(meanloc,1,1,"Should be close to 1")

    def test_findlines(self):
        ll = np.linspace(6000,6010,1001)
        noise = np.random.normal(0,0.005,len(ll))
        flux = 1 - gaussian(ll,0.7,6005,0.15)  - gaussian(ll,0.5,6005.7,0.15) + noise
        #fig= plt.figure()
        #plt.plot(ll,flux)
        #fig.savefig("test.png")
        lc, ld, i1, i2 =  ares.find_lines(ll, flux, 8, 6005, 0.995, 0.1)
        #print("find lines:", lc,ld,i1,i2)
        self.assertEqual(len(lc),2,"Should find 2 lines")

    def test_getMedidaLines(self):
        spectrum='../../sun_harps_ganymede.fits'
        ll, flux = ares.read_spec(spectrum)
        lc, ld, i1, i2, x, ynorm = ares.getMedida_lines(ll, flux, 5701.55, 3, 0.997, 0.04, 4)
        #print("get Medida lines:", lc,ld,i1,i2)
        self.assertEqual(len(lc),3," Get Medida Lines: Should find 3 lines")

    def test_getCoefsOriginal(self):
        spectrum='../../sun_harps_ganymede.fits'
        ll, flux = ares.read_spec(spectrum)
        lc, ld, i1, i2, x, ynorm = ares.getMedida_lines(ll, flux, 5701.55, 3, 0.997, 0.04, 4)
        acoef_test_list = [-0.28, 400, 5701.11,
                           -0.58, 400, 5701.56,
                           -0.03, 400, 5701.92]
        acoef = ares.getMedida_coefs_original(x, ynorm, lc, ld, 0.1)
        #for i in np.arange(0,len(acoef),3):
        #    print("::acoef[%2i]:  %.5f acoef[%2i]:  %9.5f acoef[%2i]:  %7.2f \n" %(i, acoef[i]+1., i+1, acoef[i+1], i+2, acoef[i+2]))
        #print(acoef)
        test_array = np.testing.assert_array_almost_equal(acoef, acoef_test_list,decimal=2, err_msg="Acoef Original not similar enough")
        self.assertEqual(test_array, None, "...")
    
    def test_getMedidaLocalSpec(self):
        ll = np.linspace(6000,6010,1001)
        noise = np.random.normal(0,0.005,len(ll))
        flux = 1 - gaussian(ll,0.7,6005,0.15)  - gaussian(ll,0.5,6005.7,0.15) + noise
        lc, ld, i1, i2 =  ares.find_lines(ll, flux, 8, 6005, 0.995, 0.1)
        xl,yl = ares.getMedida_local_spec(ll, flux, i1, i2)
        #print("Medida Local Spec: ", i2-i1,len(xl), len(xl)/3)
        #fig= plt.figure()
        #plt.plot(xl,yl)
        #fig.savefig("test.png")
        self.assertEqual(3*(i2-i1), len(xl),"Should have 3 times the space of the local lines")

    def test_fitngausspy(self):
        ll = np.linspace(6000,6010,1001)
        noise = np.random.normal(0,0.005,len(ll))
        flux = gaussian(ll,0.7,6005,0.15)  - gaussian(ll,0.5,6005.7,0.15) + noise        
        gauss = ll*0+0.005
        acoefi = np.array([-0.80, 400, 6005,
                  -0.45, 400, 6006])
        #fig= plt.figure()
        #plt.plot(ll,flux)
        #fig.savefig("test.png")
        (acoef, acoef_er, status) = ares.fitngausspy(ll, flux, gauss, acoefi)
        #print("Fit NGAUSS",acoef)
        self.assertAlmostEqual(6005.7,acoef[5],2,"Should be very close line ")

    def test_ares_original_py(self):
        pass


#    def test_getlines(self):
#        spectrum='../../sun_harps_ganymede.fits'
#        ll, flux = ares.read_spec(spectrum)
#        ll = ares.correct_lambda_rvpy(ll.copy(), flux, "0,0.1")
#        lc, ld, i1, i2, x, ynorm = getMedida_lines(ll, flux, line, space, rejt, distline)
        

if __name__ == '__main__':
    unittest.main()