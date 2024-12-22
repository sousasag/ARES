import unittest
import sys
import os
import numpy as np
sys.path.insert(0,os.path.dirname(repr(__file__).replace("'",""))[:-len("tests")])
import ares_module as ares # type: ignore

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


#    def test_getlines(self):
#        spectrum='../../sun_harps_ganymede.fits'
#        ll, flux = ares.read_spec(spectrum)
#        ll = ares.correct_lambda_rvpy(ll.copy(), flux, "0,0.1")
#        lc, ld, i1, i2, x, ynorm = getMedida_lines(ll, flux, line, space, rejt, distline)
        

if __name__ == '__main__':
    unittest.main()