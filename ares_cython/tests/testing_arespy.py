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

if __name__ == '__main__':
    unittest.main()