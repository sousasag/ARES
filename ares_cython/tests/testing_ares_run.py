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


    def test_ares_original_py(self):
        lines = np.loadtxt("../../linelist.dat", unpack=True, usecols=(0,), skiprows=2)
        specfits = "../../sun_harps_ganymede.fits"
        for line in lines:
            results = ares.get_medida_original(specfits, line)
            print(line,results)
        pass

        

if __name__ == '__main__':
    unittest.main()