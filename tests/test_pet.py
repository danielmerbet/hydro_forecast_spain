"""FAO-56 Penman-Monteith against the textbook worked example.

Allen et al. (1998), FAO Irrigation and Drainage Paper 56, Example 18
(Brussels, 6 July, 50°48'N, 100 m): Tmax 21.5, Tmin 12.3 °C, ea 1.409 kPa,
u2 2.078 m/s, Rn 13.28 MJ m-2 day-1, P 100.1 kPa  ->  ET0 = 3.9 mm/day.
"""
import numpy as np

from hydrocat.pet import fao56_penman_monteith


def test_fao56_example_18():
    ea = 1.409
    x = np.log(ea / 0.6108)
    tdew = 237.3 * x / (17.27 - x)                       # dewpoint giving ea
    u10 = 2.078 / (4.87 / np.log(67.8 * 10 - 5.42))      # back to 10 m
    et0 = fao56_penman_monteith(12.3, 21.5, tdew, 13.28, 100.1, u10)
    assert abs(float(et0) - 3.9) < 0.05


def test_no_negative_et0():
    assert float(fao56_penman_monteith(-10, -5, -8, -3.0, 90, 1.0)) >= 0.0


def test_fao56_net_radiation_example_18():
    """FAO-56 Example 18: Rs 22.07, Tmax 21.5, Tmin 12.3, ea 1.409, 50°48'N,
    6 July (J = 187), 100 m -> Rns 16.99, Rnl 3.71, Rn 13.28 MJ m-2 day-1."""
    from hydrocat.pet import fao56_net_radiation, pressure_from_elevation
    x = np.log(1.409 / 0.6108)
    tdew = 237.3 * x / (17.27 - x)
    rns, rnl_neg = fao56_net_radiation(22.07, 12.3, 21.5, tdew, 50.8, 187, 100)
    assert abs(float(rns) - 16.99) < 0.02
    assert abs(float(-rnl_neg) - 3.71) < 0.05
    assert abs(float(rns + rnl_neg) - 13.28) < 0.06
    assert abs(float(pressure_from_elevation(1800)) - 81.8) < 0.1          # FAO-56 Example 2
