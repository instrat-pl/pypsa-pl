import numpy as np


def calculate_annuity(lifetime, discount_rate):
    if discount_rate > 0:
        return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)
    else:
        return 1 / lifetime


def modify_vres_availability_profile(
    profile, annual_availability_factor=None, annual_correction_factor=None
):
    """
    cf --> cf * f(cf)
    f(cf) is the correction factor; it is zero at cf = 1
    f(cf) = a + (1 - a) * (2 * cf - 1)
    a is the correction factor at cf = 0.5
    a = (ACF * < cf > - < cf * (2 * cf - 1) >) / (< cf > - < cf * (2 * cf - 1) >)

    profile (cf) is 1 or 2-dimensional array where each column corresponds to a single vRES unit
    annual_availability_factor (ACF) is a number or 1-dimensional array where each element corresponds to a single vRES unit
    """
    assert (annual_availability_factor is None) or (annual_correction_factor is None)

    x = np.mean(profile, axis=0)
    y = np.mean(profile * (2 * profile - 1), axis=0)

    if annual_correction_factor is not None:
        annual_availability_factor = annual_correction_factor * x

    a = (annual_availability_factor - y) / (x - y)
    profile = profile * (a + (1 - a) * (2 * profile - 1))

    # Verify the resulting annual mean is the same as the target one for each vRES unit
    assert np.all(np.abs(np.mean(profile, axis=0) - annual_availability_factor) < 1e-5)

    return profile
