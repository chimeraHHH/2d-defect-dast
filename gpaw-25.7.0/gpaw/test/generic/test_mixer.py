from gpaw.mixer import (MixerSum, SeparateSpinMixerDriver, SpinSumMixerDriver,
                        get_mixer_from_keywords)


def test_mixer_sum_with_spin_paired_calc():
    dct = MixerSum(0.1, 3, 10)
    mixer = get_mixer_from_keywords(pbc=[1, 1, 1], nspins=1, **dct)
    assert isinstance(mixer, SeparateSpinMixerDriver)
    mixer = get_mixer_from_keywords(pbc=[1, 1, 1], nspins=2, **dct)
    assert isinstance(mixer, SpinSumMixerDriver)
