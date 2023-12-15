import mapclassify

class TestFisherJenks:
    def setup_method(self):
        self.V = load_example()

    def test_FisherJenks(self):
        fj = FisherJenks(self.V)
        assert fj.adcm == 799.24000000000001
        numpy.testing.assert_array_almost_equal(
            fj.bins, numpy.array([75.29, 192.05, 370.5, 722.85, 4111.45])
        )
        numpy.testing.assert_array_almost_equal(
            fj.counts, numpy.array([49, 3, 4, 1, 1])
        )