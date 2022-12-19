"""Some extravagant distributions"""
import typing

import numpy as np
import scipy.stats

min_x, max_x = -12, 12
# pylint: disable=protected-access
DistributionType = typing.Union[scipy.stats.rv_continuous, scipy.stats._distn_infrastructure.rv_frozen]  # type: ignore


class EnhancedSamplingContinuousRV(scipy.stats.rv_continuous):
    """A scipy distribution that is capable of rejection sampling"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rejection_log = []

    def rejection_rvs(self, c: float, envelope: DistributionType):
        """Use the rejection method to sample data"""
        rejections = 0
        while True:
            r = np.random.rand()  # samples a value from the uniform distribution
            x_trial = envelope.rvs()  # samples a value from the enveloping distribution
            if r * c * envelope.pdf(x_trial) <= self._pdf(x_trial):
                self.rejection_log.append(rejections)
                return x_trial
            rejections += 1

    def inverse_transformation_rvs(self):
        """Use the inverse transformation method to sample from the distribution"""
        return self.ppf(scipy.stats.uniform.rvs())


class GGen(EnhancedSamplingContinuousRV):
    """The g(x) distribution from the assignment sheet"""

    normalization = (1 - np.exp(-2)) / 2
    _pdf = np.vectorize(lambda x: np.sin(x) ** 2 / (GGen.normalization * np.pi * (1 + x**2)))


class ConstAndDecGen(EnhancedSamplingContinuousRV):
    """Models a constant and decreasing distribution"""

    x0 = 2
    const = 0.5
    normalization = 2 * (const * x0 + 1 / x0)  # from manual integration of the pdf
    _pdf = np.vectorize(
        lambda x: (ConstAndDecGen.const if abs(x) < ConstAndDecGen.x0 else 1 / x**2) / ConstAndDecGen.normalization
    )
    envelope = scipy.stats.uniform(min_x, max_x - min_x)
    c = const * (max_x - min_x) / normalization

    def _rvs(self, *args, size=None, random_state=None):
        """Use rejection sampling here because the native implementation relies on diverging integration"""
        return self.rejection_rvs(c=self.c, envelope=self.envelope)


class EnhancedCauchyDistribution(scipy.stats.distributions.cauchy_gen, EnhancedSamplingContinuousRV):
    """The Cauchy disribution enhanced with rejection and inverse transform sampling"""


g = GGen(name="g")
cad = ConstAndDecGen(name="const+dec")
ecauchy = EnhancedCauchyDistribution(name="enhanced cauchy")
