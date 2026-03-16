'''
The scipy stats package has great functionality for probability distributions
and random variables, but its random number generator is extremely slow
when one samples only one random number at a time. It is extremely fast,
however, when one samples multiple random numbers simultaneously.
For this reason, we have created this class that acts somewhat as a wrapper
around the scipy probability distributions. It will make sure that random
numbers are always generated in batches of n = 10000, and the rvs() function
simply returns the next random number from this list (and resamples when necessary).
'''

import numpy as np


class Distribution :

    n = 10000 # standard random numbers to generate

    '''
    Constructor for this Distribution class.

    Args:
            dist (scipy.stats random variable): A random variable from
            the scipy stats libary.

    Attributes:
            dist (scipy.stats random variable): A random variable from
            the scipy stats libary.
            n (int): a number indicating how many random numbers should
            be generated in one batch
            randomNumbers: a list of n random numbers generated from 'dist'
            idx (int): a number keeping track of how many random numbers
            have been sampled

    '''

    def __init__(self, dist):
        self.dist = dist
        self.resample()

        '''
        Sets the random state of the (internal) random number generator.
        This is typically used to have control over the random seeds.
        '''

    def setRandomState(self, rng):
        self.dist.random_state = rng
        self.resample()


    def __str__(self):
        return str(self.dist)

    def resample(self):
        self.randomNumbers = self.dist.rvs(size=self.n)
        self.idx = 0

    def rvs(self, size=1):
        '''
        A function that returns n (=1 by default) random numbers from
        the specified distribution.

        Returns:
            One random number (float) if n=1, and a list of n random numbers
            otherwise.
        '''
        if self.idx >= self.n - size :
            while size > self.n :
                self.n *= 10
            self.resample()
        if size == 1 :
            rs = self.randomNumbers[self.idx]
        else :
            rs = self.randomNumbers[self.idx:(self.idx + size)]
        self.idx += size
        return rs

    def mean(self):
        return self.dist.mean()

    def std(self):
        return self.dist.std()

    def var(self):
        return self.dist.var()

    def cdf(self, x):
        return self.dist.cdf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def sf(self, x):
        return self.dist.sf(x)

    def ppf(self, x):
        return self.dist.ppf(x)

    def moment(self, n):
        return self.dist.moment(n)

    def median(self):
        return self.dist.median()

    def interval(self, alpha):
        return self.dist.interval(alpha)


class NegativeSampleCache:
    """
    Pre-generates batches of samples from a discrete distribution
    """

    def __init__(self, probs, cache_size=1000000):
        self.probs = np.asarray(probs)
        self.cache_size = cache_size
        self.vocab_size = len(probs)
        self._resample()

    def _resample(self):
        self.cache = np.random.choice(
            self.vocab_size,
            size=self.cache_size,
            p=self.probs
        )
        self.idx = 0

    def sample(self, size):
        if self.idx + size > self.cache_size:
            if size > self.cache_size:
                self.cache_size = size * 2
            self._resample()

        samples = self.cache[self.idx:self.idx + size]
        self.idx += size
        return samples

    def sample_batch(self, batch_size, k):
        total_needed = batch_size * k
        samples = self.sample(total_needed)
        return samples.reshape(batch_size, k)
