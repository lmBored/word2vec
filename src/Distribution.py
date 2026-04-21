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
