class Sampler:

    def sample(self) -> bool:
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()
