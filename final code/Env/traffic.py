"""
This file creates generated user traffic for the cloud environment.

The traffic model has three parts:
1. A day-like traffic pattern.
2. Random request arrivals using a Poisson distribution.
3. Traffic spikes to simulate traffic workloads that behaves like cloud traffic.

The environment uses this traffic to test whether the autoscaling agent
can react to changing demand.
"""

# NumPy will help us to create the smooth wave pattern for normal traffic,
#random number generation,
#and generates a random number of requests based on the expected arrival rate.
import numpy as np


class PoissonTrafficGenerator:
    """Generate request arrivals for the cloud simulator.

    The generator produces an expected arrival rate, called lambda,
    and then samples the actual number of requests from that rate.
    """

    def __init__(
        self,
        base_rate_min: int = 10,
        base_rate_max: int = 80,
        spike_probability: float = 0.05,
        spike_multiplier: float = 3.0,
        spike_duration_min: int = 5,
        spike_duration_max: int = 15,
        period_length: int = 200,
        mode: str = "stochastic",
        seed: int = 0,
    ):
        # Lowest and highest normal traffic rates.
        self.rmin = base_rate_min
        self.rmax = base_rate_max

        # Spike settings: how often spikes happen and how large they are.
        self.p_spike = spike_probability
        self.k_spike = spike_multiplier

        # Minimum and maximum number of timesteps a spike can last.
        self.spike_duration_min = spike_duration_min
        self.spike_duration_max = spike_duration_max

        # Number of timesteps for one complete traffic cycle.
        self.period = period_length

        # Validate spike duration settings.
        if spike_duration_min <= 0 or spike_duration_max < spike_duration_min:
            raise ValueError(
                "spike_duration_min must be positive and <= spike_duration_max"
            )

        # Traffic can be stochastic for training or deterministic for testing.

        # Deterministic mode removes random spikes and Poisson noise so the
        # environment behavior can be tested predictably.

        # Stochastic mode is used for realistic training/evaluation because it
        # includes random arrivals and traffic spikes.
        if mode not in {"stochastic", "deterministic"}:
            raise ValueError("mode must be either 'stochastic' or 'deterministic'")

        self.mode = mode

        # Random generator with a seed so experiments can be repeated.
        self.rng = np.random.default_rng(seed)

        # Number of timesteps remaining in the current spike.
        # If this is 0, there is no active spike.
        self._spike_remaining = 0





    def peek_lambda(self, t: int) -> float:
        #This function gives only the smooth periodic traffic rate.
        #It does not add spikes and does not generate random arrivals.


        #The output is lambda, which represents
        #the expected number of requests at timestep t
        return self.rmin + (self.rmax - self.rmin) * (
            0.5 + 0.5 * np.sin(2 * np.pi * t / self.period)
        )








    def generate(self, t: int) -> tuple:
        """here we generate the actual number of requests at timestep t.

        Returns:
            #arrivals is the actual number of requests
            #lambda is the expected arrival rate used to generate them
        """

        # Start with the normal periodic traffic rate.
        lam = self.peek_lambda(t)

        if self.mode == "stochastic":
            # If a spike is already active, keep applying it.
            if self._spike_remaining > 0:
                lam *= self.k_spike
                self._spike_remaining -= 1

            # If there is no active spike, randomly decide whether to start one.
            elif self.rng.random() < self.p_spike:
                duration = int(
                    self.rng.integers(
                        self.spike_duration_min,
                        self.spike_duration_max + 1,
                    )
                )
                self._spike_remaining = duration - 1
                lam *= self.k_spike

            # Sample the actual request count from a Poisson distribution.
            arrivals = int(self.rng.poisson(lam))

        else:
            # Deterministic mode is used for simple testing.
            arrivals = int(round(lam))

        return arrivals, float(lam)



if __name__ == "__main__":
    gen = PoissonTrafficGenerator(seed=42)
    all_arrivals = []
    spike_steps = 0

    for t in range(500):
        arrivals, lam = gen.generate(t)
        all_arrivals.append(arrivals)

        # If lambda is higher than the normal maximum,
        # then a traffic spike is active.
        if lam > gen.rmax:
            spike_steps += 1

    arr = np.array(all_arrivals)

    print(
        f"Arrivals over 500 steps - min: {arr.min()} "
        f"max: {arr.max()} mean: {arr.mean():.1f}"
    )
    print(f"Spike steps: {spike_steps} / 500")
