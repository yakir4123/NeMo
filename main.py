import numpy as np
from nengo import LIF

from sctn import SCTNNeuron
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import nengo

    with nengo.Network() as model:
        # stim = nengo.Node(lambda t: np.sin(2 * np.pi * t))  # Example input signal
        clk_freq = 1536000
        f0 = 105
        lf = 4
        lp = 144
        w1 = w2 = w3 = w4 = 20
        t1 = t2 = t3 = t4 = -10

        def gen_sine_wave(f):
            def sine(t):
                y = np.sin(2 * np.pi * t * f / clk_freq)
                return y

            return sine

        stim = nengo.Node(gen_sine_wave(f0))
        encoder = nengo.Ensemble(
            label='encoder',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                activation_function="identity",
                inject_voltage=True,
            ),
        )

        shift_45 = nengo.Ensemble(
            label='shift_45',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=lf,
                leakage_period=lp,
                activation_function="identity",
                weights=[w1],
                theta=t1,
                membrane_should_reset=False,
            ),
        )
        shift_90 = nengo.Ensemble(
            label='shift_90',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=lf,
                leakage_period=lp,
                activation_function="identity",
                weights=[w2],
                theta=t2,
                membrane_should_reset=False,
            ),
        )
        shift_135 = nengo.Ensemble(
            label='shift_135',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=lf,
                leakage_period=lp,
                activation_function="identity",
                weights=[w3],
                theta=t3,
                membrane_should_reset=False,
            ),
        )
        shift_180 = nengo.Ensemble(
            label='shift_180',
            n_neurons=1,
            dimensions=1,
            neuron_type=SCTNNeuron(
                leakage_factor=5,
                leakage_period=lp,
                activation_function="identity",
                weights=[w4],
                theta=t4,
                membrane_should_reset=False,
            ),
        )
        nengo.Connection(stim, encoder, synapse=None)
        nengo.Connection(encoder, shift_45, synapse=None)
        nengo.Connection(shift_45, shift_90, synapse=None)
        nengo.Connection(shift_90, shift_135, synapse=None)
        nengo.Connection(shift_135, shift_180, synapse=None)

        probe_spikes = [
            nengo.Probe(encoder.neurons, "output"),
            nengo.Probe(shift_45.neurons, "output"),
            nengo.Probe(shift_90.neurons, "output"),
            nengo.Probe(shift_135.neurons, "output"),
            nengo.Probe(shift_180.neurons, "output"),
        ]

        # IMPORTANT: set dt=1.0 to match the discrete step assumption
        sim = nengo.Simulator(model, dt=1.0)
        samples = int(clk_freq / f0) * 5
        sim.run(samples)  # run 20 discrete steps

        plt.figure()
        for i, probe in enumerate(probe_spikes):
            spikes = sim.data[probe][:, 0]
            cum_spikes = np.convolve(spikes, np.ones(500), mode="valid")
            plt.plot(cum_spikes, label=f"Spikes {i}")
        # plt.plot(sim.trange(), sim.data[probe_time], label='step_time')
        plt.legend()
        plt.show()
