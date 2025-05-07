import random
from typing import Literal

import nengo
import numpy as np
from nengo.params import NumberParam
from nengo.dists import Uniform, Choice

import numpy as np
from nengo.neurons import NeuronType
from nengo.exceptions import ValidationError


class SCTNNeuron(NeuronType):
    """
    Custom implementation of the Spike Continuous Time Neuron (SCTN).
    """

    def __init__(
        self,
        leakage_factor=5,
        leakage_period=1,
        weights: list | None = None,
        theta: float = 0.0,
        threshold=1.0,
        reset_voltage=0.0,
        initial_state=None,
        membrane_should_reset: bool = True,
        activation_function: Literal["binary", "identity"] = "binary",
        inject_voltage: bool = False,
    ):
        """
        Initialize the SCTN neuron model.

        Parameters
        ----------
        lf : float
            Leakage factor (must be a positive integer).
        lp : int
            Leakage period (must be a positive integer).
        threshold : float
            Firing threshold for the neuron.
        reset_voltage : float
            Voltage to reset to after a spike.
        initial_state : dict, optional
            Mapping from state variable names to their initial values.
        """
        super().__init__(initial_state=initial_state)
        weights = weights or [1.0]
        self.voltage = 0
        self.leakage_factor = leakage_factor
        self.leakage_period = leakage_period
        self.weights = np.array(weights)
        self.theta = theta
        self.threshold = threshold
        self.reset_voltage = reset_voltage
        self.inject_voltage = inject_voltage
        self.activation_function = activation_function
        self.membrane_should_reset = membrane_should_reset

        self.leakage_timer = 0
        self.max_clip = 524287
        self.min_clip = -524287
        self.rand_gauss_var = 0
        self.identity_const = 32767
        self.is_initialize = False
        self.normalize_value = 1

    def gain_bias(self, max_rates, intercepts):
        bias = np.array([0])
        gain = np.array([1])
        return gain, bias

    def step(self, dt, J, output, **state):
        """
        Implement the SCTN dynamics in discrete time.
        """
        inp = J
        if not self.is_initialize:
            output[0] = 1
            self.is_initialize = True
            self.normalize_value = inp[0]
            return

        if self.inject_voltage:
            self.voltage = inp[0] * 1000
            inp[0] = 0
        else:
            inp = inp / self.normalize_value

        if self.leakage_factor < 3:
            self.voltage += np.sum(np.multiply(inp, self.weights))
            self.voltage += self.theta
        else:
            lf = 2 ** (self.leakage_factor - 3)
            self.voltage += np.sum(np.multiply(inp, self.weights)) * lf
            self.voltage += self.theta * lf

        self.voltage = np.clip(np.array([self.voltage]), self.min_clip, self.max_clip)[0]

        if self.activation_function == "identity":
            emit_spike = self._activation_function_identity()
        elif self.activation_function == "binary":
            emit_spike = self._activation_function_binary()
        else:
            raise ValueError(
                "Only 2 activation functions are supported [identity, binary]"
            )

        if self.leakage_timer >= self.leakage_period:
            if self.voltage < 0:
                decay_delta = (-self.voltage) / (2**self.leakage_factor)
            else:
                decay_delta = -(self.voltage / (2**self.leakage_factor))
            self.voltage += decay_delta
            self.leakage_timer = 0
        else:
            self.leakage_timer += 1
        output[0] = emit_spike

    def _activation_function_identity(self):
        const = self.identity_const
        c = self.voltage + const

        if self.voltage > const:
            emit_spike = 1
            self.rand_gauss_var = const
        elif self.voltage < -const:
            emit_spike = 0
            self.rand_gauss_var = const
        else:
            self.rand_gauss_var = int(self.rand_gauss_var + c + 1)
            if self.rand_gauss_var >= 65536:
                self.rand_gauss_var -= 65536
                emit_spike = 1
            else:
                emit_spike = 0
        return emit_spike

    def _activation_function_binary(self):
        if self.voltage > self.threshold:
            return 1.0
        return 0.0

    def rates(self, x, gain, bias):
        out = np.zeros_like(x)
        out[0] = 1
        return out

    # def rates(self, x, gain, bias):
    #     """Always returns ``x``."""
    #     J = self.current(x, gain, bias)
    #     out = np.zeros_like(J)
    #     self.step(dt=1.0, J=J, output=out, voltage=np.zeros_like(J))
    #     return out
