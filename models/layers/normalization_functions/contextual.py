import numpy as np
import tensorflow as tf
from utils import pyutils
from ops import initialization


class ContextualCircuit(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            X,
            model_version='full',
            timesteps=1,
            lesions=None,
            SRF=1,
            SSN=9,
            SSF=29,
            strides=[1, 1, 1, 1]):

        # TODO: DEDUCE SRF/SSN/SSF FROM DATA.
        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in self.X.shape()]
        self.model_version = model_version
        self.timesteps = timesteps
        self.lesions = lesions
        self.SRF = SRF
        self.SSN = SSN
        self.SSF = SSF
        self.SSN_ext = 2 * pyutils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * pyutils.ifloor(SSF / 2.0) + 1
        self.q_shape = [self.SRF, self.SRF, self.k, self.k]
        self.u_shape = [self.SRF, self.SRF, self.k, self.SRF]
        self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
        self.t_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.u_nl = tf.identity
        self.t_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity
        self.i_nl = tf.nn.relu  # input non linearity
        self.o_nl = tf.nn.relu  # output non linearity
        self.normal_initializer = False
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5

    def prepare_tensors(self):
        """ Prepare recurrent weight matrices."""
        # tuned summation: pooling in h, w dimensions
        #############################################
        self.q = tf.get_variable(
            name='q',
            shape=self.q_shape,
            dtype=self.tf.float32,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=None))

        # untuned suppression: reduction across feature axis
        ####################################################
        self.u = tf.get_variable(
            name='u',
            shape=self.u_shape,
            dtype=self.tf.float32,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=None))

        # tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.zeros(self.p_shape)
        for pdx in range(self.k):
            p_array[:self.SSN, :self.SSN, pdx, pdx] = 1.0
        p_array[
            self.SSN // 2 - pyutils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + pyutils.iceil(
                self.SRF / 2.0),
            self.SSN // 2 - pyutils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + pyutils.iceil(
                self.SRF / 2.0),
            :,  # exclude CRF!
            :] = 0.0
        self.p = tf.get_variable(
            name='p',
            shape=self.p_shape,
            dtype=self.tf.float32,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=p_array))

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        t_array = np.zeros(self.t_shape)
        for tdx in range(self.k):
            t_array[tdx, tdx, :self.SSF, :self.SSF] = 1.0
        t_array[
            self.SSF // 2 - pyutils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + pyutils.iceil(
                self.SSN / 2.0),
            self.SSF // 2 - pyutils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + pyutils.iceil(
                self.SSN / 2.0),
            :,  # exclude near surround!
            :] = 0.0
        self.t = tf.get_variable(
            name='t',
            shape=self.t_shape,
            dtype=self.tf.float32,
            initializer=initialization.xavier_initializer(
                            uniform=self.normal_initializer,
                            mask=t_array))

        # Scalar weights
        self.xi = tf.get_variable(shape=[], initializer=1.)
        self.alpha = tf.get_variable(shape=[], initializer=1.)
        self.beta = tf.get_variable(shape=[], initializer=1.)
        self.mu = tf.get_variable(shape=[], initializer=1.)
        self.nu = tf.get_variable(shape=[], initializer=1.)
        self.zeta = tf.get_variable(shape=[], initializer=1.)
        self.gamma = tf.get_variable(shape=[], initializer=1.)
        self.delta = tf.get_variable(shape=[], initializer=1.)
        self.eps_eta = tf.get_variable(shape=[], initializer=1.)
        self.eta = tf.get_variable(shape=[], initializer=1.)
        self.sig_tau = tf.get_variable(shape=[], initializer=1.)
        self.tau = tf.get_variable(shape=[], initializer=1.)

    def convolve_recurrent_RFs(self, O, I):
        """Convolve CRF and eCRF weights with input and output."""
        if 'U' in self.lesions:
            U = tf.constant(0.)
        else:
            U = tf.nn.conv2d(
                O, self._gpu_u, self.strides, padding='SAME')

        if 'T' in self.lesions:
            T = tf.constant(0.)
        else:
            T = tf.nn.conv2d(
                O, self._gpu_t, self.strides, padding='SAME')

        if 'P' in self.lesions:
            P = tf.constant(0.)
        else:
            P = tf.nn.conv2d(
                I, self._gpu_p, self.strides, padding='SAME')

        if 'Q' in self.lesions:
            Q = tf.constant(0.)
        else:
            Q = tf.nn.conv2d(
                I, self._gpu_q, self.strides, padding='SAME')
        return U, T, P, Q

    def full(self, i0, O, I):
        """Fully parameterized contextual RNN model."""
        U, T, P, Q = self.convolve_RFs(
            O=O,
            I=I)

        I_summand = tf.nn.relu(
            (self.xi * self.X)
            - ((self.alpha * I + self.mu) * U)
            - ((self.beta * I + self.nu) * T))

        I = self.tf_eps_eta * I + self.tf_eta * I_summand

        O_summand = tf.nn.relu(
            self.zeta * I
            + self.gamma * P
            + self.delta * Q)
        O = self.tf_sig_tau * O + self.tf_tau * O_summand
        return i0, O, I

    def condition(
            self, i0, O, I):
        """While loop halting condition."""
        return i0 < self.timesteps

    def run(self, in_array):
        """Run the backprop version of the CCircuit."""
        # Using run_reference implementation
        i0 = tf.constant(0)
        O = tf.identity(self.X)
        I = tf.identity(self.X)

        # While loop
        elems = [
            i0,
            O,
            I
        ]

        returned = tf.while_loop(
            self.condition,
            self[self.model_version],
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        return returned
