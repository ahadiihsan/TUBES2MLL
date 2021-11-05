from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from layers.base import Layer

class LSTM_layer(Layer):
    def __init__(self, parameters: Dict, a0: np.array):
            self.parameters = parameters
            self._Wf = parameters["Wf"]
            self._bf = parameters["bf"]
            self._Wi = parameters["Wi"]
            self._bi = parameters["bi"]
            self._Wc = parameters["Wc"]
            self._bc = parameters["bc"]
            self._Wo = parameters["Wo"]
            self._bo = parameters["bo"]
            self._Wy = parameters["Wy"]
            self._by = parameters["by"]
            self.a0 = a0
            self._a_prev, self._c_prev = None
            self._ft, self._it, self._cct, self._ot, self._xt = None
            self._yt_pred
            self._a, self._y, self._c = None
            self.caches = []

    @classmethod
    def initialize(cls, input_size: Tuple[int, int, int], hidden_units: int,) -> LSTM_layer:
        nx,m,tx = input_size
        a0 = np.random.randn(hidden_units,m)
        Wf = np.random.randn(hidden_units, hidden_units+nx)
        bf = np.random.randn(hidden_units,1)
        Wi = np.random.randn(hidden_units, hidden_units+nx)
        bi = np.random.randn(hidden_units,1)
        Wo = np.random.randn(hidden_units, hidden_units+nx)
        bo = np.random.randn(hidden_units,1)
        Wc = np.random.randn(hidden_units, hidden_units+nx)
        bc = np.random.randn(hidden_units,1)
        Wy = np.random.randn(1,hidden_units)
        by = np.random.randn(1,1)
        parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
        return cls(parameters=parameters, a0=a0)

    def forward_pass(self, a_prev: np.array, training: bool) -> np.array:
        self._a_prev = np.array(a_prev, copy=True)
        n_x, m, T_x = self._a_prev.shape
        n_y, n_a = self._Wy.shape

        a = np.zeros([n_a, m, T_x])
        c = np.zeros([n_a, m, T_x])
        y = np.zeros([n_y, m, T_x])

        a_next = self.a0
        c_next = np.zeros([n_a,m])
        for t in range(T_x):
            # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
            a_next, c_next, yt, cache = self.lstm_cell_forward(x[:,:,t], a_next, c_next, self.parameters)
            # Save the value of the new "next" hidden state in a (≈1 line)
            a[:,:,t] = a_next
            # Save the value of the prediction in y (≈1 line)
            y[:,:,t] = yt
            # Save the value of the next cell state (≈1 line)
            c[:,:,t]  = c_next
            # Append the cache into caches (≈1 line)
            self.caches.append(cache)



        # store values needed for backward propagation in cache
        self.caches = (self.caches, x)

        return a, y, c, self.caches

    def lstm_cell_forward(xt, a_prev, c_prev, parameters):
        Wf = parameters["Wf"]
        bf = parameters["bf"]
        Wi = parameters["Wi"]
        bi = parameters["bi"]
        Wc = parameters["Wc"]
        bc = parameters["bc"]
        Wo = parameters["Wo"]
        bo = parameters["bo"]
        Wy = parameters["Wy"]
        by = parameters["by"]
        n_x, m = xt.shape
        n_y, n_a = Wy.shape
        concat = np.zeros([n_x+n_a,m])
        concat[: n_a, :] = a_prev
        concat[n_a :, :] = xt

        ft = sigmoid(np.dot(Wf,concat)+bf)
        it = sigmoid(np.dot(Wi,concat)+bi)
        cct =  np.tanh(np.dot(Wc,concat)+bc)
        c_next = ft * c_prev + it * cct
        ot = sigmoid(np.dot(Wo,concat) + bo)
        a_next = ot*np.tanh(c_next)

        yt_pred = softmax(np.dot(Wy,a_next)+by)

        cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

        return a_next, c_next, yt_pred, cache

    
