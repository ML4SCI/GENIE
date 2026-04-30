import os, sys

# standard module for array manipulation
import numpy as np

# standard module for high-quality plots
import matplotlib.pyplot as plt
# standard module for machine learning
import torch
import torch.nn as nn

from tqdm import tqdm

import scipy.stats as st
import joblib
import matplotlib.patches as mpatches


class qVectorField(nn.Module):
    '''
    Compute a Monte Carlo approximation of the q vector field of the reverse-time
    diffusion equation,

     dx/dt = [(1-s0)*xt - q(t, xt)] / [s0 + (1-s0)*t],

    where xt = x(t) is a d-dimensional vector and q(t, xt) is a d-dimensional
    time-dependent vector field and s0 is that value of sigma(t) at t=0.

    The vector field q(t, xt) is defined by a d-dimensional integral which is
    approximated with a Monte Carlo (MC)-generated sample, x0, of shape (M, d),
    where M is the sample size and d is the dimension of the vector space.

    Example
    -------

    q = qVectorField(x0)
        :  :
    qt = q(t, xt)
    '''
    def __init__(self, x0, sigma0=1e-2, debug=False):

        super().__init__()

        assert(x0.ndim==2)
        # x0.shape: (M, d)

        # change shape of x0 from (M, d) to (1, M, d)
        # so that broadcasting works correctly later.
        self.x0 = x0.unsqueeze(0)

        self.sigma0 = sigma0
        self.debug  = debug

        if debug:
            print('qVectorField.__init__: x0.shape', x0.shape)

    def set_debug(self, debug=True):
        self.debug = debug

    def forward(self, t, xt):

        assert(xt.ndim==2)

        if type(t) == type(xt):
            assert(t.ndim==2)

            # change shape of t so that broadcasting works correctly
            t = t.unsqueeze(1)
            # t.shape: (N, 1) => (N, 1, 1)

        debug = self.debug
        x0 = self.x0
        sigma0 = self.sigma0

        # change shape xt so that broadcasting works correctly
        xt = xt.unsqueeze(1)
        # xt.shape: (N, d) => (N, 1, d)

        if debug:
            print('qVectorField(BEGIN)')
            print('  qVectorField: xt.shape', xt.shape)
            print('  qVectorField: x0.shape', x0.shape)

        alphat = 1 - t
        sigmat = sigma0 + (1 - sigma0) * t
        vt = (xt - alphat * x0) / sigmat

        if torch.isnan(vt).any():
            raise ValueError("vt contains at least one NAN")

        # vt.shape: (N, M, d)
        if debug:
            print('  qVectorField: vt.shape', vt.shape)
            print('  qVectorField: vt', vt)

        # Compute squared distances from each collocation point xt to every
        # MC sample x0, summed over the d feature dimensions.
        # Uses the log-sum-exp trick for numerical stability (avoids exp overflow
        # that occurred in the old implementation which used torch.where clamping).
        v2 = (vt * vt).sum(dim=-1)
        log_weights = -0.5 * v2
        log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
        wt = torch.exp(log_weights)


        # wt.shape: (N, M)
        if torch.isnan(wt).any():
            print('v2')
            print(v2)
            print()
            print('log_weights')
            print(log_weights)
            print()
            print('wt')
            print(wt)
            raise ValueError("wt contains at least one NAN")

        if debug:
            print('  qVectorField: wt.shape', wt.shape)

        # sum over the sample of M weighted elements of x0
        # x0.shape: (1, M, d)
        # wt.shape: (N, M) => (N, M, 1)
        x0_wt = x0 * wt.unsqueeze(-1)
        # x0_wt.shape: (N, M, d)

        if torch.isnan(x0_wt).any():
            raise ValueError("x0_wt contains at least one NAN")

        if debug:
            print('  qVectorField: x0_wt.shape', x0_wt.shape)

        # sum over the MC sample dimension of x0_wt
        qt = x0_wt.sum(dim=1)
        # qt.shape: (N, d)
        if torch.isnan(qt).any():
            raise ValueError("qt contains at least one NAN")

        if debug:
            print('  qVectorField: qt.shape', qt.shape)
            print('qVectorField(END)')

        return qt

    def G(self, t, xt):
        sigma0 = self.sigma0
        sigmat = sigma0 + (1 - sigma0) * t
        qt = self(t, xt)
        return ((1 - sigma0) * xt - qt) / sigmat

    def G_num(self, t, xt):
        sigma0 = self.sigma0
        qt = self(t, xt)
        return ((1 - sigma0) * xt - qt)





def get_normal_sample(x):
    try:
        x = x.cpu()  # we may be on a GPU, so must send to CPU to use numpy
    except:
        pass
    means = np.zeros_like(x)
    scales = np.ones_like(x)
    return torch.Tensor(np.random.normal(loc=means, scale=scales))

def get_target_sample(x, size=4000):
    ii = np.random.randint(0, len(x)-1, size)
    return torch.Tensor(x[ii])




class FlowDE(nn.Module):
    '''
    Given standard normal vectors z = x(t=1), compute target vectors
    x0 = x(t=0) by mapping z to x0 deterministically. x0, which is
    of shape (M, d), where M is the Monte Carlo (MC) sample size and d
    the dimension of the vector x0 = x(0), is used to compute a MC
    approximation of the q vector field. The tensor z is of shape (N, d),
    where N is the number of points sampled from a d-dimensionbal
    standard normal.

    Utility functions
    =================
    1. get_normal_sample(X0) returns a tensor z = x(1), with the same shape
    as X0, whose elements are sampled from a diagonal d-dimensional Gaussian.

    2. get_target_sample(X0, M) returns a sample of points, x0, of size M
    from X0, which will be used to approximate the q vector field.

    Example
    -------
    N = 4000
    M = 4000

    z  = get_normal_sample(X0[:N]).to(DEVICE)
    x0 = get_target_sample(X0, M).to(DEVICE)

    flow = FlowDE(x0)

    y = flow(z)

    '''
    def __init__(self, x0, sigma0=1e-2, T=250, savepath=False, debug=False,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        # x0: MC sample of shape (M, d)
        # T:  number of time steps in [1, 0]

        super().__init__()

        assert(x0.ndim==2)

        self.q = qVectorField(x0, sigma0, debug).to(device)

        if T < 4: T = 4

        self.T = T
        self.h = 1/T # step size
        self.savepath = savepath
        self.debug = debug

    def set_debug(self, debug=True):
        self.debug = debug

    def G(self, t, xt):
        # t is either a float or a 2D tensor of shape (N, 1)
        # xt.shape: (N, d)
        return self.q.G(t, xt)

    def forward(self, z):
        assert(z.ndim==2)

        debug = self.debug

        savepath = self.savepath
        T = self.T
        h = self.h
        t = 1      # initial "time"
        xt= z      # initial "state"

        if debug:
            print('FlowDE.forward: xt.shape', xt.shape)
            print('FlowDE.forward: t', t)
        '''
        if savepath:
            y = [xt]
        '''

        G1 = self.G(t, xt)
        times = []
        y = []
        for i in tqdm(range(T)):
            t -= h

            if t < 0:
                break

            if debug:
                print('FlowDE.forward: t', t)
                print('FlowDE.forward: xt.shape', xt.shape)
                print('FlowDE.forward: G1.shape', G1.shape)

            G2 = self.G(t, xt - G1 * h)

            xt = xt - (G1 + G2) * h / 2

            G1 = G2.detach().clone()

            if savepath:
                y.append(xt.cpu().numpy())
                times.append(t)

        if savepath:
            return y , times
        else:
            return xt
