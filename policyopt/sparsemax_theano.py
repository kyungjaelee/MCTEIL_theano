import numpy as np
import theano
import theano.tensor as T
from theano import gof

def simplex_projection(z):
    x0 = z.copy()
    d = x0.size

    ind_sort = np.argsort(-x0)
    y0 = x0[ind_sort]

    ycum = np.cumsum(y0)
    val = 1.0 / np.arange(1, d + 1) * (ycum - 1)
    ind = np.nonzero(y0 > val)[0]
    rho = ind[-1]
    tau = val[rho]

    y = y0 - tau
    ind = np.nonzero(y < 0)
    supp = np.nonzero(y >= 0)
    y[ind] = 0

    p = x0.copy()
    p[ind_sort] = y
    return p, tau, supp, 0.5 * (p - z) ** 2

def supporting_set(probs):
    ind = probs.nonzero()[0]
    supp = np.zeros_like(probs)
    supp[ind] = 1.
    return supp

class SparsemaxDistGrad(gof.Op):
    def make_node(self, dy, sm, **kwargs):
        dy = T.as_tensor_variable(dy)
        sm = T.as_tensor_variable(sm)
        return theano.Apply(self, [dy, sm], [sm.type.make_variable()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = np.zeros_like(sm)
        # dx[i,j] = - (\sum_{k in supp[i]} dy[i,k]) / |supp[i]| + dy[i,j]
        for i in xrange(sm.shape[0]):
            # Compute support set supp[i].
            supp_i = supporting_set(sm[i])
            dy_times_supp_i = dy[i] * supp_i
            dx[i] = dy_times_supp_i - sum(dy_times_supp_i) * supp_i / sum(supp_i)
        output_storage[0][0] = dx

    def grad(self, inputs, output_grads):
        dy, sm =inputs
        gdy = T.zeros_like(dy)
        gsm = T.zeros_like(sm)

        return [gdy, gsm]
sparsemaxdistgrad = SparsemaxDistGrad()

class SparsemaxDist(gof.Op):
    itypes = [T.dmatrix]
    otypes = [T.dmatrix]

    def perform(self, node, input_storage, output_storage):
        x = input_storage[0]
        sm = np.zeros_like(x)
        for i in xrange(x.shape[0]):
            sm[i, :], tau, _, _ = simplex_projection(x[i])
        output_storage[0][0] = sm

    def grad(self, inputs, output_grads):
        x, = inputs
        g_sm, = output_grads
        sm = sparsemaxdist(x)
        return [sparsemaxdistgrad(g_sm, sm)]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)
sparsemaxdist = SparsemaxDist()

class Sparsemax(gof.Op):
    itypes = [T.dmatrix]
    otypes = [T.dmatrix]

    def perform(self, node, input_storage, output_storage):
        z = input_storage[0]

        spmax_z = np.zeros_like(z)
        for i in range(z.shape[0]):
            spmax_z_i, tau, _, _ = simplex_projection(z[i,:])
            spmax_z[i,:] += 0.5 * np.sum(z[i][np.abs(spmax_z_i) > 1e-12] ** 2 - tau ** 2)
        spmax_z += 0.5
        output_storage[0][0] = spmax_z

    def grad(self, inputs, output_grads):
        return [sparsemaxdist(inputs[0])]
sparsemax = Sparsemax()