"""
miscellaneous.py

══════════
Miscellaneous helper functions for BiPLOM.
"""

import numpy as np
import pandas as pd
import scipy

try:
    # Numba is used only as a performance accelerator.
    # TRIDENT must remain functional without it.
    from numba import jit  # type: ignore
except Exception:  # pragma: no cover
    def jit(*_args, **_kwargs):
        """Fallback decorator used when numba is unavailable or broken."""

        def _decorator(func):
            return func

        return _decorator
import time

def get_V_old(biadjacency, columns): #Restituisce i V motivs

    if columns:
        biadjacency = np.transpose(biadjacency)

    V_mat = biadjacency.dot(biadjacency.T)
    V_emp = np.sum(V_mat, axis=0)

    return V_emp

def sufficient_decrease_condition(
    f_old, f_new, alpha, grad_f, p, c1=1e-04, c2=0.9
):
    """Returns True if upper wolfe condition is respected.
    :param f_old: Function value at previous iteration.
    :type f_old: float
    :param f_new: Function value at current iteration.
    :type f_new: float
    :param alpha: Alpha parameter of linsearch.
    :type alpha: float
    :param grad_f: Function gradient.
    :type grad_f: numpy.ndarray
    :param p: Current iteration increment.
    :type p: numpy.ndarray
    :param c1: Tuning parameter, defaults to 1e-04.
    :type c1: float, optional
    :param c2: Tuning parameter, defaults to 0.9.
    :type c2: float, optional
    :return: Condition validity.
    :rtype: bool
    """
    sup = f_old + c1 * alpha * np.dot(grad_f, p.T)
    return bool(f_new < sup)

def linsearch_fun(xx, args):
    """Linsearch function for newton and quasinewton methods.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, function f
    :type xx: (numpy.ndarray, numpy.ndarray, float, float, func)
    :param args: Tuple, step function and arguments.
    :type args: (func, tuple)
    :return: Working alpha.
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    beta = xx[2]
    alfa = xx[3]
    f = xx[4]
    step_fun = args[0]
    arg_step_fun = args[1]

    i = 0
    s_old = step_fun(x, arg_step_fun)
    while (
            sufficient_decrease_condition(
                s_old, step_fun(x + alfa * dx, arg_step_fun), alfa, f, dx
            )
            is False
            and i < 50
    ):
        alfa *= beta
        i += 1
    return alfa

def linsearch_fun_fixed(xx):
    """Linsearch function for fixed-point method.
    The function returns the step's size, alpha.
    Alpha determines how much to move on the descending direction
    found by the algorithm.
    :param xx: Tuple of arguments to find alpha:
        solution, solution step, tuning parameter beta,
        initial alpha, step.
    :type xx: (numpy.ndarray, numpy.ndarray, float, float, int)
    :return: Working alpha.
    :rtype: float
    """
    x = xx[0]
    dx = xx[1]
    dx_old = xx[2]
    alfa = xx[3]
    beta = xx[4]
    step = xx[5]

    # eps2 = 1e-2
    # alfa0 = (eps2 - 1) * x / dx
    # for a in alfa0:
    #     if a >= 0:
    #         alfa = min(alfa, a)

    if step:
        kk = 0
        cond = np.linalg.norm(alfa * dx) < np.linalg.norm(dx_old)
        while (
                cond is False
                and kk < 50
        ):
            alfa *= beta
            kk += 1
            cond = np.linalg.norm(alfa * dx) < np.linalg.norm(dx_old)

    return alfa

def matrix_regulariser_function_eigen_based(b, eps):
    """Trasform input matrix in a positive defined matrix
    by regularising eigenvalues.
    :param b: Matrix.
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix.
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    t, e = scipy.linalg.eigh(b)
    ll = np.array([0 if li > eps else eps - li for li in t])
    bf = e @ (np.diag(ll) + np.diag(t)) @ e.transpose()

    return bf

def matrix_regulariser_function(b, eps):
    """Trasforms input matrix in a positive defined matrix
    by adding positive quantites to the main diagonal.
    :param b: Matrix.
    :type b: numpy.ndarray
    :param eps: Positive quantity to add.
    :type eps: float
    :return: Regularised matrix.
    :rtype: numpy.ndarray
    """
    b = (b + b.transpose()) * 0.5  # symmetrization
    bf = b + np.identity(b.shape[0]) * eps

    return bf

def solver(
    x0,
    fun,
    step_fun,
    linsearch_fun = linsearch_fun,
    hessian_regulariser = matrix_regulariser_function_eigen_based,
    fun_jac=None,
    tol=1e-8,
    eps=1e-8,
    max_steps=100,
    method="newton",
    verbose=False,
    regularise=True,
    regularise_eps=1e-3,
    full_return=True,
    linsearch=True,
):
    """Find roots of eq. fun = 0, using newton, quasinewton or
    fixed-point algorithm.
    :param x0: Initial point
    :type x0: numpy.ndarray
    :param fun: Function handle of the function to find the roots of.
    :type fun: function
    :param step_fun: Function to compute the algorithm step
    :type step_fun: function
    :param linsearch_fun: Function to compute the linsearch
    :type linsearch_fun: function
    :param hessian_regulariser: Function to regularise fun hessian
    :type hessian_regulariser: function
    :param fun_jac: Function to compute the hessian of fun, defaults to None
    :type fun_jac: function, optional
    :param tol: The solver stops when \|fun|<tol, defaults to 1e-6
    :type tol: float, optional
    :param eps: The solver stops when the difference between two consecutive
        steps is less than eps, defaults to 1e-10
    :type eps: float, optional
    :param max_steps: Maximum number of steps the solver takes, defaults to 100
    :type max_steps: int, optional
    :param method: Method the solver uses to solve the problem.
        Choices are "newton", "quasinewton", "fixed-point".
        Defaults to "newton"
    :type method: str, optional
    :param verbose: If True the solver prints out information at each step,
         defaults to False
    :type verbose: bool, optional
    :param regularise: If True the solver will regularise the hessian matrix,
         defaults to True
    :type regularise: bool, optional
    :param regularise_eps: Positive value to pass to the regulariser function,
         defaults to 1e-3
    :type regularise_eps: float, optional
    :param full_return: If True the function returns information on the
        convergence, defaults to False
    :type full_return: bool, optional
    :param linsearch: If True a linsearch algorithm is implemented,
         defaults to True
    :type linsearch: bool, optional
    :return: Solution to the optimization problem
    :rtype: numpy.ndarray
    """
    tic_all = time.time()
    toc_init = 0
    tic = time.time()

    # algorithm
    beta = 0.5  # to compute alpha
    n_steps = 0
    x = x0  # initial point

    f = fun(x)
    norm = np.linalg.norm(f)
    diff = 1
    dx_old = np.zeros_like(x)

    if full_return:
        norm_seq = [norm]
        diff_seq = [diff]
        alfa_seq = []

    if verbose:
        print("\nx0 = {}".format(x))
        print("|f(x0)| = {}".format(norm))

    toc_init = time.time() - tic

    toc_alfa = 0
    toc_update = 0
    toc_dx = 0
    toc_jacfun = 0

    tic_loop = time.time()
    while (
        norm > tol and n_steps < max_steps and diff > eps
    ):  # stopping condition

        x_old = x  # save previous iteration

        # f jacobian
        tic = time.time()
        if method == "newton":
            # regularise
            H = fun_jac(x)  # original jacobian
            if regularise:
                b_matrix = hessian_regulariser(
                    H, np.max(np.abs(fun(x))) * regularise_eps,
                )
            else:
                b_matrix = H.__array__()
        elif method == "quasinewton":
            # quasinewton hessian approximation
            b_matrix = fun_jac(x)  # Jacobian diagonal
            if regularise:
                b_matrix = np.maximum(b_matrix,
                                      b_matrix * 0 + np.max(np.abs(fun(x)))
                                      * regularise_eps)
        toc_jacfun += time.time() - tic

        # descending direction computation
        tic = time.time()
        if method == "newton":
            try:
                dx = np.linalg.solve(b_matrix, -f)
            except np.linalg.LinAlgError:
                # Fallback: further regularise the Hessian and retry
                H = fun_jac(x)
                b_matrix = hessian_regulariser(
                    H, max(np.max(np.abs(f)), 1.0) * regularise_eps,
                )
                dx = np.linalg.solve(b_matrix, -f)
        elif method == "quasinewton":
            # Basic update using the diagonal approximation of the Hessian
            dx = -f / b_matrix

            # Guard against division by zero or infinities
            if not np.all(np.isfinite(dx)):
                # Ensure the diagonal is not too small
                if regularise:
                    base = max(np.max(np.abs(f)), 1.0)
                    min_diag = base * regularise_eps
                else:
                    min_diag = 1e-8

                safe_b = np.where(
                    np.abs(b_matrix) < min_diag,
                    np.sign(b_matrix) * min_diag + (b_matrix == 0) * min_diag,
                    b_matrix,
                )
                dx = -f / safe_b
        elif method == "fixed-point":
            dx = f - x
            # TODO: hotfix to compute dx in infty cases
            for i in range(len(x)):
                if x[i] == np.infty:
                    dx[i] = np.infty
        toc_dx += time.time() - tic

        # backtraking line search
        tic = time.time()

        # Linsearch
        if linsearch and (method in ["newton", "quasinewton"]):
            alfa1 = 1
            X = (x, dx, beta, alfa1, f)
            alfa = linsearch_fun(X)
            if full_return:
                alfa_seq.append(alfa)
        elif linsearch and (method in ["fixed-point"]):
            alfa1 = 1
            X = (x, dx, dx_old, alfa1, beta, n_steps)
            alfa = linsearch_fun(X)
            if full_return:
                alfa_seq.append(alfa)
        else:
            alfa = 1

        toc_alfa += time.time() - tic

        tic = time.time()
        # solution update
        # direction= dx@fun(x).T

        x = x + alfa * dx

        dx_old = alfa * dx.copy()

        toc_update += time.time() - tic

        f = fun(x)

        # stopping condition computation
        norm = np.linalg.norm(f)
        diff_v = x - x_old
        # to avoid nans given by inf-inf
        diff_v[np.isnan(diff_v)] = -1
        diff = np.linalg.norm(diff_v)

        if full_return:
            norm_seq.append(norm)
            diff_seq.append(diff)

        # step update
        n_steps += 1

        if verbose:
            print("\nstep {}".format(n_steps))
            # print("fun = {}".format(f))
            # print("dx = {}".format(dx))
            # print("x = {}".format(x))
            print("alpha = {}".format(alfa))
            print("|f(x)| = {}".format(norm))
            print("F(x) = {}".format(step_fun(x)))
            print("diff = {}".format(diff))

    toc_loop = time.time() - tic_loop
    toc_all = time.time() - tic_all

    if verbose:
        print("Number of steps for convergence = {}".format(n_steps))
        print("toc_init = {}".format(toc_init))
        print("toc_jacfun = {}".format(toc_jacfun))
        print("toc_alfa = {}".format(toc_alfa))
        print("toc_dx = {}".format(toc_dx))
        print("toc_update = {}".format(toc_update))
        print("toc_loop = {}".format(toc_loop))
        print("toc_all = {}".format(toc_all))

    if full_return:
        return (x, toc_all, n_steps, np.array(norm_seq),
                np.array(diff_seq), np.array(alfa_seq))
    else:
        return x


def adjacency_list_from_biadjacency(biadjacency, return_inverse=True, return_degree_sequences=True):
    """
    Creates the adjacency list from a biadjacency matrix, given in sparse format or as a list or numpy array.
    Returns the adjacency list as a dictionary, with the rows as keys and as values lists with indexes of the columns.
    If return_inverse is True, the inverse adjacency list is also returned.
    If return_degree_sequences is True, the two degree sequences are also returned.
    """
    if scipy.sparse.isspmatrix(biadjacency):
        if np.sum(biadjacency.data != 1) > 0:
            raise ValueError('Only binary matrices')
        coords = biadjacency.nonzero()
    else:
        biadjacency = np.array(biadjacency)
        if np.sum(biadjacency[biadjacency != 0] != 1) > 0:
            raise ValueError('Only binary matrices')
        coords = np.where(biadjacency != 0)
    biad_shape = np.shape(biadjacency)
    adj_list = {k: set() for k in range(biad_shape[0])}
    inv_adj_list = {k: set() for k in range(biad_shape[1])}
    for edge_i in range(len(coords[0])):
        adj_list[coords[0][edge_i]].add(coords[1][edge_i])
        inv_adj_list[coords[1][edge_i]].add(coords[0][edge_i])
    return_args = [adj_list]
    if return_inverse:
        return_args.append(inv_adj_list)
    if return_degree_sequences:
        rows_degs = np.array([len(adj_list[k]) for k in range(len(adj_list))])
        cols_degs = np.array([len(inv_adj_list[k]) for k in range(len(inv_adj_list))])
        return_args.append(rows_degs)
        return_args.append(cols_degs)
    if len(return_args) > 1:
        return tuple(return_args)
    else:
        return adj_list

def edgelist_from_edgelist_bipartite(edgelist):
    """
    Creates a new edgelist with the indexes of the nodes instead of the names.
    Method for bipartite networks.
    Returns also two dictionaries that keep track of the nodes.
    """
    edgelist = np.array(list(set([tuple(edge) for edge in edgelist])))
    out = np.zeros(np.shape(edgelist)[0], dtype=np.dtype([('source', object), ('target', object)]))
    out['source'] = edgelist[:, 0]
    out['target'] = edgelist[:, 1]
    edgelist = out
    unique_rows, rows_degs = np.unique(edgelist['source'], return_counts=True)
    unique_cols, cols_degs = np.unique(edgelist['target'], return_counts=True)
    rows_dict = dict(enumerate(unique_rows))
    cols_dict = dict(enumerate(unique_cols))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    edgelist_new = [(inv_rows_dict[edge[0]], inv_cols_dict[edge[1]]) for edge in edgelist]
    edgelist_new = np.array(edgelist_new, dtype=np.dtype([('rows', int), ('columns', int)]))
    return edgelist_new, rows_degs, cols_degs, rows_dict, cols_dict

def adjacency_list_from_edgelist_bipartite(edgelist, convert_type=True):
    """
    Creates the adjacency list from the edgelist.
    Method for bipartite networks.
    Returns two dictionaries containing an adjacency list one with the rows as keys and the other with columns as keys.
    If convert_type is True (default), then the nodes are enumerated and the adjacency list is returned as integers.
    Returns also two dictionaries that keep track of the nodes and the two degree sequences.
    """
    rows_degs = None
    cols_degs = None
    rows_dict = None
    cols_dict = None
    if convert_type:
        edgelist, rows_degs, cols_degs, rows_dict, cols_dict = edgelist_from_edgelist_bipartite(edgelist)
    adj_list = {}
    inv_adj_list = {}
    for edge in edgelist:
        adj_list.setdefault(edge[0], set()).add(edge[1])
        inv_adj_list.setdefault(edge[1], set()).add(edge[0])
    if not convert_type:
        rows_degs = np.array([len(adj_list[k]) for k in adj_list])
        rows_dict = {k: k for k in adj_list}
        cols_degs = np.array([len(inv_adj_list[k]) for k in inv_adj_list])
        cols_dict = {k: k for k in inv_adj_list}
    return adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict

def adjacency_list_from_adjacency_list_bipartite(old_adj_list):
    """
    Creates the adjacency list from another adjacency list, converting the data type to integers.
    Method for bipartite networks.
    Returns two dictionaries, each representing an adjacency list with the rows or columns as keys, respectively.
    Original keys are treated as rows, values as columns.
    The nodes are enumerated and the adjacency list contains the related integers.
    Returns also two dictionaries that keep track of the nodes and the two degree sequences.
    """
    rows_dict = dict(enumerate(np.unique(list(old_adj_list.keys()))))
    cols_dict = dict(enumerate(np.unique([el for lst in old_adj_list.values() for el in lst])))
    inv_rows_dict = {v: k for k, v in rows_dict.items()}
    inv_cols_dict = {v: k for k, v in cols_dict.items()}
    adj_list = {}
    inv_adj_list = {}
    for k in old_adj_list:
        adj_list.setdefault(inv_rows_dict[k], set()).update({inv_cols_dict[val] for val in old_adj_list[k]})
        for val in old_adj_list[k]:
            inv_adj_list.setdefault(inv_cols_dict[val], set()).add(inv_rows_dict[k])
    rows_degs = np.array([len(adj_list[k]) for k in range(len(adj_list))])
    cols_degs = np.array([len(inv_adj_list[k]) for k in range(len(inv_adj_list))])
    return adj_list, inv_adj_list, rows_degs, cols_degs, rows_dict, cols_dict

def bicmlo_from_fitnesses(mod, k, h, x, y, w, z):
    """
    Rebuilds the average probability matrix of the bicmlo from the fitnesses
    :param x: the fitness vector of the rows layer
    :type x: numpy.ndarray
    :param y: the fitness vector of the columns layer
    :type y: numpy.ndarray
    """
    n = len(x)
    n1 = len(y)
    p = np.zeros(shape=(n,n1))
    if mod == 4:
        for i in range(n):
            for j in range(n1):
                t_ij = -x[i] - y[j] - h[j] * w[i]
                # same logistic as in biplomIV, but scalar version
                if t_ij >= 0:
                    p[i, j] = 1.0 / (1.0 + np.exp(-t_ij))
                else:
                    et = np.exp(t_ij)
                    p[i, j] = et / (1.0 + et)
    elif mod == 5:
        for i in np.arange(n):
            for j in np.arange(n1):
                p[i,j] = np.exp(-x[i]-y[j]-(h[j]*w[i])-(k[i]*z[j]))/(1 + np.exp(-x[i]-y[j]-(h[j]*w[i])-(k[i]*z[j])))
    return p

def edgelist_from_adjacency_list_bipartite(adj_list):
    """
    Creates the edgelist from an adjacency list given as a dictionary.
    Returns the edgelist as a numpy array, with the keys as first elements of the couples and the values second.
    :param dict adj_list: the adjacency list to be converted.
    """
    edgelist = []
    for k in adj_list:
        for k_neighbor in adj_list[k]:
            edgelist.append((k, k_neighbor))
    return np.array(edgelist)

def biadjacency_from_adjacency_list(adj_list, fmt='array'):
    """
    Creates the biadjacency matrix from an adjacency list given as a dictionary.
    Returns the biadjacency as a numpy array by default, or sparse scipy matrix if fmt='sparse'.
    The biadjacency comes with the keys as rows of the matrix and the values as columns.
    :param dict adj_list: the adjacency list to be converted. Must contain integers that will be used as indexes.
    :param str fmt: the desired format of the output biadjacency matrix, either 'array' or 'sparse', optional
    """
    assert np.isin(fmt, ['array', 'sparse']), 'fmt must be either array or sparse'
    assert isinstance(list(adj_list.keys())[0], (np.number)), 'Adjacency list must be numeric'
    rows_index = [k for k, v in adj_list.items() for _ in range(len(v))]
    cols_index = [i for ids in adj_list.values() for i in ids]
    biad_mat = scipy.sparse.csr_array(([1] * len(rows_index), (rows_index, cols_index)))
    if fmt == 'sparse':
        return biad_mat
    else:
        return biad_mat.toarray()

def biadjacency_from_edgelist(edgelist, fmt='array'):
    """
    Build the biadjacency matrix of a bipartite network from its edgelist.
    Returns a matrix of the type specified by ``fmt``, by default a numpy array.
    """
    edgelist, rows_deg, cols_deg, rows_dict, cols_dict = edgelist_from_edgelist_bipartite(edgelist)
    if fmt == 'array':
        biadjacency = np.zeros((len(rows_deg), len(cols_deg)), dtype=int)
        for edge in edgelist:
            biadjacency[edge[0], edge[1]] = 1
    elif fmt == 'sparse':
        biadjacency = scipy.sparse.coo_matrix((np.ones(len(edgelist)), (edgelist['rows'], edgelist['columns'])))
    elif not isinstance(format, str):
        raise TypeError('format must be a string (either "array" or "sparse")')
    else:
        raise ValueError('format must be either "array" or "sparse"')
    return biadjacency, rows_deg, cols_deg, rows_dict, cols_dict