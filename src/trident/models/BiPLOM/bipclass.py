"""
biplclass.py

══════════
BipartiteGraph class for BiPLOM-IV.
"""

from scipy.stats import norm, poisson                                               #Check for solutions and reduction
from tqdm import tqdm
from functools import partial
from platform import system
import numpy as np
from .miscellaneous import *
from .biplomIV import *
import scipy

# The BiPLOM implementation historically relied on an external "bicm" package
# to obtain BiCM fitnesses used as an initial guess for the degree multipliers.
# In TRIDENT we use the in-repo implementation located at trident.models.BiCM.
from ..BiCM import bicm_solver

if system() != 'Windows':
    from multiprocessing import Pool

class BipartiteGraph1:
    """Bipartite Graph class for undirected binary bipartite networks.
    This class handles the bipartite graph object to compute the
    Bipartite Configuration Model with Local Overlap (BiCMLO), which can be used as a null model
    for the analysis of undirected and binary bipartite networks.
    The class provides methods for calculating the probabilities and matrices
    of the null model and for projecting the bipartite network on its layers.
    The object can be initialized passing one of the parameters, or the nodes and
    edges can be passed later.
    :param biadjacency: binary input matrix describing the biadjacency matrix
            of a bipartite graph with the nodes of one layer along the rows
            and the nodes of the other layer along the columns.
    :type biadjacency: numpy.array, scipy.sparse, list, optional
    :param adjacency_list: dictionary that contains the adjacency list of nodes.
        The keys are considered as the nodes on the rows layer and the values,
        to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
    :type adjacency_list: dict, optional
    :param edgelist: list of edges containing couples (row_node, col_node) of
        nodes forming an edge. each element in the couples must belong to
        the respective layer.
    :type edgelist: list, numpy.array, optional
    :param degree_sequences: couple of lists describing the degree sequences
        of both layers.
    :type degree_sequences: list, numpy.array, tuple, optional
    """

    def __init__(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None, V_mod=4, method='newton'):
        self.n_rows = None
        self.n_cols = None
        self.r_n_rows = None
        self.r_n_cols = None
        self.shape = None
        self.n_edges = None
        self.r_n_edges = None
        self.n_nodes = None
        self.r_n_nodes = None
        self.biadjacency = None
        self.r_biad_mat = None
        self.edgelist = None
        self.adj_list = None
        self.inv_adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.r_rows_deg = None
        self.r_cols_deg = None
        self.r_rows_ind = None
        self.r_cols_ind = None
        self.rows_dict = None
        self.cols_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_reduced = False
        self.rows_projection = None
        self.avg_mat = None
        self.x = None
        self.y = None
        self.r_x = None
        self.r_y = None
        self.solution_array = None
        self.dict_x = None
        self.dict_y = None
        self.theta_x = None
        self.theta_y = None
        self.r_theta_x = None
        self.r_theta_y = None
        self.r_theta_xy = None
        self.projected_rows_adj_list = None
        self.projected_cols_adj_list = None
        self.v_adj_list = None
        self.projection_method = 'poisson'
        self.threads_num = 4
        self.rows_pvals = None
        self.cols_pvals = None
        self.rows_pvals_mat = None
        self.cols_pvals_mat = None
        self.is_rows_projected = False
        self.is_cols_projected = False
        self.initial_guess = None
        self.method = method
        self.rows_multiplicity = None
        self.cols_multiplicity = None
        self.r_invert_rows_deg = None
        self.r_invert_cols_deg = None
        self.r_dim = None
        self.verbose = False
        self.full_return = False
        self.linsearch = True
        self.regularise = True
        self.tol = None
        self.eps = None
        self.nonfixed_rows = None
        self.fixed_rows = None
        self.full_rows_num = None
        self.nonfixed_cols = None
        self.fixed_cols = None
        self.full_cols_num = None
        self.J_T = None
        self.residuals = None
        self.solution_converged = None
        self.loglikelihood = None
        self.progress_bar = None
        self.pvals_mat = None
        self.exp = True
        self.error = None
        self.V_mod = V_mod            #4 or 5
        self.V_mot = None
        self.l_mot = None
        self.r_V_mot = None
        self.r_l_mot = None
        self.n_V_mot = None
        self.n_l_mot=None
        self.V = None
        self.l = None
        self.r_n_V = None
        self.r_n_l = None
        self.theta_V = None
        self.theta_l = None
        self.r_theta_V = None
        self.r_theta_l = None
        self.dict_V = None
        self.dict_l = None

        self._initialize_graph(biadjacency=biadjacency, adjacency_list=adjacency_list, edgelist=edgelist,
                               degree_sequences=degree_sequences)

    def _initialize_graph(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
        """
        Internal method for the initialization of the graph.
        Use the setter methods instead.
        :param biadjacency: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type biadjacency: numpy.array, scipy.sparse, list, optional
        :param adjacency_list: dictionary that contains the adjacency list of nodes.
            The keys are considered as the nodes on the rows layer and the values,
            to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
        :type adjacency_list: dict, optional
        :param edgelist: list of edges containing couples (row_node, col_node) of
            nodes forming an edge. each element in the couples must belong to
            the respective layer.
        :type edgelist: list, numpy.array, optional
        :param degree_sequences: couple of lists describing the degree sequences
            of both layers.
        :type degree_sequences: list, numpy.array, tuple, optional
        """

        if biadjacency is not None:
            if not isinstance(biadjacency, (list, np.ndarray)) and not scipy.sparse.isspmatrix(biadjacency):
                raise TypeError(
                    'The biadjacency matrix must be passed as a list or numpy array or scipy sparse matrix')
            else:
                if isinstance(biadjacency, list):
                    self.biadjacency = np.array(biadjacency)
                else:
                    self.biadjacency = biadjacency

                if self.biadjacency.shape[0] == self.biadjacency.shape[1]:
                    print(
                        'Your matrix is square. Please remember that it \
                        is treated as a biadjacency matrix, not an adjacency matrix.')

                continuous_weights = not np.all(np.equal(np.mod(self.biadjacency, 1), 0))
                if continuous_weights or np.max(self.biadjacency) > 1:
                    print(
                        'Your matrix is weighted. Please remember that it \
                        is treated as a binary matrix.')
                    self.biadjacency = (self.biadjacency != 0)*1

                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg = \
                        adjacency_list_from_biadjacency(self.biadjacency)
                self.V_mot = get_V_old(self.biadjacency,columns=False)

                if self.V_mod == 5:
                    self.l_mot = get_V_old(self.biadjacency,columns=True)

                self.n_rows = len(self.rows_deg)
                self.n_cols = len(self.cols_deg)
                self._initialize_node_dictionaries()
                self.is_initialized = True

        if self.is_initialized:
            self.n_rows = len(self.rows_deg)
            self.n_cols = len(self.cols_deg)
            self.n_V = len(self.V_mot)
            if self.V_mod == 5:
              self.n_l = len(self.l_mot)
            self.shape = [self.n_rows, self.n_cols]
            self.n_edges = np.sum(self.rows_deg)
            self.n_nodes = self.n_rows + self.n_cols

    def _initialize_node_dictionaries(self):
        self.rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
        self.cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))
        self.inv_rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
        self.inv_cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))

    def degree_reduction(self, rows_deg=None, cols_deg=None):
        """
        Reduce the degree sequences to contain no repetition of the degrees.
        The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
        """

        if rows_deg is None:
            rows_deg = self.rows_deg
        else:
            cols_deg -= self.full_rows_num

        if cols_deg is None:
            cols_deg = self.cols_deg
        else:
            rows_deg -= self.full_cols_num

        unique_rows = np.column_stack((rows_deg, self.V_mot))
        r_rows_deg, self.r_rows_ind, self.r_invert_rows_deg, self.rows_multiplicity \
            = np.unique(unique_rows, return_index=True, return_inverse=True, return_counts=True,axis=0)

        self.r_rows_deg = r_rows_deg[:,0]
        self.r_V_mot = r_rows_deg[:,1]

        if self.V_mod == 4:

            self.r_cols_deg, self.r_cols_ind, self.r_invert_cols_deg, self.cols_multiplicity \
                = np.unique(cols_deg, return_index=True, return_inverse=True, return_counts=True)

        elif self.V_mod == 5:

            unique_columns = np.column_stack((cols_deg, self.l_mot))
            r_cols_deg, self.r_cols_ind, self.r_invert_cols_deg, self.cols_multiplicity \
                = np.unique(unique_columns, return_index=True, return_inverse=True, return_counts=True,axis=0)

            self.r_cols_deg = r_cols_deg[:,0]
            self.r_l_mot = r_cols_deg[:,1]

        self.r_n_rows = self.r_rows_deg.size
        self.r_n_cols = self.r_cols_deg.size
        self.r_n_V = self.r_V_mot.size

        if self.V_mod == 5:
          self.r_n_l = self.r_l_mot.size

        self.r_dim = self.r_n_rows + self.r_n_cols
        self.r_n_edges = np.sum(rows_deg)
        self.is_reduced = True

    def _set_initial_guess(self):
        """
        Internal method to set the initial point of the solver. We use the solution of the BiCM as initial guess for the degree multipliers.
        """
        print('Setting initial guess')
        # Fit BiCM on the full biadjacency to obtain exponential parameters (fitnesses).
        # The solver returns arrays aligned with the original row/column ordering.
        bicm_res = bicm_solver(self.biadjacency, return_probability_matrix=False)
        my_x = np.asarray(bicm_res["x"], dtype=float)
        my_y = np.asarray(bicm_res["y"], dtype=float)
        rows_ok = np.intersect1d(self.nonfixed_rows,self.r_rows_ind)
        cols_ok = np.intersect1d(self.nonfixed_cols,self.r_cols_ind)
        x0 = np.concatenate(( my_x[rows_ok], my_y[cols_ok], np.zeros(len(self.r_V_mot))+np.power((self.r_V_mot/self.n_edges)/(1-(self.r_V_mot/self.n_edges)),self.n_cols/self.n_edges) ))

        if self.V_mod == 5:
            x0 = np.concatenate(( x0, np.zeros(len(self.r_l_mot))+np.power((self.r_l_mot/self.r_n_edges)/(1-(self.r_l_mot/self.r_n_edges)),self.r_n_rows/self.r_n_edges) ))

        if self.exp:
            x0 = -np.log(x0)
        self.x0 = x0

    def initialize_avg_mat(self):
        """
        Reduces the matrix eliminating empty or full rows or columns.
        It repeats the process on the so reduced matrix until no more reductions are possible.
        For instance, a perfectly nested matrix will be reduced until all entries are set to 0 or 1.
        """
        self.avg_mat = np.zeros_like(self.biadjacency, dtype=float)
        r_biad_mat = np.copy(self.biadjacency)
        rows_num, cols_num = self.biadjacency.shape
        rows_degs = self.biadjacency.sum(1)
        cols_degs = self.biadjacency.sum(0)
        good_rows = np.arange(rows_num)
        good_cols = np.arange(cols_num)
        zero_rows = np.where(rows_degs == 0)[0]
        zero_cols = np.where(cols_degs == 0)[0]
        full_rows = np.where(rows_degs == cols_num)[0]
        full_cols = np.where(cols_degs == rows_num)[0]
        self.full_rows_num = 0
        self.full_cols_num = 0
        while zero_rows.size + zero_cols.size + full_rows.size + full_cols.size > 0:
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), zero_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), zero_cols)]
            good_rows = np.delete(good_rows, zero_rows)
            good_cols = np.delete(good_cols, zero_cols)
            full_rows = np.where(r_biad_mat.sum(1) == r_biad_mat.shape[1])[0]
            full_cols = np.where(r_biad_mat.sum(0) == r_biad_mat.shape[0])[0]
            self.full_rows_num += len(full_rows)
            self.full_cols_num += len(full_cols)
            self.avg_mat[good_rows[full_rows][:, None], good_cols] = 1
            self.avg_mat[good_rows[:, None], good_cols[full_cols]] = 1
            good_rows = np.delete(good_rows, full_rows)
            good_cols = np.delete(good_cols, full_cols)
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), full_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), full_cols)]
            zero_rows = np.where(r_biad_mat.sum(1) == 0)[0]
            zero_cols = np.where(r_biad_mat.sum(0) == 0)[0]

        self.nonfixed_rows = good_rows
        self.fixed_rows = np.delete(np.arange(rows_num), good_rows)
        self.nonfixed_cols = good_cols
        self.fixed_cols = np.delete(np.arange(cols_num), good_cols)

        return r_biad_mat

    def _initialize_fitnesses(self):
        """
        Internal method to initialize the fitnesses of the BiCMLO.
        If there are empty rows, the corresponding fitnesses are set to 0,
        while for full rows the corresponding columns are set to numpy.inf.
        """
        if not self.exp:
            self.theta_x = np.zeros(self.n_rows, dtype=float)
            self.theta_y = np.zeros(self.n_cols, dtype=float)
            self.theta_V = np.zeros(self.n_rows, dtype=float)
            self.theta_l = np.zeros(self.n_cols, dtype=float)

        self.x = np.zeros(self.n_rows, dtype=float)
        self.y = np.zeros(self.n_cols, dtype=float)
        self.V = np.zeros(self.n_rows, dtype=float)
        self.l = np.zeros(self.n_cols, dtype=float)
        good_rows = np.arange(self.n_rows)
        good_cols = np.arange(self.n_cols)
        bad_rows = np.array([])
        bad_cols = np.array([])
        self.full_rows_num = 0
        self.full_cols_num = 0
        if np.any(np.isin(self.rows_deg, (0, self.n_cols))) or np.any(np.isin(self.cols_deg, (0, self.n_rows))):
            print('''
                      WARNING: this system has at least a node that is disconnected or connected to all nodes
                       of the opposite layer. This may cause some convergence issues.
                      Please use the full mode providing a biadjacency matrix or an edgelist,
                       or clean your data from these nodes.
                      ''')
            zero_rows = np.where(self.rows_deg == 0)[0]
            zero_cols = np.where(self.cols_deg == 0)[0]
            if not self.exp:
                self.theta_x[zero_rows] = np.inf
                self.theta_y[zero_cols] = np.inf
                self.theta_V[zero_rows] = np.inf
                self.theta_l[zero_cols] = np.inf

            full_rows = np.where(self.rows_deg == self.n_cols)[0]
            full_cols = np.where(self.cols_deg == self.n_rows)[0]
            self.x[full_rows] = np.inf
            self.y[full_cols] = np.inf
            self.V[full_rows] = np.inf
            self.l[full_cols] = np.inf

            if not self.exp:
                self.theta_x[full_rows] = - np.inf
                self.theta_y[full_cols] = - np.inf
                self.theta_V[full_rows] = - np.inf
                self.theta_l[full_cols] = - np.inf

            bad_rows = np.concatenate((zero_rows, full_rows))
            bad_cols = np.concatenate((zero_cols, full_cols))
            good_rows = np.delete(np.arange(self.n_rows), bad_rows)
            good_cols = np.delete(np.arange(self.n_cols), bad_cols)
            self.full_rows_num += len(full_rows)
            self.full_cols_num += len(full_cols)

        self.nonfixed_rows = good_rows
        self.fixed_rows = bad_rows
        self.nonfixed_cols = good_cols
        self.fixed_cols = bad_cols

    def _initialize_problem(self, rows_deg=None, cols_deg=None):
        """
        Initializes the solver reducing the degree sequences,
        setting the initial guess and setting the functions for the solver.
        The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
        """
        if ~self.is_reduced:
            self.degree_reduction(rows_deg=rows_deg, cols_deg=cols_deg)  # if weighted rows_deg=rows_seq
        self._set_initial_guess()

        if self.V_mod == 4:
            self.args = (self.r_rows_deg, self.r_cols_deg, self.r_V_mot, self.rows_multiplicity, self.cols_multiplicity, self.n_rows, self.n_cols)
            d_fun = {
                    'newton_exp': lambda x: loglikelihood_prime_IV(x, self.args),
                    'quasinewton_exp': lambda x: loglikelihood_prime_IV(x, self.args),
                }
            d_fun_jac = {
                    'newton_exp': lambda x: loglikelihood_hessian_IV(x, self.args),
                    'quasinewton_exp': lambda x: loglikelihood_hessian_diag_IV(x, self.args),
                }
            d_fun_step = {
                    'newton_exp': lambda x: loglikelihood_IV(x, self.args),
                    'quasinewton_exp': lambda x: loglikelihood_IV(x, self.args),
                }

        if self.exp:
            self.hessian_regulariser = matrix_regulariser_function_eigen_based
        else:
            self.hessian_regulariser = matrix_regulariser_function

        if self.exp:
            method = self.method + '_exp'
        else:
            method = self.method

        # lins_args = (d_fun_step[method], self.args)
        if self.exp:
            lins_args = (loglikelihood_IV, self.args)

        lins_fun = {
            'newton_exp': lambda x: linsearch_fun(x, lins_args),
            'quasinewton_exp': lambda x: linsearch_fun(x, lins_args),
            'fixed-point_exp': lambda x: linsearch_fun_fixed(x),
        }

        try:
            self.fun = d_fun[method]
            self.fun_jac = d_fun_jac[method]
            self.step_fun = d_fun_step[method]
            self.fun_linsearch = lins_fun[method]
        except (TypeError, KeyError):
            raise ValueError('Method must be "newton", "quasinewton" or "fixed-point".')

    @staticmethod
    def check_sol(biad_mat, avg_bicmlo, V_mot = None, l_mot = None, return_error=False, in_place=False):
        """
        Static method.
        This function prints the rows sums differences between two matrices,
        that originally are the biadjacency matrix and its bicmlo average matrix.
        The intended use of this is to check if an average matrix is actually a solution
         for a bipartite configuration model.
        If return_error is set to True, it returns 1 if the sum of the differences is bigger than 1.
        If in_place is set to True, it checks and sums also the total error entry by entry.
        The intended use of this is to check if two solutions are the same solution.
        """
        error = 0
        if np.any(avg_bicmlo < 0):
            print('Negative probabilities in the average matrix!')
            error = 1
        if np.any(avg_bicmlo > 1):
            print('Probabilities greater than 1 in the average matrix!')
            error = 1
        rows_sums = np.sum(biad_mat, axis=1)
        cols_sums = np.sum(biad_mat, axis=0)
        rows_error_vec = np.abs(rows_sums - np.sum(avg_bicmlo, axis=1))
        cols_error_vec = np.abs(cols_sums - np.sum(avg_bicmlo, axis=0))
        V_sums = get_V_old(avg_bicmlo,columns=False)
        V_error_vec = np.abs(V_sums - V_mot)
        err_rows = np.max(rows_error_vec)
        err_cols = np.max(cols_error_vec)
        err_V = np.max(V_error_vec)
        print('max rows error =', err_rows)
        print('max columns error =', err_cols)
        print('max V error =', err_V)

        if l_mot:
            l_sums = get_V_old(avg_bicmlo,columns=True)
            l_error_vec = np.abs(l_sums - l_mot)
            err_l = np.max(l_error_vec)
            print('max l error =', err_l)
            err_V = max(err_V,err_l)


        max_err = np.max([err_rows, err_cols])
        if max_err > 1e-1:
            # error = 1
            print('WARNING max rel degree error > 0.1')

        if err_V > 1e-1:
            print('WARNING max rel V error > 0.1')

        if in_place:
            diff_mat = np.abs(biad_mat - avg_bicmlo)
            print('In-place total error:', np.sum(diff_mat))
            print('In-place max error:', np.max(diff_mat))
        if return_error:
            if error == 1:
                return error
            else:
                return max_err
        else:
            return

    def _check_solution(self, return_error=False, in_place=False):
        """
        Check if the solution of the BiCMLO is compatible with the degree sequences of the graph.
        :param bool return_error: If this is set to true, return 1 if the solution is not correct, 0 otherwise.
        :param bool in_place: check also the error in the single entries of the matrices.
            Always False unless comparing two different solutions.
        """
        if self.biadjacency is not None and self.avg_mat is not None:
            return self.check_sol(self.biadjacency, self.avg_mat, V_mot=self.V_mot, l_mot=self.l_mot, return_error=return_error, in_place=in_place)
        #else:
            #return self.check_sol_light(return_error=return_error)

    def _set_solved_problem(self, solution):
        """
        Sets the solution of the problem.
        :param numpy.ndarray solution: A numpy array containing that reduced fitnesses of the two layers, consecutively.
        """
        if not self.exp:
            if self.theta_x is None:
                self.theta_x = np.zeros(self.n_rows)
                self.theta_x[:] = np.inf
            if self.theta_y is None:
                self.theta_y = np.zeros(self.n_cols)
                self.theta_y[:] = np.inf
            if self.theta_V is None:
                self.theta_V = np.zeros(self.n_rows)
                self.theta_V[:] = np.inf

            self.r_theta_xy = solution
            self.r_theta_x = self.r_theta_xy[:self.r_n_rows]
            self.r_theta_y = self.r_theta_xy[self.r_n_rows:self.r_n_rows+self.r_n_cols]
            self.r_theta_V = self.r_theta_xy[self.r_n_rows+self.r_n_cols:self.r_n_rows+self.r_n_rows+self.r_n_cols]
            self.solution_array = np.exp(- self.r_theta_xy)
            self.r_x = self.r_theta_x
            self.r_y = self.r_theta_y
            self.r_V_mot = self.r_theta_V
            self.theta_x[self.nonfixed_rows] = self.r_theta_x[self.r_invert_rows_deg]
            self.theta_y[self.nonfixed_cols] = self.r_theta_y[self.r_invert_cols_deg]
            self.theta_V[self.nonfixed_rows] = self.r_theta_V[self.r_invert_rows_deg]

            if self.V_mod == 5:
                if self.theta_l is None:
                    self.theta_l = np.zeros(self.n_cols)
                    self.theta_l[:] = np.inf
                self.r_theta_l = self.r_theta_xy[self.r_n_rows+self.r_n_rows+self.r_n_cols:]
                self.r_l_mot = self.r_theta_l
                self.theta_l[self.nonfixed_cols] = self.r_theta_l[self.r_invert_cols_deg]

        else:
            if self.theta_x is None:
                self.theta_x = np.zeros(self.n_rows)
                self.theta_x[:] = np.inf
            if self.theta_y is None:
                self.theta_y = np.zeros(self.n_cols)
                self.theta_y[:] = np.inf
            if self.theta_V is None:
                self.theta_V = np.zeros(self.n_rows)
                self.theta_V[:] = np.inf

            self.r_theta_xy = solution
            self.r_theta_x = self.r_theta_xy[:self.r_n_rows]
            self.r_theta_y = self.r_theta_xy[self.r_n_rows:self.r_n_rows+self.r_n_cols]
            self.r_theta_V = self.r_theta_xy[self.r_n_rows+self.r_n_cols:self.r_n_rows+self.r_n_rows+self.r_n_cols]
            self.solution_array = np.exp(- self.r_theta_xy)
            self.r_x = np.exp(- self.r_theta_x)
            self.r_y = np.exp(- self.r_theta_y)
            self.r_V_mot = np.exp(- self.r_theta_V)
            self.theta_x[self.nonfixed_rows] = self.r_theta_x[self.r_invert_rows_deg]
            self.theta_y[self.nonfixed_cols] = self.r_theta_y[self.r_invert_cols_deg]
            self.theta_V[self.nonfixed_rows] = self.r_theta_V[self.r_invert_rows_deg]

            if self.V_mod == 5:
                if self.theta_l is None:
                    self.theta_l = np.zeros(self.n_cols)
                    self.theta_l[:] = np.inf
                self.r_theta_l = self.r_theta_xy[self.r_n_rows+self.r_n_rows+self.r_n_cols:]
                self.r_l_mot = np.exp(- self.r_theta_l)
                self.theta_l[self.nonfixed_cols] = self.r_theta_l[self.r_invert_cols_deg]

        if self.x is None:
            self.x = np.zeros(self.n_rows)
        if self.y is None:
            self.y = np.zeros(self.n_cols)
        if self.V is None:
            self.V = np.zeros(self.n_rows)

        self.x[self.nonfixed_rows] = self.r_x[self.r_invert_rows_deg]
        self.y[self.nonfixed_cols] = self.r_y[self.r_invert_cols_deg]
        self.V[self.nonfixed_rows] = self.r_V_mot[self.r_invert_rows_deg]
        self.dict_x = dict([(self.rows_dict[i], self.x[i]) for i in range(len(self.x))])
        self.dict_y = dict([(self.cols_dict[j], self.y[j]) for j in range(len(self.y))])
        self.dict_V = dict([(self.rows_dict[i], self.V[i]) for i in range(len(self.x))])

        if self.V_mod == 5:
            if self.l is None:
                self.l = np.zeros(self.n_cols)
            self.l[self.nonfixed_cols] = self.r_l_mot[self.r_invert_cols_deg]
            self.dict_l = dict([(self.cols_dict[j], self.l[j]) for j in range(len(self.l))])

        if self.full_return:
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        # Reset solver lambda functions for multiprocessing compatibility
        self.hessian_regulariser = None
        self.fun = None
        self.fun_jac = None
        self.step_fun = None
        self.fun_linsearch = None

    def _solve_bicmlo_full(self):
        """
        Internal method for computing the solution of the BiCMLO via matrices.
        """
        r_biadjacency = self.initialize_avg_mat()
        if len(r_biadjacency) > 0:  # Every time the matrix is not perfectly nested

            rows_deg = self.rows_deg[self.nonfixed_rows]
            cols_deg = self.cols_deg[self.nonfixed_cols]
            self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)

            x0 = self.x0
            sol = solver(
                x0,
                fun=self.fun,
                fun_jac=self.fun_jac,
                step_fun=self.step_fun,
                linsearch_fun=self.fun_linsearch,
                hessian_regulariser=self.hessian_regulariser,
                tol=self.tol,
                eps=self.eps,
                max_steps=self.max_steps,
                method=self.method,
                verbose=self.verbose,
                regularise=self.regularise,
                full_return=self.full_return,
                linsearch=self.linsearch,
            )
            self._set_solved_problem(sol)

            if self.V_mod == 4:
                r_avg_mat = bicmlo_from_fitnesses(self.V_mod, self.rows_deg[self.nonfixed_rows], self.cols_deg[self.nonfixed_cols], self.theta_x[self.nonfixed_rows], self.theta_y[self.nonfixed_cols], self.theta_V[self.nonfixed_rows], None)
            if self.V_mod == 5:
                r_avg_mat = bicmlo_from_fitnesses(self.V_mod, self.rows_deg[self.nonfixed_rows], self.cols_deg[self.nonfixed_cols], self.theta_x[self.nonfixed_rows], self.theta_y[self.nonfixed_cols], self.theta_V[self.nonfixed_rows], self.theta_l[self.nonfixed_cols])



            self.avg_mat[self.nonfixed_rows[:, None], self.nonfixed_cols] = np.copy(r_avg_mat)


    def _solve_bicmlo_light(self):
        """
        Internal method for computing the solution of the BiCMLO via degree sequences.
        """
        self._initialize_fitnesses()
        rows_deg = self.rows_deg[self.nonfixed_rows]
        cols_deg = self.cols_deg[self.nonfixed_cols]
        self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
        x0 = self.x0
        sol = solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=self.tol,
            eps=self.eps,
            max_steps=self.max_steps,
            method=self.method,
            verbose=self.verbose,
            regularise=self.regularise,
            full_return=self.full_return,
            linsearch=self.linsearch,
        )
        self._set_solved_problem(sol)


    def _set_parameters(self, method, initial_guess, tol, eps, regularise, max_steps, verbose, linsearch, exp,
                        full_return):
        """
        Internal method for setting the parameters of the solver.
        """
        self.method = method
        self.initial_guess = initial_guess
        if tol is None:
            self.tol = 1e-8
        else:
            self.tol = tol
        if eps is None:
            self.eps = 1e-8
        else:
            self.eps = tol
        self.verbose = verbose
        self.linsearch = linsearch
        self.regularise = regularise
        self.exp = exp
        self.full_return = full_return
        if max_steps is None:
            if method == 'fixed-point':
                self.max_steps = 200
            else:
                self.max_steps = 100
        else:
            self.max_steps = max_steps

    def solve_tool(
            self,
            method=None,
            initial_guess=None,
            light_mode=None,
            tol=None,
            eps=None,
            max_steps=None,
            verbose=False,
            linsearch=True,
            regularise=None,
            print_error=True,
            full_return=False,
            exp=True,
            model=None):
        """Solve the BiCMLO of the graph.
        It does not return the solution, use the getter methods instead.
        :param bool light_mode: Doesn't use matrices in the computation if this is set to True.
            If the graph has been initialized without the matrix, the light mode is used regardless.
        :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is set by the model
        solved
        :param str initial_guess: Initial guess of choice among *None*, *random*, *uniform* or *degrees*,
        default is None
        :param float tol: Tolerance of the solution, optional
        :param float eps: Tolerance of the difference between consecutive solutions, optional
        :param int max_steps: Maximum number of steps, optional
        :param bool, optional verbose: Print elapsed time, errors and iteration steps, optional
        :param bool linsearch: Implement the linesearch when searching for roots, default is True
        :param bool regularise: Regularise the matrices in the computations, optional
        :param bool print_error: Print the final error of the solution
        :param bool full_return: If True, the solver returns some more insights on the convergence. Default False.
        :param bool exp: if this is set to true the solver works with the reparameterization $x_i = e^{-\theta_i}$,
            $y_\alpha = e^{-\theta_\alpha}$. It might be slightly faster but also might not converge.
        :param str model: Model to be used, to be passed only if the user wants to use a different model
        than the recognized one.
        """
        if model == 'BiPLOM IV':
            self.V_mod = 4
        elif model == 'BiPLOM V':
            self.V_mod = 5
        method = self.method
        if method is None:
            method = 'newton'
        if not self.is_initialized:
            print('Graph is not initialized. I can\'t compute the BiCMLO.')
            return
        if regularise is None:
            if method in ("newton", "quasinewton"):
                regularise = True
            else:
                # fixed-point does not use the Hessian in the same way
                regularise = not exp

        if regularise and exp and method in ("newton", "quasinewton"):
            print("Using Hessian regularisation in exp mode for numerical stability.")

        self._set_parameters(
            method=method,
            initial_guess=initial_guess,
            tol=tol,
            eps=eps,
            regularise=regularise,
            max_steps=max_steps,
            verbose=verbose,
            linsearch=linsearch,
            exp=exp,
            full_return=full_return,
        )
        if self.biadjacency is not None and (light_mode is None or not light_mode):
            self._solve_bicmlo_full()
        else:
            if light_mode is False:
                print('''
                I cannot work with the full mode without the biadjacency matrix.
                This will not account for disconnected or fully connected nodes.
                Solving in light mode...
                ''')
            self._solve_bicmlo_light()
        if print_error:
            max_err = self._check_solution(return_error=True)

            # Treat non-finite errors (NaN or Inf) as non-convergence
            if (max_err is None) or (not np.isfinite(max_err)):
                self.solution_converged = False
                print('Solver did not converge: non-finite error (NaN or Inf).')
            else:
                if max_err >= 0.001:
                    self.solution_converged = False
                else:
                    self.solution_converged = True

                if self.solution_converged:
                    print('Solver converged.')
                else:
                    if max_err >= 1:
                        print('Solver did not converge: error', max_err)
                    else:
                        print('Solver finished with an error of {:.2f}%'.format(max_err * 100))

            self.error = max_err
        self.is_randomized = True

    def get_bicmlo_matrix(self):
        """Get the matrix of probabilities of the BiCMLO.
        If the BiCMLO has not been computed, it also computes it with standard settings.
        :returns: The average matrix of the BiCMLO
        :rtype: numpy.ndarray
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCMLO')
        elif not self.is_randomized:
            self.solve_tool(method=self.method)
        if self.avg_mat is not None:
            return self.avg_mat
        else:

            if self.V_mod == 4:
                r_avg_mat = bicmlo_from_fitnesses(self.V_mod, self.r_rows_deg, self.r_cols_deg, self.theta_x[self.nonfixed_rows], self.theta_y[self.nonfixed_cols], self.theta_V[self.nonfixed_rows], None)
            if self.V_mod == 5:
                r_avg_mat = bicmlo_from_fitnesses(self.V_mod, self.r_rows_deg, self.r_cols_deg, self.theta_x[self.nonfixed_rows], self.theta_y[self.nonfixed_cols], self.theta_V[self.nonfixed_rows], self.theta_l[self.nonfixed_cols])

            return self.avg_mat

    def get_bicmlo_fitnesses(self):
        """Get the fitnesses of the BiCMLO.
        If the BiCMLO has not been computed, it also computes it with standard settings.
        :returns: The fitnesses of the BiCMLO in the format **rows fitnesses dictionary, columns fitnesses dictionary**
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCMLO')
        elif not self.is_randomized:
            print('Computing the BiCMLO...')
            self.solve_tool(method=self.method)
        return self.dict_x, self.dict_y, self.dict_V, self.dict_l

    def get_fitnesses(self):
        """See get_bicmlo_fitnesses."""
        self.get_bicmlo_fitnesses()

    def pval_calculator(self, v_list_key, x, y, V):
        """
        Calculate the p-values of the v-motifs numbers of one vertex and all its neighbours.
        :param int v_list_key: the key of the node to consider for the adjacency list of the v-motifs.
        :param numpy.ndarray x: the fitnesses of the layer of the desired projection.
        :param numpy.ndarray y: the fitnesses of the opposite layer.
        :returns: a dictionary containing as keys the nodes that form v-motifs with the
        considered node, and as values the corresponding p-values.
        """
        node_xy = x[v_list_key] * y * np.power(self.V[v_list_key],self.cols_deg)
        temp_pvals_dict = {}
        for neighbor1 in self.v_adj_list[v_list_key].keys():
            neighbor1_xy = x[neighbor1] * y * np.power(self.V[neighbor1],self.cols_deg)
            temp_pvals_dict[neighbor1] = {}
            for neighbor2 in self.v_adj_list[v_list_key][neighbor1].keys():
                neighbor2_xy = x[neighbor2] * y * np.power(self.V[neighbor2],self.cols_deg)

                probs = node_xy * neighbor1_xy * neighbor2_xy / ((1 + node_xy) * (1 + neighbor1_xy) * (1 + neighbor2_xy))
                avg_v = np.sum(probs)
                if self.projection_method == 'poisson':
                    temp_pvals_dict[neighbor1][neighbor2] = \
                        poisson.sf(k=self.v_adj_list[v_list_key][neighbor1][neighbor2] - 1, mu=avg_v)
                elif self.projection_method == 'normal':
                    sigma_v = np.sqrt(np.sum(probs * (1 - probs)))
                    temp_pvals_dict[neighbor1][neighbor2] = \
                        norm.cdf((self.v_adj_list[v_list_key][neighbor1][neighbor2] + 0.5 - avg_v) / sigma_v)
                elif self.projection_method == 'rna':
                    var_v_arr = probs * (1 - probs)
                    sigma_v = np.sqrt(np.sum(var_v_arr))
                    gamma_v = (sigma_v ** (-3)) * np.sum(var_v_arr * (1 - 2 * probs))
                    eval_x = (self.v_adj_list[v_list_key][neighbor1][neighbor2] + 0.5 - avg_v) / sigma_v
                    pval = norm.cdf(eval_x) + gamma_v * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
                    temp_pvals_dict[neighbor1][neighbor2] = max(min(pval, 1), 0)
                elif self.projection_method == 'poibin':
                    pb_obj = PoiBin(probs)
                    temp_pvals_dict[neighbor1][neighbor2] = pb_obj.pval(int(self.v_adj_list[v_list_key][neighbor1][neighbor2]))
        return temp_pvals_dict

    def _projection_calculator(self):

        if self.rows_projection:
            adj_list = self.adj_list
            inv_adj_list = self.inv_adj_list
            x = self.x
            y = self.y
            V = self.V
        else:
            adj_list = self.inv_adj_list
            inv_adj_list = self.adj_list
            x = self.y
            y = self.x
            V = self.l

        v_adj_list = {k: dict() for k in list(adj_list.keys())}

        for node_i in np.sort(list(self.adj_list.keys())):
            for neighbor in self.adj_list[node_i]:
                l_inv_adj_list = np.sort(np.array(list(self.inv_adj_list[neighbor])))
                for ind_node_j in range(0,len(l_inv_adj_list)):
                    if l_inv_adj_list[ind_node_j] > node_i:
                        flag = False
                        v_adj_list[node_i][l_inv_adj_list[ind_node_j]] = dict()
                        for ind_node_k in range(ind_node_j+1,len(l_inv_adj_list)):
                            v_adj_list[node_i][l_inv_adj_list[ind_node_j]][l_inv_adj_list[ind_node_k]] = len(self.adj_list[node_i].intersection(self.adj_list[l_inv_adj_list[ind_node_j]], self.adj_list[l_inv_adj_list[ind_node_k]]))
                            flag = True
                        if not flag:
                            del v_adj_list[node_i][l_inv_adj_list[ind_node_j]]

        self.v_adj_list = v_adj_list
        v_list_keys = list(self.v_adj_list.keys())
        pval_adj_list = dict()
        if self.threads_num > 1:
            with Pool(processes=self.threads_num) as pool:
                partial_function = partial(self.pval_calculator, x=x, y=y, V=V)
                if self.progress_bar:
                    pvals_dicts = pool.map(partial_function, tqdm(v_list_keys))
                else:
                    pvals_dicts = pool.map(partial_function, v_list_keys)
            for k_i in range(len(v_list_keys)):
                k = v_list_keys[k_i]
                pval_adj_list[k] = pvals_dicts[k_i]
        else:
            if self.progress_bar:
                for k in tqdm(v_list_keys):
                    pval_adj_list[k] = self.pval_calculator(k, x=x, y=y, V=V)
            else:
                for k in v_list_keys:
                    pval_adj_list[k] = self.pval_calculator(k, x=x, y=y, V=V)

        return pval_adj_list

    def compute_projection(self, rows=True, alpha=0.05, approx_method=None, method=None,
                           threads_num=None, progress_bar=True, validation_method='fdr'):
        """Compute the projection of the network on the rows or columns layer.
        If the BiCMLO has not been computed, it also computes it with standard settings.
        This is the most customizable method for the pvalues computation.
        :param bool rows: True if requesting the rows' projection.
        :param float alpha: Threshold for the p-values validation.
        :param str approx_method: Method for the approximation of the pvalues computation.
            Implemented methods are *poisson*, *poibin*, *normal*, *rna*.
        :param str method: Deprecated, same as approx_method.
        :param threads_num: Number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :param bool progress_bar: Show progress bar of the pvalues computation.
        :param str validation_method:  The type of validation to apply: 'global' for a global threshold,
         'fdr' for False Discovery Rate or 'bonferroni' for Bonferroni correction.
        """
        if approx_method is None:
            if method is not None:
                print('"method" is deprecated, use approx_method instead')
                approx_method = method
            else:
                approx_method = 'poisson'
        self.rows_projection = rows
        self.projection_method = approx_method
        self.progress_bar = progress_bar
        if threads_num is None:
            if system() == 'Windows':
                threads_num = 1
            else:
                threads_num = 4
        else:
            if system() == 'Windows' and threads_num != 1:
                threads_num = 1
                print('Parallel processes not yet implemented on Windows, computing anyway...')
        self.threads_num = threads_num
        if self.adj_list is None and self.biadjacency is None:
            print('''
            Without the edges I can't compute the projection.
            Use set_biadjacency_matrix, set_adjacency_list or set_edgelist to add edges.
            ''')
            return
        else:
            if not self.is_randomized:
                print('First I have to compute the BiCMLO. Computing...')
                self.solve_tool(method=self.method)
            if rows:
                self.rows_pvals = self._projection_calculator()
                self.projected_rows_adj_list = self._projection_from_pvals(alpha=alpha,
                                                                           validation_method=validation_method)
                self.is_rows_projected = True

            else:
                self.cols_pvals = self._projection_calculator()
                self.projected_cols_adj_list = self._projection_from_pvals(alpha=alpha,
                                                                           validation_method=validation_method)
                self.is_cols_projected = True

    def get_validated_matrix(self, significance=0.01, validation_method=None):
        """
        Extract a backbone of the original network keeping only the most significant links.
        At the moment this method only applies a global significance level for any link.
        :param float significance:  Threshold for the pvalues significance.
        :param str validation_method:  The type of validation to apply: 'global' for a global threshold,
         'fdr' for False Discovery Rate or 'bonferroni' for Bonferroni correction.
        """
        if self.pvals_mat is None:
            self.compute_weighted_pvals_mat()
        if validation_method is None:
            print('FDR validated matrix will be returned by default. '
                  'Set validation_method to "global" or "bonferroni" for alternatives or use .get_weighted_pvals_mat.')
            validation_method = 'fdr'
        assert validation_method in ['bonferroni', 'fdr', 'global'], 'validation_method must be a valid string'
        pvals_array = self.pvals_mat.flatten()
        val_threshold = self._pvals_validator(pvals_array, alpha=significance, validation_method=validation_method)
        return (self.pvals_mat < val_threshold).astype(np.ubyte)

    def _pvals_validator(self, pval_list, alpha=0.05, validation_method='fdr'):
        sorted_pvals = np.sort(pval_list)
        if self.rows_projection:
            multiplier = 6 * alpha / (self.n_rows * (self.n_rows - 1) * (self.n_rows - 2))
        else:
            multiplier = 6 * alpha / (self.n_cols * (self.n_cols - 1) * (self.n_cols - 2))
        eff_fdr_th = alpha
        if validation_method == 'bonferroni':
            eff_fdr_th = multiplier
            if sorted_pvals[0] > eff_fdr_th:
                print('No V-motifs will be validated. Try increasing alpha')
        elif validation_method == 'fdr':
            try:
                eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * alpha * multiplier))[0][-1]
            except IndexError:
                print('No V-motifs will be validated. Try increasing alpha')
                eff_fdr_pos = 0
            eff_fdr_th = (eff_fdr_pos + 1) * multiplier  # +1 because of Python numbering: our pvals are ordered 1,...,n
        return eff_fdr_th

    def _projection_from_pvals(self, alpha=0.05, validation_method='fdr'):
        """Internal method to build the projected network from pvalues.
        :param float alpha:  Threshold for the validation.
        :param str validation_method:  The type of validation to apply.
        """
        pval_list = []
        if self.rows_projection:
            pvals_adj_list = self.rows_pvals
        else:
            pvals_adj_list = self.cols_pvals

        for node in pvals_adj_list:
            for neighbor1 in pvals_adj_list[node].keys():
                for neighbor2 in pvals_adj_list[node][neighbor1].keys():
                    pval_list.append(pvals_adj_list[node][neighbor1][neighbor2])
        eff_fdr_th = self._pvals_validator(pval_list, alpha=alpha, validation_method=validation_method)
        projected_adj_list = dict()
        for node in self.v_adj_list:
            for neighbor1 in self.v_adj_list[node].keys():
                for neighbor2 in self.v_adj_list[node][neighbor1].keys():
                    if pvals_adj_list[node][neighbor1][neighbor2] <= eff_fdr_th:
                        if node not in projected_adj_list.keys():
                            projected_adj_list[node] = dict()
                        if neighbor1 not in projected_adj_list[node].keys():
                            projected_adj_list[node][neighbor1] = []
                        projected_adj_list[node][neighbor1].append(neighbor2)
        return projected_adj_list

    def get_rows_projection(self,
                            alpha=0.05,
                            method='poisson',
                            threads_num=None,
                            progress_bar=True,
                            fmt='adjacency_list',
                            validation_method='fdr'):
        """Get the projected network on the rows layer of the graph.
        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :param str fmt: the desired format for the output: adjacency_list (default) or edgelist
        :returns: the projected network on the rows layer, in the format specified by fmt
        :param str validation_method:  The type of validation to apply: 'global' for a global threshold,
         'fdr' for False Discovery Rate or 'bonferroni' for Bonferroni correction.
        """
        if not self.is_rows_projected:
            self.compute_projection(rows=True, alpha=alpha, approx_method=method, threads_num=threads_num,
                                    progress_bar=progress_bar, validation_method=validation_method)

        if self.rows_dict is None:
            adj_list_to_return = self.projected_rows_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_rows_adj_list:
                adj_list_to_return[self.rows_dict[node]] = {}
                for neighbor1 in self.projected_rows_adj_list[node]:
                    adj_list_to_return[self.rows_dict[node]][self.rows_dict[neighbor1]] = []
                    for neighbor2 in self.projected_rows_adj_list[node][neighbor1]:
                        adj_list_to_return[self.rows_dict[node]][self.rows_dict[neighbor1]].append(self.rows_dict[neighbor2])
        if fmt == 'adjacency_list':
            return adj_list_to_return
        elif fmt == 'edgelist':
            return edgelist_from_adjacency_list_bipartite(adj_list_to_return)

    def get_cols_projection(self,
                            alpha=0.05,
                            method='poisson',
                            threads_num=None,
                            progress_bar=True,
                            fmt='adjacency_list'):
        """Get the projected network on the columns layer of the graph.
        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :param str fmt: the desired format for the output: adjacency_list (default) or edgelist
        :returns: the projected network on the columns layer, in the format specified by fmt
        """
        if not self.is_cols_projected:
            self.compute_projection(rows=False,
                                    alpha=alpha, approx_method=method, threads_num=threads_num, progress_bar=progress_bar)
        if self.cols_dict is None:
            return self.projected_cols_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_cols_adj_list:
                adj_list_to_return[self.cols_dict[node]] = []
                for neighbor in self.projected_cols_adj_list[node]:
                    adj_list_to_return[self.cols_dict[node]].append(self.cols_dict[neighbor])
        if fmt == 'adjacency_list':
            return adj_list_to_return
        elif fmt == 'edgelist':
            return edgelist_from_adjacency_list_bipartite(adj_list_to_return)

    def _compute_projected_pvals_mat(self, layer='rows'):
        """
        Compute the pvalues matrix representing the significance of the original matrix.
        """
        if layer == 'rows':
            n_dim = self.n_rows
            pval_adjlist = self.rows_pvals
        elif layer == 'columns':
            n_dim = self.n_cols
            pval_adjlist = self.cols_pvals
        pval_mat = np.ones((n_dim, n_dim))
        for v in pval_adjlist:
            for w in pval_adjlist[v]:
                pval_mat[v,w] = pval_adjlist[v][w]
                pval_mat[w,v] = pval_mat[v,w]
        if layer == 'rows':
            self.rows_pvals_mat = pval_mat
        elif layer == 'columns':
            self.cols_pvals_mat = pval_mat

    def get_projected_pvals_mat(self, layer=None):
        """
        Return the pvalues matrix of the projection if it has been computed.

        :param layer: the layer to return.
        :type layer: string, optional
        """
        if layer is None:
            if not self.is_rows_projected and not self.is_cols_projected:
                print('First compute a projection.')
                return None
            elif self.is_rows_projected and self.is_cols_projected:
                print("Please specify the layer with layer='rows' or layer='columns'.")
                return None
            elif self.is_rows_projected:
                layer = 'rows'
            elif self.is_cols_projected:
                layer = 'columns'
        if layer == 'rows':
            if self.rows_pvals_mat is None:
                self._compute_projected_pvals_mat(layer=layer)
            return self.rows_pvals_mat
        elif layer == 'columns':
            if self.cols_pvals_mat is None:
                self._compute_projected_pvals_mat(layer=layer)
            return self.cols_pvals_mat
        else:
            raise ValueError("layer must be either 'rows' or 'columns' or None")

    def set_biadjacency_matrix(self, biadjacency):
        """Set the biadjacency matrix of the graph.
        :param biadjacency: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type biadjacency: numpy.array, scipy.sparse, list
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(biadjacency=biadjacency)

    def set_adjacency_list(self, adj_list):
        """Set the adjacency list of the graph.
        :param adj_list: a dictionary containing the adjacency list
                of a bipartite graph with the nodes of one layer as keys
                and lists of neighbor nodes of the other layer as values.
        :type adj_list: dict
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(adjacency_list=adj_list)

    def set_edgelist(self, edgelist):
        """Set the edgelist of the graph.
        :param edgelist: list of edges containing couples (row_node, col_node) of
            nodes forming an edge. each element in the couples must belong to
            the respective layer.
        :type edgelist: list, numpy.array
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequences):
        """Set the degree sequence of the graph.
        :param degree_sequences: couple of lists describing the degree sequences
            of both layers.
        :type degree_sequences: list, numpy.array, tuple
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(degree_sequences=degree_sequences)

    def set_to_continuous(self):
        self.continuous_weights = True

    def clean_edges(self):
        """Clean the edges of the graph.
        """
        self.biadjacency = None
        self.edgelist = None
        self.adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.is_initialized = False

    def model_loglikelihood(self):
        """Returns the loglikelihood of the solution of last model executed.
        """
        return self.loglikelihood
