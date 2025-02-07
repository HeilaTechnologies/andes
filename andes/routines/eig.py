import logging
import scipy.io

from math import ceil, pi
from cvxopt import mul, div, spdiag
from cvxopt.lapack import gesv

from andes.io.txt import dump_data
from andes.utils.misc import elapsed
from andes.routines.base import BaseRoutine
from andes.shared import np, matrix, spmatrix, plt, mpl
from andes.plot import set_latex

logger = logging.getLogger(__name__)
__cli__ = 'eig'


class EIG(BaseRoutine):
    """
    Eigenvalue analysis routine
    """
    def __init__(self, system, config):
        super().__init__(system=system, config=config)

        self.config.add(plot=0)
        self.config.add_extra("_help", plot="show plot after computation")
        self.config.add_extra("_alt", plot=(0, 1))

        # internal flags and storage
        self.As = None
        self.eigs = None
        self.mu = None
        self.part_fact = None
        self.singular_idx = np.array([], dtype=int)
        self.x_name = []

    def calc_state_matrix(self):
        r"""
        Return state matrix and store to ``self.As``.

        Notes
        -----
        For systems with the form

        .. math ::

            T \dot{x} = f(x, y) \\
            0 = g(x, y)

        The state matrix is calculated from

        .. math ::

            A_s = T^{-1} (f_x - f_y * g_y^{-1} * g_x)

        Returns
        -------
        cvxopt.matrix
            state matrix
        """
        system = self.system

        gyx = matrix(system.dae.gx)
        self.solver.linsolve(system.dae.gy, gyx)

        Tfnz = system.dae.Tf + np.ones_like(system.dae.Tf) * np.equal(system.dae.Tf, 0.0)
        iTf = spdiag((1 / Tfnz).tolist())
        self.As = matrix(iTf * (system.dae.fx - system.dae.fy * gyx))
        return self.As

    def remove_singular_rc(self):
        """
        Remove rows and cols associated with zero time constant.
        """
        self.As = np.delete(self.As, self.singular_idx, axis=0)
        self.As = np.delete(self.As, self.singular_idx, axis=1)

    def calc_eigvals(self):
        """
        Solve eigenvalues of the state matrix ``self.As``

        Returns
        -------
        None
        """
        self.eigs = np.linalg.eigvals(self.As)
        return self.eigs

    def calc_part_factor(self, As=None):
        """
        Compute participation factor of states in eigenvalues

        Returns
        -------

        """
        if As is None:
            As = self.As
        mu, N = np.linalg.eig(As)

        N = matrix(N)
        n = len(mu)
        idx = range(n)

        mu_complex = np.array([0] * n, dtype=complex)
        W = matrix(spmatrix(1.0, idx, idx, As.shape, N.typecode))
        gesv(N, W)

        partfact = mul(abs(W.T), abs(N))

        b = matrix(1.0, (1, n))
        WN = b * partfact
        partfact = partfact.T

        for item in idx:
            mu_real = float(mu[item].real)
            mu_imag = float(mu[item].imag)
            mu_complex[item] = complex(round(mu_real, 5), round(mu_imag, 5))
            partfact[item, :] /= WN[item]

        # participation factor
        self.mu = matrix(mu_complex)
        self.part_fact = matrix(partfact)

        return self.mu, self.part_fact

    def summary(self):
        """
        Print out a summary to ``logger.info``.
        """
        out = list()
        out.append('')
        out.append('-> Eigenvalue Analysis:')
        out_str = '\n'.join(out)
        logger.info(out_str)

    def run(self, **kwargs):
        succeed = False
        system = self.system
        self.singular_idx = np.array([], dtype=int)

        if system.PFlow.converged is False:
            logger.warning('Power flow not solved. Eig analysis will not continue.')
            return succeed
        else:
            if system.TDS.initialized is False:
                system.TDS.init()
                system.TDS._itm_step()

        if system.dae.n == 0:
            logger.error('No dynamic model. Eig analysis will not continue.')

        else:
            if sum(system.dae.Tf != 0) != len(system.dae.Tf):
                self.singular_idx = np.argwhere(np.equal(system.dae.Tf, 0.0)).ravel().astype(int)
                logger.info(f"System contains {len(self.singular_idx)} zero time constants. ")
                logger.debug([system.dae.x_name[i] for i in self.singular_idx])

            self.x_name = np.array(system.dae.x_name)
            self.x_name = np.delete(self.x_name, self.singular_idx)

            self.summary()
            t1, s = elapsed()

            self.calc_state_matrix()
            self.remove_singular_rc()
            self.calc_part_factor()

            if not self.system.files.no_output:
                self.report()
                if system.options.get('state_matrix') is True:
                    self.export_state_matrix()

            if self.config.plot:
                self.plot()
            _, s = elapsed(t1)
            logger.info('Eigenvalue analysis finished in {:s}.'.format(s))

            succeed = True

        system.exit_code = 0 if succeed else 1
        return succeed

    def plot(self, mu=None, fig=None, ax=None, left=-6, right=0.5, ymin=-8, ymax=8, damping=0.05,
             line_width=0.5, dpi=150, show=True, latex=True):
        mpl.rc('font', family='Times New Roman', size=12)

        if mu is None:
            mu = self.mu
        mu_real = mu.real()
        mu_imag = mu.imag()
        p_mu_real, p_mu_imag = list(), list()
        z_mu_real, z_mu_imag = list(), list()
        n_mu_real, n_mu_imag = list(), list()

        for re, im in zip(mu_real, mu_imag):
            if re == 0:
                z_mu_real.append(re)
                z_mu_imag.append(im)
            elif re > 0:
                p_mu_real.append(re)
                p_mu_imag.append(im)
            elif re < 0:
                n_mu_real.append(re)
                n_mu_imag.append(im)

        if len(p_mu_real) > 0:
            logger.warning(
                'System is not stable due to {} positive eigenvalues.'.format(
                    len(p_mu_real)))
        else:
            logger.info(
                'System is small-signal stable in the initial neighbourhood.')

        set_latex(latex)

        if fig is None or ax is None:
            fig, ax = plt.subplots(dpi=dpi)

        ax.scatter(n_mu_real, n_mu_imag, marker='x', s=40, linewidth=0.5, color='black')
        ax.scatter(z_mu_real, z_mu_imag, marker='o', s=40, linewidth=0.5, facecolors='none', edgecolors='black')
        ax.scatter(p_mu_real, p_mu_imag, marker='x', s=40, linewidth=0.5, color='black')
        ax.axhline(linewidth=0.5, color='grey', linestyle='--')
        ax.axvline(linewidth=0.5, color='grey', linestyle='--')

        # plot 5% damping lines
        xin = np.arange(left, 0, 0.01)
        yneg = xin / damping
        ypos = - xin / damping

        ax.plot(xin, yneg, color='grey', linewidth=line_width, linestyle='--')
        ax.plot(xin, ypos, color='grey', linewidth=line_width, linestyle='--')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_xlim(left=left, right=right)
        ax.set_ylim(ymin, ymax)

        if show is True:
            plt.show()
        return fig, ax

    def export_state_matrix(self):
        """
        Export state matrix to a ``<CaseName>_As.mat`` file with the variable name ``As``, where
        ``<CaseName>`` is the test case name.

        State variable names are stored in variables ``x_name`` and ``x_tex_name``.

        Returns
        -------
        bool
            True if successful
        """
        system = self.system
        out = {'As': self.As,
               'x_name': np.array(system.dae.x_name, dtype=np.object),
               'x_tex_name': np.array(system.dae.x_tex_name, dtype=np.object),
               }
        scipy.io.savemat(system.files.mat, mdict=out)
        logger.info(f'State matrix saved to "{system.files.mat}".')
        return True

    def report(self, x_name=None):
        """
        Save eigenvalue analysis reports

        Returns
        -------
        None
        """
        from andes.variables.report import report_info

        system = self.system
        mu = self.mu
        part_fact = self.part_fact
        if x_name is None:
            x_name = self.x_name

        text = []
        header = []
        rowname = []
        data = []

        neig = len(mu)
        mu_real = mu.real()
        mu_imag = mu.imag()
        n_positive = sum(1 for x in mu_real if x > 0)
        n_zeros = sum(1 for x in mu_real if x == 0)
        n_negative = sum(1 for x in mu_real if x < 0)

        numeral = []
        for idx, item in enumerate(range(neig)):
            if mu_real[idx] == 0:
                marker = '*'
            elif mu_real[idx] > 0:
                marker = '**'
            else:
                marker = ''
            numeral.append('#' + str(idx + 1) + marker)

        # compute frequency, un-damped frequency and damping
        freq = [0] * neig
        ufreq = [0] * neig
        damping = [0] * neig
        for idx, item in enumerate(mu):
            if item.imag == 0:
                freq[idx] = 0
                ufreq[idx] = 0
                damping[idx] = 0
            else:
                ufreq[idx] = abs(item) / 2 / pi
                freq[idx] = abs(item.imag / 2 / pi)
                damping[idx] = -div(item.real, abs(item)) * 100

        # obtain most associated variables
        var_assoc = []
        for prow in range(neig):
            temp_row = part_fact[prow, :]
            name_idx = list(temp_row).index(max(temp_row))
            var_assoc.append(x_name[name_idx])

        pf = []
        for prow in range(neig):
            temp_row = []
            for pcol in range(neig):
                temp_row.append(round(part_fact[prow, pcol], 5))
            pf.append(temp_row)

        # opening info section
        text.append(report_info(self.system))
        header.append(None)
        rowname.append(None)
        data.append(None)
        text.append('')

        header.append([''])
        rowname.append(['EIGENVALUE ANALYSIS REPORT'])
        data.append('')

        text.append('STATISTICS\n')
        header.append([''])
        rowname.append(['Positives', 'Zeros', 'Negatives'])
        data.append([n_positive, n_zeros, n_negative])

        text.append('EIGENVALUE DATA\n')
        header.append([
            'Most Associated',
            'Real',
            'Imag.',
            'Damped Freq.',
            'Frequency',
            'Damping [%]'])
        rowname.append(numeral)
        data.append(
            [var_assoc,
             list(mu_real),
             list(mu_imag),
             freq,
             ufreq,
             damping])

        n_cols = 7  # columns per block
        n_block = int(ceil(neig / n_cols))

        if n_block <= 100:
            for idx in range(n_block):
                start = n_cols * idx
                end = n_cols * (idx + 1)
                text.append('PARTICIPATION FACTORS [{}/{}]\n'.format(
                    idx + 1, n_block))
                header.append(numeral[start:end])
                rowname.append(x_name)
                data.append(pf[start:end])

        dump_data(text, header, rowname, data, system.files.eig)
        logger.info(f'Report saved to "{system.files.eig}".')
