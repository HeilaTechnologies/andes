"""
The Andes plotting tool.
"""

import logging
import os
import re
from distutils.spawn import find_executable

from andes.utils.misc import is_notebook
from andes.core.var import BaseVar, Algeb, ExtAlgeb
from andes.utils.paths import get_dot_andes_path
from andes.shared import np, mpl, plt

logger = logging.getLogger(__name__)


class TDSData(object):
    """
    A data container for loading and plotting results from Andes time-domain simulation.
    """

    def __init__(self, full_name=None, mode='file', dae=None, path=None):
        # paths and file names
        self._mode = mode
        self.full_name = full_name
        self.dae = dae
        self._path = path if path else os.getcwd()
        self.file_name = None
        self.file_ext = None
        self._npy_file = None
        self._lst_file = None

        # data members for raw data
        self._idx = []  # indices of variables
        self._uname = []  # unformatted variable names
        self._fname = []  # formatted variable names
        self._data = []  # data loaded from file

        # auxillary data members for fast query
        self.t = []
        self.nvars = 0  # total number of variables including `t`

        if self._mode == 'file':
            self._process_names()
            self.load_lst()
            self.load_npy_or_csv()
        elif self._mode == 'memory':
            self.load_dae()
            self._process_names()
        else:
            raise NotImplementedError(f'Unknown mode {self._mode}.')

        self._process_names()

    def _process_names(self):
        self.file_name, self.file_ext = os.path.splitext(self.full_name)
        self._npy_file = os.path.join(self._path, self.file_name + '.npy')

        npz_path = os.path.join(self._path, self.file_name + '.npz')
        if os.path.isfile(npz_path):
            self._npy_file = npz_path

        self._lst_file = os.path.join(self._path, self.file_name + '.lst')
        self._csv_file = os.path.join(self._path, self.file_name + '.csv')

    def load_dae(self):
        """Load from DAE time series"""
        dae = self.dae
        self.t = dae.ts.t
        self.nvars = dae.n + dae.m + dae.o + 1

        self._idx = list(range(self.nvars))
        self._uname = ['Time [s]'] + dae.x_name + dae.y_name + dae.z_name
        self._fname = ['Time [s]'] + dae.x_tex_name + dae.y_tex_name + dae.z_tex_name
        self._data = dae.ts.txyz

        if dae.system.files.lst is not None:
            self.full_name = dae.system.files.lst
        else:
            self.full_name = dae.system.files.case

    def load_lst(self):
        """
        Load the lst file into internal data structures `_idx`, `_fname`, `_uname`, and counts the number of
        variables to `nvars`.

        Returns
        -------
        None
        """
        with open(self._lst_file, 'r') as fd:
            lines = fd.readlines()

        idx, uname, fname = list(), list(), list()

        for line in lines:
            values = line.split(',')
            values = [x.strip() for x in values]

            # preserve the idx ordering here in case variables are not
            # ordered by idx
            idx.append(int(values[0]))  # convert to integer
            uname.append(values[1])
            fname.append(values[2])

        self._idx = idx
        self._fname = fname
        self._uname = uname
        self.nvars = len(uname)

    def find(self, query, exclude=None, formatted=False, idx_only=False):
        """
        Return variable names and indices matching `query`.

        Parameters
        ----------
        query : str
            The string for querying variables. Multiple conditions can be separated by comma without space.
        exclude  : str, optional
            A string pattern to be excluded
        formatted : bool, optional
            True to return formatted names, False otherwise
        idx_only : bool, optional
            True if only return indices

        Returns
        -------
        (list, list)
            (List of found indices, list of found names)
        """

        # load the variable list to search in
        names = self._uname if formatted is False else self._fname

        found_idx, found_names = list(), list()

        query_list = query.split(',')
        for idx, name in zip(self._idx, names):
            for q in query_list:
                if re.search(q, name):
                    if exclude and re.search(exclude, name):
                        continue

                    found_idx.append(idx)
                    found_names.append(name)

        if idx_only:
            return found_idx
        else:
            return found_idx, found_names

    def load_npy_or_csv(self, delimiter=','):
        """
        Load the npy, zpy or (the legacy) csv file into the
        internal data structure `self._xy`.

        Parameters
        ----------
        delimiter : str, optional
            The delimiter for the case file. Default to comma.

        Returns
        -------
        None
        """
        try:
            data = np.load(self._npy_file)

            if self._npy_file.endswith('npz'):
                data = data['data']

        except FileNotFoundError:
            data = np.loadtxt(self._csv_file, delimiter=delimiter, skiprows=1)

        self._data = data

    def get_values(self, idx):
        """
        Return the variable values at the given indices.

        Parameters
        ----------
        idx : list
            The indicex of the variables to retrieve. `idx=0` is for Time. Variable indices start at 1.

        Returns
        -------
        np.ndarray
            Variable data
        """
        return self._data[:, idx]

    def get_header(self, idx, formatted=False):
        """
        Return a list of the variable names at the given indices.

        Parameters
        ----------
        idx : list or int
            The indices of the variables to retrieve
        formatted : bool
            True to retrieve latex-formatted names, False for unformatted names

        Returns
        -------
        list
            A list of variable names (headers)

        """

        if isinstance(idx, int):
            idx = [idx]
        header = self._uname if not formatted else self._fname
        return [header[x] for x in idx]

    def export_csv(self, path=None, idx=None, header=None, formatted=False,
                   sort_idx=True, fmt='%.18e'):
        """
        Export to a csv file.

        Parameters
        ----------
        path : str
            path of the csv file to save
        idx : None or array-like, optional
            the indices of the variables to export. Export all by default
        header : None or array-like, optional
            customized header if not `None`. Use the names from the lst file
            by default
        formatted : bool, optional
            Use LaTeX-formatted header. Does not apply when using customized
            header
        sort_idx : bool, optional
            Sort by idx or not, # TODO: implement sort
        fmt : str
            cell formatter
        """
        if not path:
            path = self._csv_file
        if not idx:
            idx = self._idx
        if not header:
            header = self.get_header(idx, formatted=formatted)

        if len(idx) != len(header):
            raise ValueError("Idx length does not match header length")

        body = self.get_values(idx)

        with open(path, 'w') as fd:
            fd.write(','.join(header) + '\n')
            np.savetxt(fd, body, fmt=fmt, delimiter=',')

        logger.info(f'CSV data saved to "{path}".')

    def plot(self, yidx, xidx=(0,), a=None, ycalc=None,
             left=None, right=None, ymin=None, ymax=None, ytimes=None,
             xlabel=None, ylabel=None,
             legend=None, grid=False, greyscale=False,
             latex=True, dpi=150, line_width=1.0, font_size=12, savefig=None, save_format=None, show=True,
             title=None, use_bqplot=False,
             hline1=None, hline2=None, vline1=None, vline2=None, **kwargs):
        """
        Entery function for plot scripting. This function retrieves the x and y values based
        on the `xidx` and `yidx` inputs and then calls `plot_data()` to do the actual plotting.

        Note that `ytimes` and `ycalc` are applied sequentially if apply.

        Refer to `plot_data()` for the definition of arguments.

        Parameters
        ----------
        xidx : list or int
            The index for the x-axis variable

        yidx : list or int
            The indices for the y-axis variables

        Returns
        -------
        (fig, ax)
            Figure and axis handles
        """
        if self._mode == 'memory':
            if isinstance(yidx, BaseVar):
                if yidx.n == 0:
                    logger.error(f"Variable <{yidx.name}> contains no values.")
                    return
                offs = 1
                if isinstance(yidx, (Algeb, ExtAlgeb)):
                    offs += self.dae.n

                yidx = yidx.a + offs

        if a is not None:
            yidx = np.take(yidx, a)

        x_value = self.get_values(xidx)
        y_value = self.get_values(yidx)

        # header: names for variables
        # axis labels: the texts next to axes
        x_header = self.get_header(xidx, formatted=latex)
        y_header = self.get_header(yidx, formatted=latex)

        ytimes = float(ytimes) if ytimes is not None else ytimes
        if ytimes and (ytimes != 1):
            y_scale_func = scale_func(ytimes)
        else:
            y_scale_func = None

        # apply `ytimes` first
        if y_scale_func:
            y_value = y_scale_func(y_value)

        # `ycalc` is a callback function for manipulating data
        if ycalc is not None:
            y_value = ycalc(y_value)

        if use_bqplot is True or (use_bqplot is None and is_notebook()):

            return self.bqplot_data(xdata=x_value, ydata=y_value, xheader=x_header, yheader=y_header,
                                    left=left, right=right, ymin=ymin, ymax=ymax,
                                    xlabel=xlabel, ylabel=ylabel, legend=legend, grid=grid, greyscale=greyscale,
                                    latex=latex, dpi=dpi, line_width=line_width, font_size=font_size,
                                    savefig=savefig, save_format=save_format, show=show, title=title,
                                    **kwargs)

        else:
            return self.plot_data(xdata=x_value, ydata=y_value, xheader=x_header, yheader=y_header,
                                  left=left, right=right, ymin=ymin, ymax=ymax,
                                  xlabel=xlabel, ylabel=ylabel, legend=legend, grid=grid, greyscale=greyscale,
                                  latex=latex, dpi=dpi, line_width=line_width, font_size=font_size,
                                  savefig=savefig, save_format=save_format, show=show, title=title,
                                  hline1=hline1, hline2=hline2, vline1=vline1, vline2=vline2,
                                  **kwargs)

    def data_to_df(self):
        """Convert to pandas.DataFrame"""
        pass

    def guess_event_time(self):
        """Guess the event starting time from the input data by checking
        when the values start to change
        """
        pass

    def bqplot_data(self, xdata, ydata, xheader=None, yheader=None, xlabel=None, ylabel=None,
                    left=None, right=None, ymin=None, ymax=None, legend=True, grid=False, fig=None,
                    latex=True, dpi=150, line_width=1.0, greyscale=False, savefig=None, save_format=None,
                    show=True, **kwargs):
        """
        Plot with ``bqplot``. Experimental and imcomplete.
        """

        from bqplot import pyplot as plt
        if not isinstance(ydata, np.ndarray):
            TypeError("ydata must be numpy array. Retrieve with get_values().")

        if ydata.ndim == 1:
            ydata = ydata.reshape((-1, 1))

        plt.figure(dpi=dpi)
        plt.plot(xdata, ydata.transpose(),
                 linewidth=line_width,
                 )

        if yheader:
            plt.label(yheader)

        plt.show()

    def plot_data(self, xdata, ydata, xheader=None, yheader=None, xlabel=None, ylabel=None, line_styles=None,
                  left=None, right=None, ymin=None, ymax=None, legend=None, grid=False, fig=None, ax=None,
                  latex=True, dpi=150, line_width=1.0, font_size=12, greyscale=False, savefig=None,
                  save_format=None, show=True, title=None, hline1=None, hline2=None, vline1=None,
                  vline2=None, **kwargs):
        """
        Plot lines for the supplied data and options. This functions takes `xdata` and `ydata` values. If
        you provide variable indices instead of values, use `plot()`.

        Parameters
        ----------
        xdata : array-like
            An array-like object containing the values for the x-axis variable

        ydata : array
            An array containing the values of each variables for the y-axis variable. The row
            of `ydata` must match the row of `xdata`. Each column correspondings to a variable.

        xheader : list
            A list containing the variable names for the x-axis variable

        yheader : list
            A list containing the variable names for the y-axis variable

        xlabel : str
            Text label for the x axis

        ylabel : str
            Text label for the y axis

        left : float
            The starting value of the x axis

        right : float
            The ending value of the x axis

        ymin : float
            The minimum value of the y axis
        ymax : float
            The maximum value of the y axis

        legend : bool
            True to show legend and False otherwise
        grid : bool
            True to show grid and False otherwise
        fig
            Matplotlib fig object to draw the axis on
        ax
            Matplotlib axis object to draw the lines on
        latex : bool
            True to enable latex and False to disable
        greyscale : bool
            True to use greyscale, False otherwise
        savefig : bool
            True to save to png figure file
        save_format : str
            File extension string (pdf, png or jpg) for the savefig format
        dpi : int
            Dots per inch for screen print or save. savefig uses a minimum of 200 dpi
        line_width : float
            Plot line width
        font_size : float
            Text font size (labels and legends)
        show : bool
            True to show the image
        title : str
            Title string to be shown at the top
        hline1: float, optional
            Dashed horizontal line 1
        hline2: float, optional
            Dashed horizontal line 2
        vline1: float, optional
            Dashed horizontal line 1
        vline2: float, optional
            Dashed vertical line 2
        kwargs
            Optional kwargs

        Returns
        -------
        (fig, ax)
            The figure and axis handles
        """
        mpl.rc('font', family='Arial', size=font_size)

        if not isinstance(ydata, np.ndarray):
            TypeError("ydata must be a numpy array. Retrieve with get_values().")

        if ydata.ndim == 1:
            ydata = ydata.reshape((-1, 1))

        n_lines = ydata.shape[1]

        set_latex(latex)

        # set default x min based on simulation time
        if not left:
            left = xdata[0] - 1e-6
        if not right:
            right = xdata[-1] + 1e-6

        if not line_styles:
            line_styles = ['-', '--', '-.', ':']

        line_styles = line_styles * int(n_lines / len(line_styles) + 1)

        hold = True
        if not (fig and ax):
            fig = plt.figure(dpi=dpi)
            ax = plt.gca()
            hold = False

        if greyscale:
            plt.gray()

        for i in range(n_lines):
            ax.plot(xdata,
                    ydata[:, i],
                    ls=line_styles[i],
                    label=yheader[i] if yheader else None,
                    linewidth=line_width,
                    color='0.2' if greyscale else None,
                    )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(xheader[0])

        if ylabel:
            ax.set_ylabel(ylabel)

        ax.ticklabel_format(useOffset=False)

        if not hold:
            ax.set_xlim(left=left, right=right)
            ax.set_ylim(bottom=ymin, top=ymax)
        else:
            ax.autoscale(axis='y')

        if grid:
            ax.grid(b=True, linestyle='--')

        if yheader is None:
            legend = False
        elif legend is None:
            if len(yheader) <= 8:
                legend = True

        if legend:
            ax.legend()

        if title:
            ax.set_title(title)

        if hline1:
            ax.axhline(y=hline1, linewidth=1, ls=':', color='grey')
        if hline2:
            ax.axhline(y=hline2, linewidth=1, ls=':', color='grey')
        if vline1:
            ax.axvline(x=vline1, linewidth=1, ls=':', color='grey')
        if vline2:
            ax.axvline(x=vline2, linewidth=1, ls=':', color='grey')

        plt.draw()

        if savefig:
            if save_format is None:
                save_format = 'png'

            if dpi is None:
                dpi = 200
            else:
                dpi = max(dpi, 200)

            count = 1
            while True:
                outfile = f'{self.file_name}_{count}.{save_format}'
                if not os.path.isfile(outfile):
                    break
                count += 1

            fig.savefig(outfile, dpi=dpi)
            logger.info(f'Figure saved to "{outfile}".')

        if show:
            plt.show()

        return fig, ax


def parse_y(y, upper, lower=0):
    """
    Parse command-line input for Y indices and return a list of indices

    Parameters
    ----------
    y : Union[List, Set, Tuple]
        Input for Y indices. Could be single item (with or without colon), or
         multiple items

    upper : int
        Upper limit. In the return list y, y[i] <= uppwer.

    lower : int
        Lower limit. In the return list y, y[i] >= lower.

    Returns
    -------

    """
    if len(y) == 1:
        if y[0].count(':') >= 3:
            logger.error('Index format not acceptable. Must not contain more than three colons.')
            return []

        elif y[0].count(':') == 0:
            if isint(y[0]):
                y[0] = int(y[0])
                return y
        elif y[0].count(':') == 1:
            if y[0].endswith(':'):
                y[0] += str(upper)
            if y[0].startswith(':'):
                y[0] = str(lower) + y[0]

        elif y[0].count(':') == 2:
            if y[0].endswith(':'):
                y[0] += str(1)
            if y[0].startswith(':'):
                y[0] = str(lower) + y[0]

            if y[0].count('::') == 1:
                y[0] = y[0].replace('::', ':{}:'.format(upper))
                print(y)

        y = y[0].split(':')

        for idx, item in enumerate(y[:]):
            try:
                y[idx] = int(item)
            except ValueError:
                logger.warning(f'y contains non-numerical values <{item}>. Parsing cannot proceed.')
                return []

        y_from_range = list(range(*y))

        y_in_range = []

        for item in y_from_range:
            if lower <= item < upper:
                y_in_range.append(item)

        return y_in_range

    else:
        y_to_int = []
        for idx, val in enumerate(y):
            try:
                y_to_int.append(int(val))
            except ValueError:
                logger.warning('Y indices contains non-numerical values. Skipped <{}>.'.format(val))

        y_in_range = [item for item in y_to_int if lower <= item < upper]
        return list(y_in_range)


def add_plot(x, y, xl, yl, fig, ax, LATEX=False, linestyle=None, **kwargs):
    """Add plots to an existing plot"""
    if LATEX:
        # xl_data = xl[1]  # NOQA
        yl_data = yl[1]
    else:
        # xl_data = xl[0]  # NOQA
        yl_data = yl[0]

    for idx, y_val in enumerate(y):
        ax.plot(x, y_val, label=yl_data[idx], linestyle=linestyle)

    ax.legend(loc='upper right')
    ax.set_ylim(auto=True)


def eig_plot(name, args):
    fullpath = os.path.join(name, '.txt')
    raw_data = []
    started = 0
    fid = open(fullpath)
    for line in fid.readline():
        if '#1' in line:
            started = 1
        elif 'PARTICIPATION FACTORS' in line:
            started = -1

        if started == 1:
            raw_data.append(line)
        elif started == -1:
            break
    fid.close()

    for line in raw_data:
        # data = line.split()
        # TODO: complete this function
        pass


def tdsplot(filename, y, x=(0,),
            tocsv=False,
            find=None,
            xargs=None,
            exclude=None,
            **kwargs):
    """
    TDS plot main function based on the new TDSData class

    Parameters
    ----------
    filename : str
        Path to the ANDES TDS output data file. Works without extension.
    x : list or int, optional
        The index for the x-axis variable. x=0 by default for time
    y : list or int
        The indices for the y-axis variable
    tocsv : bool
        True if need to export to a csv file
    find : str, optional
        if not none, specify the variable name to find
    xargs : str, optional
        similar to find, but return the result indices with file name, x idx name for xargs
    exclude : str, optional
        variable name pattern to exclude

    Returns
    -------
    TDSData object
    """

    # single data file
    if len(filename) == 1:
        tds_data = TDSData(filename[0])
        if tocsv is True:
            tds_data.export_csv()
            return
        if find is not None:
            out = tds_data.find(query=find, exclude=exclude)
            print(out)
            return
        if xargs is not None:
            out = tds_data.find(query=xargs, exclude=exclude, idx_only=True)
            out = [str(i) for i in out]
            print(filename[0] + ' 0 ' + ' '.join(out))
            return
        if len(y) == 0:
            logger.error('Must specify Y indices to plot.')
            return
        y_num = parse_y(y, lower=0, upper=tds_data.nvars)
        tds_data.plot(xidx=x, yidx=y_num, **kwargs)
        return tds_data
    else:
        raise NotImplementedError("Plotting multiple data files are not supported yet")


def check_init(yval, yl):
    """"
    Check initialization by comparing t=0 and t=end values for a flat run.

    Warnings
    --------
    This function is deprecated as the initialization check feature is built into TDS.
    See ``TDS.test_initialization()``.
    """
    suspect = []
    for var, label in zip(yval, yl):
        if abs(var[0] - var[-1]) >= 1e-6:
            suspect.append(label)
    if suspect:
        logger.error('Initialization failed:')
        logger.error(', '.join(suspect))
    else:
        logger.error('Initialization is correct.')


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def scale_func(k):
    """
    Return a lambda function that scales its input by k

    Parameters
    ----------
    k : float
        The scaling factor of the returned lambda function
    Returns
    -------
    Lambda function

    """
    return lambda y_values_input: k * y_values_input


def label_latexify(label):
    """
    Convert a label to latex format by appending surrounding $ and escaping spaces

    Parameters
    ----------
    label : str
        The label string to be converted to latex expression

    Returns
    -------
    str
        A string with $ surrounding
    """
    return '$' + label.replace(' ', r'\ ') + '$'


def set_latex(enable=True):
    """
    Enables LaTeX for matplotlib based on the `with_latex` option and `dvipng` availability.

    Parameters
    ----------
    enable : bool, optional
        True for latex on and False for off

    Returns
    -------
    bool
        True for LaTeX on, False for off
    """

    has_dvipng = find_executable('dvipng')

    if has_dvipng and enable:
        mpl.rc('text', usetex=True)

        no_warn_file = os.path.join(get_dot_andes_path(), '.no_warn_latex')
        if not os.path.isfile(no_warn_file):
            logger.info('Info:')
            logger.info('Using LaTeX for rendering. If an error occurs:')
            logger.info('a) If you are using `andes plot`, disable with option "-d",')
            logger.info('b) If you are using `plot()`, set "latex=False".')

            try:
                with open(os.path.join(get_dot_andes_path(), '.no_warn_latex'), 'w'):
                    pass
            except OSError:
                pass

        return True

    else:
        mpl.rc('text', usetex=False)
        return False
