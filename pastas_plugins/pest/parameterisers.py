import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyemu
from pandas import DataFrame, DatetimeIndex, Series
from pastas import Model
from pastas.stressmodels import StressModel, WellModel
from pastas.timeseries_utils import _frequency_is_supported
from pypestutils.pestutilslib import PestUtilsLib

from pastas_plugins.pest.solver import PestSolver

pputils = PestUtilsLib()  # the constructor searches for the shared lib
logger = getLogger(__name__)

__all__ = [
    "WellModelParameteriser",
]


class BaseParameteriser(ABC):
    """
    Custom PEST parameteriser base class for StressModel stresses.
    Designed to be first instantiated outside the PestSolver for a given StressModel,
    and then passed to PestSolver via the pest_parameterisers argument.

    Parameters
    ----------
    model : pastas.Model
        Pastas model.
    stressmodel_name : str
        Name of StressModel for which to apply StressModel.stress rate parameters to via PEST.
    date_format : Optional[str]
        Datetime format for saving PEST model parameter input files for stressmodel. Default is \"%d/%m/%Y %H:%M:%S\".
    interp_kwargs : Optional[dict[str, Any]]
        kwargs to pass to BaseParameteriser.interpolate_stresses(). Default is {}.

    Attributes
    ----------
    modelfile : Path
        PestSolver model input file for scaling (optimising) stressmodel stresses during PEST runs.
    modelfile_df_org : DataFrame
        DataFrame of original PestSolver model input file for scaling (optimising) stressmodel stresses during PEST runs.
    stress : DataFrame | Series
        Stressmodel stress TimeSeries.  Series if a single stress, DataFrame if multiple (eg WellModel)
    krig_factorfile : Path
        Path to kriging factors file for use during PestSolver.run() calls.
    krig_mpts : int
        Number of kriging target points.
    stress_pars : DataFrame
        DataFrame of stressmodel.stress parameterisation info, as returned by solver.pf (pyemu.PestFrom).
    stress_parcov  : pyemu.Cov
        Prior covariance matrix for the WellModel parameters defined through this class. For use in building PEST uncertainty (.unc) files.

    Returns
    -------
    None
    """

    _name = "BaseParameteriser"

    def __getstate__(self):
        # Exclude the logger and its handlers from the state to be pickled
        state = self.__dict__.copy()
        if "logger" in state:
            del state["logger"]
        return state

    def __setstate__(self, state):
        # Reconstruct the logger after unpickling
        self.__dict__.update(state)
        logger = getLogger(__name__)  # noqa: F841

    @abstractmethod
    def __init__(
        self,
        model: Model,
        stressmodel_name: str,
        date_format: Optional[str] = "%d/%m/%Y",
        interp_kwargs: Optional[dict[str, Any]] = {},
    ) -> None:
        self.model = model
        self.stressmodel_name = stressmodel_name
        self.stressmodel = self._validate_stressmodel_name(sname=stressmodel_name)
        self.date_format = date_format
        self.interp_kwargs = interp_kwargs

        # PestSolver related things.
        # These are designed to be populated on 'BaseParameteriser.solver = solver' calls from within the PestSolver.
        self._solver = None
        self.model_ws = None
        self.modelfile = None
        self.modelfile_df_org = None

        # Stress TimeSeries
        self.stress = (
            self.stressmodel.get_stress()
        )  # not sure Series-based stressmodels can handle squeeze argument
        self.stress_names = self.stress.columns.to_list()
        if isinstance(self.stressmodel, WellModel):
            self.stress = self.stressmodel.get_stress(squeeze=False)

        # Parameterisation things
        self.source_points = None
        self.target_points = None
        self.krig_factorfile = None
        self.krig_mpts = None
        self.stress_pars = None
        self.stress_parcov = None
        self._parnme_indexer = None

    def _validate_stressmodel_name(
        self,
        sname: str,
    ) -> StressModel:
        """
        Return StressModel of given name. If stressmodel.name is not in model.stressmodels
        or it is not a WellModel, an error is raised end execution ceases.
        """
        try:
            smodel = self.model.stressmodels.get(sname)
        except KeyError:
            logger.exception(
                f"StressModel.name {sname} not found in Model {self.model.name}."
            )
            sys.exit()
        if isinstance(self, WellModelParameteriser) and not isinstance(
            smodel, WellModel
        ):
            logger.error(
                f"StressModel.name {sname} does not refer to a WellModel instance. This is required for {self._name}"
            )
            sys.exit()
        else:
            return smodel

    @property
    def solver(self) -> PestSolver:
        return self._solver

    @solver.setter
    def solver(self, solver: PestSolver) -> None:
        """Defines model solver object and related attributes for this BaseParameteriser instance."""
        self._solver = solver
        self.model_ws = solver.model_ws
        self.modelfile = Path(
            solver.model_ws / f"{self.stressmodel.name}.stress_pars.csv"
        )

    @staticmethod
    def _source_pts_minmax_range(source_points: DataFrame) -> float:
        """Calculate distance between two most distant coords in source points.
        Parameters
        ----------
        source_points : DataFrame
            DataFrame with 'x' and 'y' columns (x and y source point coordinates for kriging).

        Returns
        -------
        float :
            Distance between most distant coords in source points.
        """
        loc1 = [source_points.x.min(), source_points.y.min()]
        loc2 = [source_points.x.max(), source_points.y.max()]
        return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5

    @staticmethod
    def _get_zones(source_points) -> int | npt.NDArray[int]:
        """
        Define zones array or int for kriging.
        Zones are based on source_points if \"zone\" is in source_points.columns,
        otherwise zones are set uniformly to 1.

        Parameters
        ----------
        source_points : DataFrame
            DataFrame with 'x' and 'y' columns (x and y source point coordinates for kriging).

        Returns
        -------
        int | npt.NDArray[int] :
            Zone value(s) for kriging.
        """
        zns = 1
        if "zone" in source_points.columns:
            zns = source_points.zone.astype(int).values.flatten()
        return zns

    def _get_ppoint_cov(
        self,
        source_points: DataFrame,
    ) -> None:
        """
        Generate 2d kriging pilot point covariance matrix for each stressmodel stress timeseries,
        and concatenate them into a single block-diagonal matrix. Assign to BaseParameteriser.stress_parcov

        Parameters
        ----------
        source_points : DataFrame
            DataFrame with 'x' and 'y' columns (x and y source point coordinates for kriging),
            with MultiIndex of ["column_names","Datetime"] ("column_names" being stressmodel
            stress timeseries names).

        Returns
        -------
        None
        """
        if self.par_bounds is not None:
            sill = self.par_bounds.loc[self.source_points.index, ["parubnd", "parlbnd"]]
            if self.solver.par_transform == "log":
                sill_mins = sill.groupby(level="column_names")["parlbnd"].transform(
                    "min"
                )
                sill = (sill.add(sill_mins.abs(), axis="index") + 0.1).apply(
                    np.log10
                )  # log nonzero values
            sill["par_range"] = sill.parubnd - sill.parlbnd
            sill["variance"] = (sill.par_range / 4.0) ** 2

        if self.par_freq is None:
            self.stress_parcov = pyemu.Cov(
                x=self.t_variogram_sill,
                names=self.source_points.column_names,
                isdiagonal=True,
            )
        else:
            # easiest to use build_covar_matrix_2d here with zones being column_names,
            # but build_covar_matrix_2d can only handle <=10 zones. Se we concat to block diagonal matrix manually
            covs = [
                pputils.build_covar_matrix_2d(
                    ec=source_points.xs(col).x.values.flatten(),
                    nc=source_points.xs(col).y.values.flatten(),
                    zn=1,
                    vartype=1,
                    nugget=0.0,
                    aa=source_points.xs(col).vario_ranges.values,
                    sill=self.t_variogram_sill
                    if self.par_bounds is None
                    else sill.xs(col).variance.values,
                    anis=1.0,
                    bearing=0.0,
                    ldcovmat=source_points.xs(col).shape[
                        0
                    ],  # I think this is right (?). Don't think it matters as this covmat is square.
                )
                for col in source_points.index.get_level_values("column_names").unique()
            ]  # one cov per wellmodel stress timeseries -> all to be concatenated into one block diagonal cov
            names_list = [
                names
                for names in [
                    source_points.xs(col).parnme.to_list()
                    for col in source_points.index.get_level_values(
                        "column_names"
                    ).unique()
                ]
            ]
            stress_parcovs = [
                pyemu.Cov(x=cov, names=names, isdiagonal=False).df()
                for cov, names in zip(covs, names_list)
            ]
            self.stress_parcov = pyemu.Cov.from_dataframe(
                pd.concat(stress_parcovs).fillna(0.0)
            )

        return

    def _calc_factors_2d(
        self,
        source_points: DataFrame,
        target_points: DataFrame,
    ) -> npt.NDArray[np.float64]:
        """
        Generate 2d kriging factors file.

        Parameters
        ----------
        source_points : DataFrame
            DataFrame with 'x' and 'y' columns (x and y source point coordinates for kriging).
        target_points : DataFrame
            DataFrame with 'x' and 'y' columns (x and y target coordinates for kriging).

        Returns
        -------
        npt.NDArray[np.float64]
            2D matrix covmat(source_points.shape[0], npts) for source_points.
        """
        # calc_kriging_factors_2d(ecs, ncs, zns, ect, nct, znt, vartype<1:spher, 2:exp, 3:gauss, 4:pow>, krigtype, aa, anis, bearing, searchrad, maxpts, minpts, factorfile, factorfiletype)
        # # ^ first 6 vars: x, y, zone of source and target points
        self.krig_factorfile = Path(
            self.model_ws
            / f"{Path(str(self.modelfile).replace('.csv', ''))}.krigfactors.dat"
        )
        self.krig_mpts = pputils.calc_kriging_factors_2d(
            ecs=source_points.x.values.flatten(),
            ncs=source_points.y.values.flatten(),
            zns=self._get_zones(source_points),
            ect=target_points.x,
            nct=target_points.y,
            znt=np.ones(target_points.shape[0]).astype(int),
            vartype=1,
            krigtype=1,
            aa=self.t_variogram_range,
            anis=1.0,
            bearing=0.0,
            searchrad=self._source_pts_minmax_range(source_points),
            maxpts=source_points.shape[0],
            minpts=1,
            factorfile=self.krig_factorfile,
            factorfiletype=0,
        )
        return self._get_ppoint_cov(source_points)

    @staticmethod
    def _interpolate(
        func: Callable,
        sourceval: npt.NDArray[np.float64],
        targval_min: float,
        targval_max: float,
        kwargs_pputils: dict[str, Any],
    ) -> npt.NDArray[np.float64]:
        """
        Interpolate source values to target coordinates using user-specified method.

        Parameters
        ----------
        func : Callable
            Pypestutils (or other) interpolation function
        sourceval: npt.NDArray[np.float64]
            Source values from which to interpolate to target coordinates.
        targval_min : float
            Interpolation output minimum allowed value. Default is 0.0.
        targval_max : float
            Interpolation output maximum allowed value. Default is 1.0e+16.

        Returns
        -------
        targval : npt.NDArray[np.float64]
            Target location values from the interpolation.
        """

        krig_offset = 0.0
        if sourceval.min() <= 0.0:
            krig_offset = abs(sourceval.min()) + 1.0e-3  # because we krig in log space
            sourceval += krig_offset

        kwargs_pputils["sourceval"] = sourceval

        targval = func(**kwargs_pputils)
        if isinstance(targval, dict):  # kriging output dict
            targval = targval["targval"]

        targval -= krig_offset
        targval[targval < targval_min] = targval_min
        targval[targval > targval_max] = targval_max

        return targval

    @staticmethod
    def _dtindex_to_days_elapsed(dtindex: DatetimeIndex) -> Series:
        """
        Convert a Pandas datetimeindex to float ndays elapsed since min datetime.
        """
        return dtindex.to_series().sub(dtindex.min()).dt.total_seconds() / 86400.0

    @property
    def parnme_indexer(self) -> DataFrame:
        """
        Returns DataFrame indexed by PEST (pyemu.PstFrom) parnme, with columns of:
        -usecol (column name from original model file df that is being parameterised); and
        -indices (index from original model file df that is being parameterised).
        """
        return self._parnme_indexer

    def _build_parnme_indexer(self) -> None:
        # usecols = self.stress_pars.parnme.apply(lambda s: s.split("_usecol:")[-1].split("_pstyle:")[0]).rename("column_names")
        usecols = (
            self.source_points.column_names
        )  # should be in same order as stress_pars
        indices = self.stress_pars.parnme.apply(
            lambda s: s.split("_pstyle:d_datetime:")[-1]
        )
        indices = pd.to_datetime(
            indices.apply(lambda s: s.split("_column_names:")[0]),
            format="%d/%m/%Y",
        ).rename("indices")
        parnme_indexer = (
            usecols.reset_index(drop=False)
            .set_index(indices.index)
            .rename(columns={"Datetime": "indices"})
        )
        if (parnme_indexer.indices != indices.values).any():
            logger.error("parnmes and source_points are misaligned. Something's up.")
            raise Exception
        self._parnme_indexer = parnme_indexer
        self.stress_pars["column_names"] = parnme_indexer.column_names
        self.stress_pars["index_org"] = parnme_indexer.indices
        # the above will not work for pest_hp solves as can't use pyemu longnames. Hence exception below in interpolate_stresses().
        # TODO: Need a way of getting from shortnames to usecols (need to mod pyemu.PstFrom to spit the usecol name out? Would be very handy)
        # Also really would be easiest/safest if original file index value was included.
        # Possible solution: A function to read a tpl file into a dataframe (skipping line 0), and replace the marker (found in line 0) with ""
        # Then we have parnames at given df locations. So from that we can build a dataframe of row/col indexers / nans where no par exists.
        # From that, we can dropna and flatten it. Only works for structured tpl files (smp/csv types), not weirdly structured text output with complicated tpl markers.
        # But that's ok here cos we use pyemu to build everything.

    def interpolate_stresses(
        self,
        targval_min: Optional[float] = 0.0,
        targval_max: Optional[float] = 1.0e16,
        invpow=2.0,
        method: Literal["inv_dist_weighted", "step", "kriging"] = "inv_dist_weighted",
        updated_sourcevals: Optional[Series | None] = None,
    ) -> None:
        """
        Class method for PestSolver.run() to interpolate stress pilot point interpolation
        parameters to the full stress timeseries, for each stress in stressmodel.
        Updates stressmodel.stress inplace with new values, and returns a DataFrame with the
        updated data for use in pypestworker calls.

        Parameters
        ----------
        targval_min : Optional[float]
            Interpolation output minimum allowed value. Default is 0.0, on the assumption that pastas is
            using positive abstraction rates.
        targval_max : Optional[float]
            Interpolation output maximum allowed value. Default is 1.0e+16.
        method : Literal["inv_dist_weighted","step","kriging"]
            Interpolation method to apply. Method \"step\" simply forward fills and then backward fills
            between the temporal pilot point parameter values, resulting in stepped stress rates.
            Kriging results in a smoothed interpolation between the pilot points. If only a single
            parameter is defined per stress, then step interpolation is also used. Default is \"kriging\".
        updated_sourcevals: Optional[Series | None],
            Optional Series (indexed by PEST parnme) of interpolation pilot point source values per PEST
            parnme for this WellModelParameteriser instance. Provided by a PyPestWorker instance for example.
            If None, then (x)Parameteriser.modelfile is read to obtain these values. Default is None.

        Returns
        -------
        updated_source_stresses : DataFrame | Series
            Stressmodel stress TimeSeries. Series if a single stress, DataFrame if multiple (eg WellModel)
        """
        if updated_sourcevals is None:  # non-pypestworker call (worker dirs on disk)
            sourcevals = pd.read_csv(
                self.modelfile,
                index_col=["column_names", "Datetime"],
                parse_dates=["Datetime"],
                date_format=self.date_format,
            )  # read pest-updated values from disk
        else:  # pypestworker call - updated values from series (parnme:value) in memory
            """if not self.solver.long_names:
                raise Exception(
                    f"PestSolver.long_names must be True for {self._name}.interpolate_stress.updated_sourcevals to work as currently coded.\n \
                                Hence Pest_HP solver is not yet supported with this function."
                )  # see TODO note above for a possible solution."""
            sourcevals = self.modelfile_df_org.copy()
            sourcevals.loc[:, "value"] = updated_sourcevals # this should already be indexed by parnme in forward run, so in order
            # self.stress_pars (returned pstfrom() df) has usecol and parnme in it? We could use that (better than reading from disk). <--see self.parnme_indexer
            """for krig_col, df in sourcevals.groupby(level="column_names"):
                parnames = self.stress_pars.loc[
                    self.stress_pars.index.get_level_values("column_names") == krig_col
                ].parnme
                # usecols = self.parnme_indexer.loc[parnames].column_names
                indexer = self.parnme_indexer.loc[parnames].indices  # Datetime
                sourcevals.loc[[krig_col, indexer], "value"] = updated_sourcevals"""

        krig_cols = self.stress_names
        source_stresses = self.stress.copy()  # crosstab with columns of stressmodel rate timeseries (per bore). Index is datetime

        if method == "step":
            source_stresses.loc[:, krig_cols] = np.nan
            for krig_col in krig_cols:
                source_stresses.loc[:, krig_cols] = sourcevals.xs(krig_col)
                source_stresses.loc[:, krig_cols] = (
                    source_stresses.loc[:, krig_cols].ffill().bfill()
                )
        else:  # kriging or ipd interpolation over time required.
            # In hindsight, kriging won't work with pypestworker because krige_using_file requires a file...not in memory.
            # Although, this kriging factors file never changes - it's written once on add_parameters() in this class' instantiation.
            # So we probably could use it fine, but it would be slower than a purely in-memory method.
            # TODO: work out how to deal with this in pypestworker / check it works.
            if method == "kriging":
                logger.info("Interpolating stresses with kriging method.")
                logger.warning(
                    "Kriging not currently supported with pypestworker runs via PestSolver"
                )
                kwargs_pputils = dict(
                    factorfile=self.krig_factorfile,
                    factorfiletype=0,
                    mpts=self.krig_mpts,
                    krigtype=1,
                    transtype=1,
                    nointerpval=np.nan,
                )
                func = pputils.krige_using_file
            else:  # ipd
                logger.info(
                    "Interpolating stresses with inverse-power-of-distance method."
                )
                kwargs_pputils = dict(
                    ecs=self.source_points.x.values.flatten(),
                    ncs=self.source_points.y.values.flatten(),
                    zns=self._get_zones(self.source_points),
                    ect=self.target_points.x,
                    nct=self.target_points.y,
                    znt=np.ones(self.target_points.shape[0]).astype(int),
                    transtype=1,
                    anis=1.0,
                    bearing=0.0,
                    invpow=invpow,
                )
                func = pputils.ipd_interpolate_2d

            source_stresses.loc[:, krig_cols] = np.nan
            for krig_col in krig_cols:
                targval = self._interpolate(
                    func,
                    sourceval=sourcevals.xs(
                        krig_col
                    ).values,  # not right - we need to filter sourcevals from parnames per "column_names" (stress name)
                    targval_min=targval_min,
                    targval_max=targval_max,
                    kwargs_pputils=kwargs_pputils,
                )
                source_stresses.loc[self.stress.index, krig_col] = targval
                # fill nans by bfill/ffill, just in case.
                source_stresses.loc[:, krig_col] = (
                    source_stresses.loc[:, krig_col].ffill().bfill()
                )

        updated_source_stresses = source_stresses

        # replace stressmodel.stress
        for stress_series in self.stressmodel.stress:
            if stress_series in self.stress_names:
                stress_series.series_original = source_stresses.loc[
                    :, stress_series.name
                ]

        return updated_source_stresses  # self.stress #self.stressmodel.stress

    def add_stress_obs(
            self,
            obs_data : Series = None,

    ) -> None:
        """
        Add observations of stress rates for pest.

        Parameters
        ----------
        obs_data : Optional[Series]
            Observed stress value data points. Indexed by datetime.

        Returns
        -------
        None
        """

        self.obs_data = obs_data

    def _mod2obs(self) -> DataFrame:
        """
        Internal method to intrpolate modelled stress rates to observed datetimes.
        """
        # define interp function: stressmodel stress rates to obs data datetimes
        # insert obs indices --> interp(linear) -->keep only obs dts
        new_idx = self.stress.index.union(self.obs_data.index, sort=True)
        modobs = self.stress.reindex(new_idx).melt(
                var_name="column_names", ignore_index=False
            ).groupby(by="column_names").apply(
            lambda x: x.interpolate(method='linear')
        )
        modobs = modobs.loc[self.obs_data.index].reset_index(
            drop=False
        ).set_index(
            ["Datetime","column_names"]
        )

        return modobs

class WellModelParameteriser(BaseParameteriser):
    """
    Custom WellModel stress parameteriser.
    Designed to be first instantiated outside the PestSolver for a given StressModel,
    and then passed to PestSolver via the pest_parameterisers argument.

    Parameters
    ----------
    model : pastas.Model
        Pastas model.
    wellmodel_name : str
        WellModel.name for which to apply WellModel.stress rate parameters to. Must refer to a WellModel object.
    date_format : Optional[str]
        Datetime format for saving PEST model parameter input files for stressmodel. Default is \"%d/%m/%Y\".
    interp_kwargs : Optional[dict[str, Any]]
        kwargs to pass to BaseParameteriser.interpolate_stresses(). Default is {}.
    stress_names : list[str] | None, optional
        List of WellModel.stress.TimeSeries names for which stress rate parameters are to be optimised.
        If None, all stresses in the wellmodel are parameterised. Default is None.
    par_freq : str | None, optional
        If 'at_rate_changes': pilot points are placed at rate step change points (and first record). Covariance
        range of these points is then defined based on the median time interval between adjacent pilot points
        (multiplied by t_variogram_range_freq_factor, and capped by max_vario_range; see below).
        Otherwise: Frequency at which temporal WellModel rate pilot points are defined for each stress in stress_names.
        Must be None or one of the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        If None, a single (constant-in-time) stress rate parameter is defined for all stresses in stress_names.
        Default is None.
    t_variogram_range_freq_factor: float, optional
        par_freq factor to define temporal variogram range to build a parameter covariance matrix for input to
        pyemu.helpers.first_order_pearson_tikhonov(). Default is 2.0, so for example if par_freq is 365D, this means
        pars covary up to the sill variance over 730D.
    max_vario_range Optional[float] :
        Maximum variogram range for temporal interpolation points (units: days). Default is 730.0.
    t_variogram_sill: float, optional
        Temporal variogram sill (variance at t_variogram_range) used to build a scaling parameter
        covariance matrix for input to pyemu.helpers.first_order_pearson_tikhonov().
        If PestSolver.par_transform == "log", then this must pertain to the log of the parameters.
        If par_freq is None, t_variogram_sill is used to define parameter variance in the returned
        diagonal prior (co)variance matrix. Default is 1.0.
        t_variogram_sill is ignored if par_bounds is provided.
    par_bounds: Optional[DataFrame| None]
        DataFrame multiindexed by [stressmodel stress TimeSeries name (bore), Datetime].
        Columns must include 'parlbnd' and 'parubnd'; these must be in untransformed parameter space.
        Assigned to nearest pilot point in time to Datetime (depends on par_freq).
        Bounds will be used to define interpolation and covariance sill value (variance) by dividing par range
        by 4.0 and squaring that (95%CI assumed); that variance overrides t_variogram_sill if par_bounds is provided.

    Attributes
    ----------
    model_file : Path
        Path to parameterised model input file to be used by solver.run(),
        with pest tpl file to be constructed using pyemu.PestFrom.add_parameters()
    krig_factorfile : Path
        Path to kriging factors file for use during PestSolver.run() calls.
    stress_pars : DataFrame
        DataFrame of stressmodel.stress parameterisation info, as returned by solver.pf (pyemu.PestFrom).
    stress_parcov  : pyemu.Cov
        Prior covariance matrix for the WellModel parameters defined through this class.
        For use in building PEST uncertainty (.unc) files.

    Returns
    -------
    None
    """

    _name = "WellModelParameteriser"

    def __init__(
        self,
        model: Model,
        wellmodel_name: str,
        date_format: Optional[str] = "%d/%m/%Y",
        interp_kwargs: Optional[dict[str, Any]] = {},
        stress_names: list[str] | None = None,
        par_freq: str | None = None,
        t_variogram_range_freq_factor: float | None = None,
        max_vario_range: Optional[float] = 730.0,
        t_variogram_sill: float = 1.0,
        par_bounds: Optional[DataFrame | None] = None,
    ) -> None:
        BaseParameteriser.__init__(
            self,
            model=model,
            stressmodel_name=wellmodel_name,
            date_format=date_format,
            interp_kwargs=interp_kwargs,
        )

        if (
            stress_names is not None
        ):  # replace default all stress_names (from BaseParameteriser.__init__() with only a selection to parameterise
            self.stress_names = stress_names
        logger.info(
            rf"Modelling {len(self.stress_names)} stress names in {wellmodel_name}:\n --> {', '.join(self.stress_names)}"
        )
        if par_freq:
            if par_freq.lower() == "at_rate_changes":
                self.par_freq = par_freq
            else:
                self.par_freq = _frequency_is_supported(par_freq)
        else:
            self.par_freq = None
        self.t_variogram_range_freq_factor = t_variogram_range_freq_factor
        self.t_variogram_sill = t_variogram_sill
        self.max_vario_range = max_vario_range
        self.par_bounds = par_bounds

        # filter stress based on provided wellmodel_names
        self.stress = self.stress.filter(items=self.stress_names, axis="columns")

    def _get_stress_pars(self, par_name_base: str) -> DataFrame:
        """Build interpolation source points for None, "par_freq" and "at_rate_changes" methods"""
        if self.par_freq == "at_rate_changes":
            source_points = self.stress.diff().melt(
                var_name="column_names", ignore_index=False
            )  # value_name (stress rate) is left at "value"
            # add first and last entry as a par too
            source_points["firstdt"] = (
                source_points.assign(Datetime=source_points.index)[
                    ["Datetime", "column_names"]
                ]
                .groupby(by="column_names")
                .transform("min")
                .Datetime
            )
            source_points["lastdt"] = (
                source_points.assign(Datetime=source_points.index)[
                    ["Datetime", "column_names"]
                ]
                .groupby(by="column_names")
                .transform("max")
                .Datetime
            )
            source_points.loc[
                (source_points.firstdt == source_points.index)
                | (source_points.lastdt == source_points.index),
                "value",
            ] = 1
            source_points.drop(columns=["firstdt", "lastdt"], inplace=True)
            # drop zeroes
            source_points = source_points[source_points["value"] != 0.0].copy()
            # convert stress datetime to timedelta from t0 as float(totaldays) for kriging
            source_points["x"] = self._dtindex_to_days_elapsed(source_points.index)
            # define geostat variogram range for parameter interpolation (in krig_t space)
            source_points.loc[:, "intervals"] = (
                source_points[["x", "column_names"]]
                .groupby(by="column_names")
                .transform("diff")
                .fillna(0.0)
                .x
            )
            source_points.loc[:, "median_intervals"] = (
                source_points[["intervals", "column_names"]]
                .groupby(by="column_names")
                .transform("median")
                .intervals
            )
            """source_points = source_points.reset_index(drop=False).set_index(["Datetime", "column_names"])
            windexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
            source_points.loc[:, "rolling_4xmean_intervals"] = (
                source_points.groupby(level=["column_names"])["intervals"]
                .rolling(window=windexer, min_periods=1)
                .mean().ffill().bfill().droplevel(level=-1).values #transform(lambda x: x)
            )
            source_points.to_csv("temp.source_points.csv", date_format=self.date_format)
            source_points = source_points.reset_index(drop=False).set_index("Datetime")
            source_points.loc[:, "vario_ranges"] = (
                source_points.rolling_4xmean_intervals * self.t_variogram_range_freq_factor
            ).clip(upper=self.max_vario_range)"""
            source_points.loc[:, "vario_ranges"] = (
                source_points.median_intervals * self.t_variogram_range_freq_factor
            ).clip(upper=self.max_vario_range)

        elif self.par_freq is not None:  # regular frequency pilot points
            stress_pars = self.stress.resample(self.par_freq).first()
            if (
                self.stress.index.max() not in stress_pars.index
            ):  # include the last time so we don't get boundary effects in the interp
                stress_pars.reindex(
                    stress_pars.index.to_list() + [self.stress.index.max()]
                )
                stress_pars.loc[stress_pars.index] = self.stress.loc[stress_pars.index]
            source_points = stress_pars.melt(
                var_name="column_names", ignore_index=False
            )  # value_name (stress rate) is left at "value"
            # convert stress datetime to timedelta from t0 as float(totaldays) for kriging
            source_points["x"] = self._dtindex_to_days_elapsed(source_points.index)
            # define geostat variogram range for parameter interpolation (in krig_t space)
            self.t_variogram_range = (
                pd.to_timedelta(self.par_freq) * self.t_variogram_range_freq_factor
            ).total_seconds() / 86400.0
            source_points.loc[:, "vario_ranges"] = min(
                self.t_variogram_range, self.max_vario_range
            )

        elif (
            self.par_freq is None
        ):  # - a single parameter per stress TimeSeries, which is applied constant in time
            # constant-in-time scaling parameter applied
            source_points = self.stress.iloc[[0], :].melt(
                var_name="column_names", ignore_index=False
            )  # value_name (stress rate) is left at "value"
        else:
            logger.error(f"Unsupported value for par_freq provided ({self.par_freq}).")
            raise Exception

        # revert flow rate diffs to flow rates
        source_points.index.name = "Datetime"  # does this work? should do now with df.melt(ignore_index=False) above
        source_points = source_points.reset_index(drop=False).set_index(
            ["column_names", "Datetime"]
        )
        source_points["value"] = (
            self.stress.melt(var_name="column_names", ignore_index=False)
            .reset_index(drop=False)
            .set_index(["column_names", "Datetime"])["value"]
        )
        source_points = source_points.reset_index(drop=False).set_index("Datetime")

        # convert stress datetime to timedelta from t0 as float(totaldays) for kriging
        source_points["x"] = self._dtindex_to_days_elapsed(source_points.index)
        source_points["y"] = 1.0

        source_points.to_csv(
            self.modelfile, date_format=self.date_format
        )  # parameterised by pstfrom

        index_cols = [source_points.index.name, "column_names"]
        use_cols = ["value"]
        pargp_indices = (
            source_points.column_names != source_points.column_names.shift()
        ).cumsum()
        pargp = pargp_indices.apply(
            lambda s: f"wellq.{self.stressmodel.name}.{str(s).zfill(2)}"
        ).to_list()

        self.stress_pars = self.solver.pf.add_parameters(
            self.modelfile,
            index_cols=index_cols,
            use_cols=use_cols,
            par_type="grid",
            par_style="direct",
            transform=self.solver.par_transform,
            pargp=pargp,
            par_name_base=par_name_base,
            # lower_bound=self.ml.parameters.loc[self.vary, "pmin"].values.tolist(),
            # upper_bound=self.ml.parameters.loc[self.vary, "pmax"].values.tolist(),
            # ult_lbound = self.ml.parameters.loc[self.vary, ["pmin"]].transpose().values.tolist(),
            # ult_ubound = self.ml.parameters.loc[self.vary, ["pmax"]].transpose().values.tolist(),
        )

        return source_points

    def add_stress_parameters(self, par_name_base: str) -> None:
        """
        Add WellModel pumping rate parameters for PestSolver.model.pf (pyemu.PstFrom) for each WellModel stress TimeSeries.
        Modifies the solver.pf (pyemu.PstFrom) object in-place on calling this function.

        Parameters
        ----------
        par_name_base : str
            PEST parameter name base to provide to self.pf (pyemu.PstFrom). Required, because each WellModelParameteriser
            instance should have a unique parameter base name.

        Returns
        -------
        None
        """
        # check that solver is defined first. It must be for this method to operate.
        if self.solver is None:
            logger.error(
                "Solver not yet defined for {self._name}, so can't add stress parameters yet.\n\
                         Define a solver before calling f{self._name}.add_stress_parameters()"
            )
            sys.exit()

        if (
            self.solver.par_transform == "log"
            and self.t_variogram_range_freq_factor is not None
        ):
            logger.warning(
                't_variogram_sill provided to PestSolver.add_well_rate_parameters() must pertain to log-transfomed parameter space (PestSolver.par_transform == "log")'
            )

        self.stress.index.name = "Datetime"

        # define target points for kriging by time
        target_points = self.stress.copy()
        target_points["x"] = self._dtindex_to_days_elapsed(target_points.index)
        target_points["y"] = 1.0
        target_points.drop(
            columns=[c for c in target_points.columns if c not in ["x", "y"]],
            inplace=True,
        )
        self.target_points = target_points
        # define source points for kriging / build pest tpl file
        self.source_points = self._get_stress_pars(par_name_base)

        # build index between pest parnames from self.stress_pars and self.modelfile parameterised columns and indices.
        self._build_parnme_indexer()  # now self.parnme_indexer is gettable. Also adds a usecol field to the self.stress_pars df (and "index_org" - the original index values from stress df)

        # assign parnme to source_points; make all df indices a mux: (stressmodelname,datetime)
        self.stress_pars = self.stress_pars.reset_index(drop=False).set_index(
            ["column_names", "index_org"]
        )
        self.source_points = self.source_points.reset_index(drop=False).set_index(
            ["column_names", "Datetime"]
        )
        self.source_points["parnme"] = self.stress_pars.parnme

        # TODO: DEFINE/HANDLE RATE PAR BOUNDS
        # Need a dict or df of stress TimeSeries name: ubnd/lbnd at a minimum.
        # Probs need time field in there too, so maybe initial stress rates pilot points can be used,
        # with user-supplied ubnd and lbnd factors of the initial rate. <- this is it
        if self.par_bounds is not None:
            self.stress_pars.loc[:, ["parlbnd", "parubnd"]] = self.par_bounds.loc[
                :, ["parlbnd", "parubnd"]
            ]

        # make pcov for pilot points
        self._get_ppoint_cov(self.source_points)

        # and save a copy of self.modelfile data in memory for pypestworker updates
        self.modelfile_df_org = pd.read_csv(
            self.modelfile, index_col=[1,0], date_format=self.date_format
        )
