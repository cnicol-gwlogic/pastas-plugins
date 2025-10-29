import sys
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, Optional
from shutil import copy as copy_file

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
    "Parameteriser",
]


class Parameteriser:
    """
    Custom PEST parameteriser base class for StressModel stresses.
    Designed to be first instantiated outside the PestSolver for a given StressModel,
    and then passed to PestSolver via the pest_parameterisers argument.

    Parameters
    ----------
    parameteriser_name: str
        Base name used in PEST stress parameter files for this parameteriser instance.
    stress: DataFrame
        Stressmodel stresses dataframe, for example attained from stressmodel.get_stress(squeeze=False), or
        a subset of that. These are the stress Timeseries across all models in models list and stressmodel_names list
        that will be updated by this parameteriser.
    model_ws: str
        Model working directory.
    model_names: list
        List of model names for which the stresses provided will be updated.
    stressmodel_names: list,
        List of stressmodel names for which the stresses provided will be updated.
    date_format : Optional[str]
        Datetime format for saving PEST model parameter input files for stressmodel. Default is \"%d/%m/%Y %H:%M:%S\".
    interp_kwargs : Optional[dict[str, Any]]
        kwargs to pass to BaseParameteriser.interpolate_stresses(). Default is {}.
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
    stress_minmax_dates : NOT USED ANYMORE TODO: REMOVE Optional[DataFrame| None]
        Dataframe indexed by stressmodel stress TimeSeries name (bore), with columns of min_date and max_date for
        non-zero stress values. min_date and max_date can contain null (NaT) values, in which case min and max dates
        for non-zero stress values are set to the min/max date of the stressmodel stresses. Stress parameters outside
        of this date range are fixed at zero.

    Attributes
    ----------
    stress : DataFrame | Series
        Stressmodel stress TimeSeries.  Series if a single stress, DataFrame if multiple (eg WellModel)
    stress_pars : DataFrame
        DataFrame of stressmodel.stress parameterisation info, as returned by solver.pf (pyemu.PestFrom).
    stress_parcov  : pyemu.Cov
        Prior covariance matrix for the WellModel parameters defined through this class.
        For use in building PEST uncertainty (.unc) files.
    modelfile : Path
        PestSolver model input file for scaling (optimising) stressmodel stresses during PEST runs.
    modelfile_df_org : DataFrame
        DataFrame of original PestSolver model input file for scaling (optimising) stressmodel stresses during PEST runs.
    krig_factorfile : Path
        Path to kriging factors file for use during PestSolver.run() calls.
    krig_mpts : int
        Number of kriging target points.
    Returns
    -------
    None
    """

    _name = "Parameteriser"

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

    def __init__(
        self,
        parameteriser_name: str,
        stress: DataFrame,
        model_ws: str,
        model_names: list,
        stressmodel_names: list,
        date_format: Optional[str] = "%d/%m/%Y",
        interp_kwargs: Optional[dict[str, Any]] = {},
        par_freq: str | None = None,
        t_variogram_range_freq_factor: float | None = None,
        max_vario_range: Optional[float] = 730.0,
        t_variogram_sill: float = 1.0,
        par_bounds: Optional[DataFrame | None] = None,
    ) -> None:
        self.date_format = date_format
        self.interp_kwargs = interp_kwargs

        # PestSolver related things.
        self.model_ws = model_ws
        self.model_names = model_names
        self.stressmodel_names = stressmodel_names
        self.parameteriser_name = parameteriser_name
        self.modelfile = Path(
            self.model_ws / f"{self.parameteriser_name}.stress_pars.csv"
        )
        self.modelfile_df_org = None

        # Stress TimeSeries
        self.stress = stress
        self.stress_names = self.stress.columns.to_list()
        # stress obs data
        self.obs_data = None

        # Parameterisation things
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
        self.source_points = None
        self.target_points = None
        self.krig_factorfile = None
        self.krig_mpts = None
        self.stress_pars = None
        self.stress_parcov = None
        self._parnme_indexer = None

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
        solver: PestSolver,
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
            if "partrans" not in sill.columns:
                sill["partrans"] = "log" if solver.par_transform == "log" else "none"
            logmask = sill.partrans == "log"
            if logmask.any():
                sill_mins = sill.loc[logmask].groupby(level="column_names")["parlbnd"].transform(
                    "min"
                )
                sill.loc[logmask, ["parubnd", "parlbnd"]] = (sill.loc[logmask, ["parubnd", "parlbnd"]].add(
                    sill_mins.abs(), axis="index") + 0.1).apply(
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

            for var in [stress_parcovs, covs, names_list, sill, logmask]:
                var = None

        return

    def _calc_factors_2d(
        self,
        source_points: DataFrame,
        target_points: DataFrame,
    ) -> None:
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
        None
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
        return

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

        for var in [usecols, indices, parnme_indexer]:
            var = None

    def interpolate_stresses(
        self,
        targval_min: Optional[float] = 0.0,
        targval_max: Optional[float] = 1.0e16,
        invpow=2.0,
        method: Literal["inv_dist_weighted", "step", "kriging"] = "inv_dist_weighted",
        updated_sourcevals: Optional[Series | None] = None,
        stress_names: Optional[list | None] = None,
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
        stress_names: Optional[list | None],
            Optional list of stressmodel stress names (istress column names in stressmodel.get_stress() df)
            which will be updated. Default is None.

        Returns
        -------
        updated_source_stresses : DataFrame | Series
            Stressmodel stress TimeSeries. Series if a single stress, DataFrame if multiple (eg WellModel)
        """
        if updated_sourcevals is None:  # non-pypestworker call (worker dirs on disk)
            sourcevals = pd.read_csv(
                self.modelfile.name,
                index_col=["column_names", "Datetime"],
                parse_dates=["Datetime"],
                date_format=self.date_format,
            )  # read pest-updated values from disk
        else:  # pypestworker call - updated values from series (parnme:value) in memory
            sourcevals = self.modelfile_df_org.reset_index(drop=False).set_index("parnme")
            sourcevals.loc[:, "value"] = updated_sourcevals # this should already be indexed by parnme in forward run, so in order
            sourcevals = sourcevals.reset_index(drop=False).set_index(["column_names","Datetime"])

        krig_cols = self.stress_names
        if stress_names is not None:
            krig_cols = stress_names
        source_stresses = self.stress.copy()  # crosstab with columns of stressmodel rate timeseries (per bore). Index is datetime

        if method == "step":
            source_stresses.loc[:, krig_cols] = np.nan
            for krig_col in krig_cols:
                source_stresses.loc[:, krig_col] = sourcevals.xs(krig_col).loc[:,"value"]
                source_stresses.loc[:, krig_col] = (
                    source_stresses.loc[:, krig_col].ffill().bfill()
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
                self._calc_factors_2d(self.source_points, self.target_points)
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
                    ).loc[:,"value"].values,
                    targval_min=targval_min,
                    targval_max=targval_max,
                    kwargs_pputils=kwargs_pputils,
                )
                source_stresses.loc[self.stress.index, krig_col] = targval
                # fill nans by bfill/ffill, just in case.
                source_stresses.loc[:, krig_col] = (
                    source_stresses.loc[:, krig_col].ffill().bfill()
                )
            targval = None

        # replace stressmodel.stress
        self.stress.loc[:, krig_cols] = source_stresses.loc[:, krig_cols]

        for var in [sourcevals, source_stresses, krig_cols]:
            var = None

        return self.stress

    def add_stress_obs(
            self,
            obs_data: Optional[Series | None] = None,
    ) -> None:
        """
        Add observations of stress rates for pest.

        Parameters
        ----------
        obs_data : Optional[Series]
            Observed stress value data points. Indexed by [column_names (bore), Datetime]

        Returns
        -------
        None
        """
        self.obs_data = obs_data
        self.obs_data.index.names = ["column_names","date"]

    def mod2obs(self) -> Series:
        """
        Interpolate modelled stress rates to observed datetimes.
        """
        # insert obs indices --> interp(linear) -->keep only obs dts
        self.stress.index.name = "Datetime"  # somewhere this has reverted to None...no idea why/where.
        modobs = self.stress.melt(
            var_name="column_names", ignore_index=False,
            value_name="Observations"
        ).reset_index(drop=False).set_index(["column_names","Datetime"])
        modobs.index.names = ["column_names","date"]
        new_idx = modobs.index.union(self.obs_data.index)
        modobs = modobs.reindex(new_idx).groupby(level="column_names").transform(
            lambda x: x.interpolate(method='linear')
        )
        modobs = modobs.loc[self.obs_data.index].Observations

        for var in [new_idx]:
            var = None

        return modobs

    def _get_stress_pars(self, solver: PestSolver, par_name_base: str) -> DataFrame:
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

            stress_pars = None

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
        copy_file(self.modelfile, solver.temp_ws / self.modelfile.name)

        index_cols = [source_points.index.name, "column_names"]
        use_cols = ["value"]
        pargp_indices = (
                source_points.column_names != source_points.column_names.shift()
        ).cumsum()
        pargp = pargp_indices.apply(
            lambda s: f"{self.parameteriser_name}.{str(s).zfill(2)}"
        ).to_list()

        self.stress_pars = solver.pf.add_parameters(
            self.modelfile,
            index_cols=index_cols,
            use_cols=use_cols,
            par_type="grid",
            par_style="direct",
            transform=solver.par_transform,
            pargp=pargp,
            par_name_base=par_name_base,
        )

        for var in [pargp, pargp_indices]:
            var = None

        return source_points

    def add_stress_parameters(self, solver, par_name_base: str) -> None:
        """
        Add WellModel pumping rate parameters for PestSolver.model.pf (pyemu.PstFrom) for each WellModel stress TimeSeries.
        Modifies the solver.pf (pyemu.PstFrom) object in-place on calling this function.

        Parameters
        ----------
        solver : PestSolver
            PestSolver instance from which this function is called on this stressmodel parameteriser.
        par_name_base : str
            PEST parameter name base to provide to self.pf (pyemu.PstFrom). Required, because each WellModelParameteriser
            instance should have a unique parameter base name.

        Returns
        -------
        None
        """
        if (
                solver.par_transform == "log"
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
        self.source_points = self._get_stress_pars(solver, par_name_base)

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

        # Define rate parameter bounds
        # self.par_bounds is a multiiindex df of [TimeSeries name, datetime]: stress parubnd/parlbnd.
        if self.par_bounds is not None:
            self.stress_pars.loc[:, ["parval1", "partrans", "parlbnd", "parubnd"]] = self.par_bounds.loc[
                :, ["parval1", "partrans", "parlbnd", "parubnd"]
            ]

        # make pcov for pilot points
        self._get_ppoint_cov(self.source_points, solver)

        # and save a copy of self.modelfile data in memory for pypestworker updates
        self.modelfile_df_org = pd.read_csv(
            self.modelfile, index_col=[1, 0], date_format=self.date_format
        )
        self.modelfile_df_org["parnme"] = self.source_points.parnme
