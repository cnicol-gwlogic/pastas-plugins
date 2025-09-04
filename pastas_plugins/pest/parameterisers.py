import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame, DatetimeIndex, Series
from pastas import Model
from pastas.stressmodels import StressModel, WellModel
from pastas.timeseries_utils import _frequency_is_supported
from pyemu import Cov
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
    ) -> npt.NDArray[np.float64]:
        """
        Generate 2d kriging pilot point covariance matrix.

        Parameters
        ----------
        source_points : DataFrame
            DataFrame with 'x' and 'y' columns (x and y source point coordinates for kriging).

        Returns
        -------
        npt.NDArray[np.float64] :
            2D matrix covmat(source_points.shape[0], npts) for source_points.
        """
        ppoint_cov = pputils.build_covar_matrix_2d(
            ecs=source_points.x.values.flatten(),
            ncs=source_points.y.values.flatten(),
            zn=self._get_zones(source_points),
            vartype=1,
            nugget=0.0,
            aa=self.t_variogram_range,
            sill=self.t_variogram_sill,
            anis=1.0,
            bearing=0.0,
            ldcovmat=source_points.shape[
                0
            ],  # I think this is right (?). Don't think it matters as this covmat is square.
        )
        return ppoint_cov

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
            self.model_ws / f"{self.modelfile.replace('.csv', '')}.krigfactors.dat"
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
    def _dtindex_to_days_elapsed(dtindex: DatetimeIndex) -> Series[float]:
        """
        Convert a Pandas datetimeindex to float ndays elapsed since min datetime.
        """
        return dtindex.to_series().sub(dtindex.min()).total_seconds() / 86400.0

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
        if not updated_sourcevals:  # non-pypestworker call (worker dirs on disk)
            sourcevals = pd.read_csv(
                self.modelfile,
                index_col=0,
                parse_dates=[0],
                date_format=self.date_format,
            )  # one col per stress. Need to krig or step interp each for which we have adjustable parameters.
        else:  # pypestworker call
            if not self.solver.longnames:
                raise Exception(
                    f"PestSolver.longnames must be True for {self._name}.interpolate_stress.updated_)sourcevals to work as currently coded.\n \
                                Hence Pest_HP solver is not yet supported with this function."
                )
            sourcevals = self.modelfile_df_org.copy()
            # self.stress_pars (returned pstfrom() df) has usecol and parnme in it? We could use that (better than reading from disk).
            usecols = self.stress_pars.parnme.split("_usecol:")[-1].split("_pstyle:")[0]
            indices = pd.to_datetime(
                self.stress_pars.parnme.split("_pstyle:d_datetime:")[-1],
                format="%d/%m/%Y",
            )
            parnme_indexer = pd.DataFrame(
                pd.concat([indices, usecols], axis=1),
                index=self.stress_pars.parnme,
                names=["indices", "column_names"],
            )
            # the above willnot work for pest_hp solves as can't use pyemu longnames. Hence exception above.
            # TODO: Need a way of getting from shortnames to usecols (need to mod pyemu.PstFrom to spit the usecol name out? Would be very handy)
            # Also really would be easiest/safest if original file index value was included.
            for krig_col in sourcevals.columns:
                parnames = self.stress_pars.loc[usecols == krig_col].parnme
                usecols = parnme_indexer.loc[parnames].columns_names
                indexer = parnme_indexer.loc[parnames].indices
                sourcevals.loc[indexer, usecols] = updated_sourcevals.loc[parnames]

        krig_cols = self.stress_names
        source_stresses = self.stress.copy()

        if sourcevals.shape[0] == 1:  # constant-in-time ffill / bfill needed
            method = "step"  # force step interp for single parameter values per stress TimeSeries

        if method == "step":
            source_stresses.loc[:, krig_cols] = np.nan
            source_stresses.loc[sourcevals.index, krig_cols] = sourcevals
            source_stresses.loc[:, krig_cols] = (
                source_stresses.loc[:, krig_cols].ffill().bfill()
            )
        else:  # kriging or ipd interpolation over time required.
            # In hindsight, kriging won't work with pypestworker because krige_using_file requires a file...not in memory.
            # TODO: work out how to deal with this in pypestworker.
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
                    sourceval=sourcevals.loc[:, krig_col].values,
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

        # replace stressmodel.stress -> TODO: check that this works ok
        for stress_series in self.stressmodel.stress:
            stress_series.series_original = source_stresses.loc[:, stress_series.name]

        return updated_source_stresses  # self.stress #self.stressmodel.stress


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
        Frequency at which temporal WellModel rate pilot points are defined for each stress in stress_names.
        Must be None or one of the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
        If None, a single (constant-in-time) stress rate parameter is defined for all stresses in stress_names.
        Default is None.
    t_variogram_range_freq_factor: float, optional
        par_freq factor to define temporal variogram range to build a parameter covariance matrix for input to
        pyemu.helpers.first_order_pearson_tikhonov(). Default is 2.0, so for example if par_freq is 365D, this means
        pars covary up to the sill variance over 730D.
    t_variogram_sill: float, optional
        Temporal variogram sill (variance at t_variogram_range) used to build a scaling parameter
        covariance matrix for input to pyemu.helpers.first_order_pearson_tikhonov(). If PestSolver.par_transform == "log", then
        this must pertain to the log of the parameters. If par_freq is None, t_variogram_sill is used to define parameter
        variance in the returned diagonal prior (co)variance matrix. Default is 1.0

    Attributes
    ----------
    model_file : Path
        Path to parameterised model input file to be used by solver.run(), with pest tpl file to be constructed using pyemu.PestFrom.add_parameters()
    krig_factorfile : Path
        Path to kriging factors file for use during PestSolver.run() calls.
    stress_pars : DataFrame
        DataFrame of stressmodel.stress parameterisation info, as returned by solver.pf (pyemu.PestFrom).
    stress_parcov  : pyemu.Cov
        Prior covariance matrix for the WellModel parameters defined through this class. For use in building PEST uncertainty (.unc) files.

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
        t_variogram_sill: float = 1.0,
    ) -> None:
        super().__init__(
            self,
            model=model,
            stressmodel=wellmodel_name,
            date_format=date_format,
            interp_kwargs=interp_kwargs,
        )

        if (
            stress_names is not None
        ):  # replace default all stress_names with only a selection to parameterise
            self.stress_names = stress_names
        if par_freq:
            self.par_freq = _frequency_is_supported(par_freq)
        else:
            self.par_freq = None
        self.t_variogram_range_freq_factor = t_variogram_range_freq_factor
        self.t_variogram_sill = t_variogram_sill

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
                't_variogram_sill provided to PestSolver.add_well_rate_parameters() must pertain to log-transfomed \
                parameter space (PestSolver.par_transform == "log"'
            )

        # TODO: DEFINE/HANDLE RATE PAR BOUNDS

        # TODO: check par_name_base is not already in self.solver.pf - error if it is
        # (if pyemu.PstFrom.add_parameters() doesn't deal with incrementing par names/indices.)

        if self.par_freq is None:
            # constant-in-time scaling parameter applied
            self.stress.iloc[0, :].to_csv(self.modelfile, date_format=self.date_format)
            self.stress_pars = self.solver.pf.add_parameters(
                self.modelfile,
                index_cols=[self.stress.index.name],
                use_cols=self.stress_names,
                par_type="grid",
                par_style="direct",
                transform=self.solver.par_transform,
                pargp=[
                    f"wellq.{self.stressmodel.name}{str(idx).zfill(2)}"
                    for idx, col in enumerate(self.stress.columns)
                ],
                par_name_base=par_name_base,
                # lower_bound=self.ml.parameters.loc[self.vary, "pmin"].values.tolist(),
                # upper_bound=self.ml.parameters.loc[self.vary, "pmax"].values.tolist(),
                # ult_lbound = self.ml.parameters.loc[self.vary, ["pmin"]].transpose().values.tolist(),
                # ult_ubound = self.ml.parameters.loc[self.vary, ["pmax"]].transpose().values.tolist(),
            )

            self.stress_parcov = Cov(
                x=self.t_variogram_sill, names=self.stress_pars.index, isdiagonal=True
            )

        else:
            # define target points for kriging by time
            target_points = self.stress.copy()
            target_points["x"] = super()._dtindex_to_days_elapsed(target_points.index)
            target_points["y"] = 1.0
            target_points.drop(
                columns=[c for c in target_points.columns if c not in ["x", "y"]],
                inplace=True,
            )
            # define source points for kriging
            stress_pars = self.stress.resample(self.par_freq).first()
            if (
                self.stress.index.max() not in stress_pars.index
            ):  # include the last time so we don't get boundary effects in the interp
                stress_pars.reindex(
                    stress_pars.index.to_list() + [self.stress.index.max()]
                )
                stress_pars.loc[stress_pars.index] = self.stress.loc[stress_pars.index]
            # convert stress datetime to timedelta from t0 as float(totaldays) for kriging
            source_points = stress_pars.copy()
            source_points["x"] = super()._dtindex_to_days_elapsed(stress_pars.index)
            source_points["y"] = 1.0
            source_points.drop(
                columns=[c for c in source_points.columns if c not in ["x", "y"]],
                inplace=True,
            )
            self.source_points = source_points
            self.target_points = target_points
            # define geostat variogram range for parameter interpolation (in krig_t space)
            self.t_variogram_range = (
                pd.to_timedelta(self.par_freq) * self.t_variogram_range_freq_factor
            ).total_seconds() / 86400.0
            stress_parcov = super()._calc_factors_2d(
                source_points=source_points,
                target_points=target_points,
            )
            stress_pars.to_csv(self.modelfile, date_format=self.date_format)
            self.stress_pars = self.solver.pf.add_parameters(
                self.modelfile,
                index_cols=[stress_pars.index.name],
                use_cols=self.stress_names,
                par_type="grid",
                par_style="direct",
                transform=self.solver.par_transform,
                pargp=[
                    f"wellq.{self.stressmodel.name}{str(idx).zfill(2)}"
                    for idx, col in enumerate(stress_pars.columns)
                ],
                par_name_base=[
                    f"{par_name_base}.{str(idx).zfill(2)}"
                    for idx, col in enumerate(stress_pars.columns)
                ],
                # lower_bound=self.ml.parameters.loc[self.vary, "pmin"].values.tolist(),
                # upper_bound=self.ml.parameters.loc[self.vary, "pmax"].values.tolist(),
                # ult_lbound = self.ml.parameters.loc[self.vary, ["pmin"]].transpose().values.tolist(),
                # ult_ubound = self.ml.parameters.loc[self.vary, ["pmax"]].transpose().values.tolist(),
            )
            self.stress_parcov = Cov(
                x=stress_parcov, names=self.stress_pars.index, isdiagonal=False
            )
            # and save a copy of self.modelfile data in memory for pypestworker updates
            self.modelfile_df_org = pd.read_csv(
                self.modelfile, index_col=0, date_format=self.date_format
            )
            """
            # define a covariance matrix called cov using pyemu's geostatistics capabilities
            cov = gs.covariance_matrix(df_pp.x, df_pp.y, df_pp.parnme)
            # use the pyemu helper to construct preferred difference regularization equations 
            # using the covariance for regularization weight
            pyemu.helpers.first_order_pearson_tikhonov(pst, cov, reset=False)
            """

            # finally, pickle to disk for pest workers
            with open(f"{self.stressmodel.name}.parameteriser.pkl", "wb") as f:
                pickle.dump(self, os.path.join(self.solver.temp_ws, f))
