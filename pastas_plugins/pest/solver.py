import json, shutil
import logging
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from platform import node as get_computername
from shutil import copy as copy_file
from typing import Any, Literal, Optional

import dill  # pickle
import gzip
import numpy as np
import pandas as pd
import pyemu
from numpy.typing import NDArray
from pandas import DataFrame,Series
from pastas.solver import BaseSolver
from pastas.typing import TimestampType
from psutil import cpu_count
from scipy.stats import norm, truncnorm

from pastas_plugins.pest.forward_run import run, run_pypestworker

logger = logging.getLogger(__name__)


class PestSolver(BaseSolver):
    """PEST solver base class"""

    def __init__(
        self,
        exe_name: str | Path,
        model_ws: Optional[str | Path] = Path("pastas_files"),
        temp_ws: Optional[str | Path] = Path("pest"),
        master_ws: Optional[str | Path] = Path("pest"),
        noptmax: Optional[int] = 0,
        control_data: Optional[dict[str, Any] | None] = None,
        pcov: Optional[DataFrame | None] = None,
        nfev: Optional[int | None] = None,
        long_names: Optional[bool] = True,
        port_number: Optional[int] = 4004,
        timeout: Optional[int] = 0.1,
        use_pypestworker: Optional[bool] = True,
        par_transform: Optional[Literal["none", "log"]] = "log",
        par_group_settings: Optional[dict[str, dict[str, Any]] | None] = None,
        add_tikhonov_reg: Optional[bool] = False,
        stressmodel_parameterisers: Optional[list | None] = None,
        obs_diff: Optional[bool] = False,
        save_stress_contributions: Optional[bool] = False,
        stress_contribution_groups: Optional[DataFrame | None] = None,
        phi_factors: Optional[dict] = {},
        covary_multimodels_constant_d: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the PEST solver.

        Parameters
        ----------
        exe_name : str | Path
            The name or path to the PEST executable.
        model_ws : str | Path, optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : str | Path, optional
            The template workspace directory for PEST files. Default is "temp".
        master_ws : str | Path, optional
            The master working directory, by default Path("master") unless
            use_pypestworker is True, then master_ws is equal to temp_ws.
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : dict[str, Any] | None, optional
            Control data for the PEST solver. Default is None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
        nfev : int | None, optional
            The number of function evaluations. Default is None.
        long_names : bool, optional
            Whether to use long names in the PEST control file. Default is True.
        port_number : int, optional
            The port number for communication. Default is 4004.
        timeout : float, optional
            Timeout in seconds for PyPestWorker sockets.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        par_transform : Literal["none","log"], optional
            PEST parameter transformation. Default is "log".
        par_group_settings : dict[str, dict[str, Any]]
            Parameter group settings. Outer dict keyed by pargp. Inner dict keys
            are pest setting keywords. Inner dict values are pest parameter group
            values for the given keyword. Default is None.
        add_tikhonov_reg : bool, optional
            Whether to apply preferred-value regularisation in the pest control file.
            Default is False.
        stressmodel_parameterisers : list[pastas_plugins.pest.parameterisers.BaseParameteriser] | None, optional
            Parameteriser objects for StressModel(s), defining how to parameterise each StressModel via PEST.
            Default is None.
        obs_diff : bool, optional
            Option to calibrate to head differences from previous head. Default is False.
        save_stress_contributions : bool, optional
            Whether to save stressmodel contributions to pest obs_data file (zero-weighted).
        stress_contribution_groups: DataFrame, optional
            Series multi-indexed by: [Pastas model name, stressmodel name, label string].
            Columns ("istress_names", "save_all") contain stress names for which stress contributions are summed
            for each model/stressmodel/label key.
            Column "save_all" is a flag for each model - for each model (index 0 model name set),
            if any of these are True, then all stress contributions are saved for this model
             (which can be big), not just the identified istress_names/groups. Default is None.
        phi_factors : dict, optional
            Dict keyed by obs group (obgnme) tag, with values being the factor of Phi desired for that obs group
            via weighting the prior. Default is an empty dict (no phi factors applied in pest). NOTE: This is
            not yet supported for GLM/HP - only IES.
        **kwargs : dict
            Additional keyword arguments passed to the BaseSolver.

        Returns
        -------
        None
        """
        """ TODO (MAYBE) 
        covary_multimodels_constant_d : bool, optional
            Whether to apply Pastas constant_d parameter covariance between models in solver.models.
            Covariance calculated by distance, using model.oseries.metadata "x" and "y" coord keys.
            Variance (along the diagonal) is as calculated internally for parameters regardless of this option
            (via stdev of parbounds / 4, i.e.,  an assumed 95% CI).
        """
        def __getstate__(self):
            # Exclude the logger and its handlers from the state to be pickled
            state = self.__dict__.copy()
            if "logger" in state:
                del state["logger"]
            return state

        def __setstate__(self, state):
            # Reconstruct the logger after unpickling
            self.__dict__.update(state)
            logger = logging.getLogger(__name__)

        BaseSolver.__init__(self, pcov=pcov, nfev=nfev, **kwargs)
        self.long_names = long_names
        # model workspace (for pastas files)
        self.model_ws = Path(model_ws).resolve()
        if not self.model_ws.exists():
            self.model_ws.mkdir(parents=True)
        # template workspace (for pest files)
        self.temp_ws = Path(temp_ws).resolve()
        master_ws = Path(master_ws).resolve()
        self.master_ws = temp_ws if use_pypestworker else master_ws # not sure why we do this...why would ppw runs differ? Isn't that just unnecessarily confusing?
        self.reuse_master = use_pypestworker # depending on if i change the logic as above line, this could be removed...
        # If a user specs the same pest worker template folder as is spec'd for the master_ws, then we do not want to
        # delete master in pyemu os_utils (pyemu would crap out). So reuse_master is True in these cases.
        if self.master_ws == self.temp_ws:
            self.reuse_master = True

        self.exe_name = Path(exe_name)  # pest executable
        self.pf = pyemu.utils.PstFrom(
            original_d=self.model_ws,
            new_d=self.temp_ws,
            remove_existing=True,
            longnames=long_names,
        )
        copy_file(self.exe_name, self.temp_ws)  # copy pest executable
        self.noptmax: int = noptmax
        self.control_data: dict[str, Any] = control_data
        self.port_number = port_number
        self.timeout = timeout
        self.use_pypestworker: bool = use_pypestworker
        self.run_function: Callable = run
        self.ppw_function: Callable = run_pypestworker
        self.par_transform: Literal["none", "log"] = par_transform
        self.par_group_settings: dict[str, dict[str, Any]] = par_group_settings
        self.add_tikhonov_reg: bool = add_tikhonov_reg
        self.stressmodel_parameterisers: list | None = stressmodel_parameterisers
        self.obs_diff: bool = obs_diff
        self.save_stress_contributions = save_stress_contributions
        self.stress_contribution_groups = None
        if self.save_stress_contributions:
            self.stress_contribution_groups = stress_contribution_groups
        self.phi_factors: dict = phi_factors
        # TODO MAYBE self.covary_multimodels_constant_d: bool = covary_multimodels_constant_d

        self.models = {} # dict of models {model.name: model} to be solved by pest simultaneously
        self.vary_by_model = {} # pastas par vary bools for each model
        self.pcovs = {}  # dict of pcovs {model.name: pcov} for each model to be solved by pest simultaneously

        self.stress_obs = None

    @property
    def stress_contribution_groups(self) -> DataFrame | None:
        return self._stress_contribution_groups

    @stress_contribution_groups.setter
    def stress_contribution_groups(
            self, stress_contribution_groups
    ) -> None:
        if stress_contribution_groups is not None:
            self._stress_contribution_groups = stress_contribution_groups.sort_index(level=[0, 1, 2])
            self._stress_contribution_groups.index = self._stress_contribution_groups.index.set_levels(
                self._stress_contribution_groups.index.levels[2].str.replace(" ", ""),
                level=2
            ) # we need spaces removed because pyemu will do this too,
            #   and we need to be able to link between pyemu names and these user-provided labels
            self._stress_contribution_groups.index.names = ['ml_name', 'sm_name','label']
            if not set(["istress_names","save_all"]).issubset(set(self._stress_contribution_groups.columns)):
                raise Exception(f"stress_contribution_groups must contain columns: ['istress_names','save_all']\n" +
                                "but it contains only {stress_contribution_groups.columns}")
            self._stress_contribution_groups.save_all = self._stress_contribution_groups.save_all.astype(bool)
        else:
            self._stress_contribution_groups = None

    def add_model(
            self,
            model,
            pcov: Optional[DataFrame | None] = None,
    ):
        """
        Add model to the PEST solver.

        Parameters
        ----------
        model : pastas.Model
            Pastas model to add to the PEST solver.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
            Eg: par cov from a pre-PEST least squares solve in pastas.
        """
        self.remove_model(model)
        self.models[model.name] = model
        logger.info(f"Model: {model.name} added to solver.models")
        if pcov is not None:
            self.pcovs[model.name] = pcov
            logger.info(f"Model pcov: pcov for {model.name} added to solver.pcovs")

    def _remove_pcov(self, model):
        try:
            self.pcovs.pop(model.name)
            logger.info(f"pcov: {model.name} removed from solver.pcovs")
        except KeyError as e:
            logger.info(f"pcov: {model.name} not in solver.pcovs")

    def remove_model(self, model):
        try:
            self.models.pop(model.name)
            logger.info(f"Model: {model.name} removed from solver.models")
        except KeyError as e:
            logger.info(f"Model: {model.name} not in solver.models")
        self._remove_pcov(model)

    @property
    def stressmodel_parameterisers(self) -> list:
        """Returns list of stressmodel_parameterisers"""
        return self._stressmodel_parameterisers

    @stressmodel_parameterisers.setter
    def stressmodel_parameterisers(self, stressmodel_parameterisers) -> None:
        """Set stressmodel_parameterisers"""
        self._stressmodel_parameterisers = (
            [] if not stressmodel_parameterisers else stressmodel_parameterisers
        )

    @staticmethod
    def _get_uncfile_str(
        cov: pyemu.Cov,
        covmat_file: str | None = None,
        var_mult: float = 1.0,
        include_path: bool = False,
    ) -> str:
        """Get PEST unc file content string for provided pyemu.Cov matrix"""
        cov.to_uncfile(
            "temp.unc",
            covmat_file=covmat_file,
            var_mult=var_mult,
            include_path=include_path,
        )
        with open("temp.unc", "r") as unc:
            unc_str = unc.read()
        Path("temp.unc").unlink()
        return unc_str

    @staticmethod
    def _get_obs_diff(observations: DataFrame) -> DataFrame:
        """Returns difference from previous head obs Series"""
        mask = observations.obs_type=="head"
        diffs = observations.loc[mask].copy()
        diffs.loc[:, "Observations"] = diffs.Observations - diffs.Observations.shift().values
        diffs.dropna(subset=["Observations"], inplace=True)
        diffs.loc[:,"obs_type"] = "headdiff"
        diffs.loc[:, "weight"] = 1.0
        high_mask = (diffs.Observations.abs() >= diffs.Observations.abs().quantile(0.75))
        diffs.loc[~high_mask, "weight"] = 2.0
        return diffs

    @staticmethod
    def _setup_base_obs(
            data: Series,
            obs_type: str = "head",
            weight: float = 1.0,
            series_name: str = "Observations",
            other_col_data: dict | None = None,
    ) -> DataFrame:
        """convert Pastas sim-type Series to DataFrame, and add obs_type and weight fields"""
        data.name = series_name
        data.index.name = "date"
        data = data.to_frame()
        data["obs_type"] = obs_type
        data["weight"] = weight
        if other_col_data:
            data = data.assign(**other_col_data)
        return data

    def _get_stressmodel_contributions(self):
        """
        Returns a DataFrame with all stressmodel contributions specified through stress_contribution_groups parameter
        """
        # stress contributions
        self.sm_contribs = DataFrame()
        sm_contribs_list = []
        self.stress_contribution_groups.to_csv(self.model_ws / "stress_contribution_groups.csv")
        copy_file(self.model_ws / "stress_contribution_groups.csv", self.temp_ws)
        for fp in Path(self.model_ws).glob("simulation_stress_contributions_*.csv"):
            fp.unlink(missing_ok=True)
        for ml_name, ml in self.models.items():
            # get all stress contributions for each model at a minimum.
            contribs_all = ml.get_contributions(
                split=True,
                #tmin=ml.get_tmin(tmin=None, use_oseries=False, use_stresses=True),
                #tmax=ml.get_tmax(tmax=None, use_oseries=False, use_stresses=True),
            )  # all contributions
            contribs_all = [s.resample("ME").mean() for s in
                            contribs_all]  # downsample from daily. Should make this an option...
            contribs_all = pd.concat(contribs_all, axis=1, ignore_index=False)
            # if stress_contribution_groups are user-provided, sum those stress contributions up too.
            ml_stress_groups = self.stress_contribution_groups.xs(ml_name)
            for sm_name, istress_groups in ml_stress_groups.groupby(level="sm_name"):
                istress_groups = ml_stress_groups.xs(sm_name)
                for label, istress_names in istress_groups.groupby(level=0):
                    names = istress_groups.xs(label)[["istress_names"]].values.flatten()
                    # aggregate selected groups of istress contributions
                    contribs_all.loc[:,label] = contribs_all.loc[:, names].sum(axis=1)
            # drop unspecific istress_names / labels from the df
            if (self.stress_contribution_groups.xs(ml_name).save_all == False).any():
                contribs_all = contribs_all.loc[:,
                contribs_all.columns.isin(ml_stress_groups.index.get_level_values("label"))
                ]
            # melt from xtab to flat array and save
            contribs_all.index.name = "date"
            contribs_all = contribs_all.reset_index(drop=False).melt(
                id_vars="date",
                value_vars=contribs_all.columns,
                var_name="column_names",
                value_name="Observations",
            ).set_index(["column_names","date"])
            contribs_file = Path(self.model_ws / f"simulation_stress_contributions_{ml_name}.csv")
            contribs_all.to_csv(contribs_file, date_format="%d/%m/%Y")
            copy_file(contribs_file, self.temp_ws)
            contribs_all.loc[:, "model_name"] = ml_name
            contribs_all.loc[:, "obs_type"] = "stress_contribution"
            contribs_all.loc[:, "weight"] = 0.0
            sm_contribs_list.append(contribs_all)
        # merge all models' data together
        sm_contribs = pd.concat(sm_contribs_list, ignore_index=False)
        return sm_contribs

    def setup_model(self):
        """Setup and export Pastas model for PEST optimization"""
        if self.models == {}:
            self.models[self.ml.name] = self.ml
            if self.pcov is not None:
                self.pcovs[self.ml.name] = self.pcov
        # observations
        obs_list, obs_diffs_list, stress_obs_list, headsmp_obs_list = [], [], [], []
        for ml_idx, (ml_name, ml) in enumerate(self.models.items()):
            ml.oseries.metadata.update({"ml_code": f"m{str(ml_idx).zfill(2)}"})
            # heads
            observations = PestSolver._setup_base_obs(
                ml.observations(),
                other_col_data={"model_name": ml_name},
            )
            # head differences from previous
            if self.obs_diff:
                obs_diffs = self._get_obs_diff(observations)
                obs_diff_file = self.model_ws / f"simulation_head_diffs_{ml_name}.csv"
                obs_diffs.Observations.to_csv(obs_diff_file, date_format="%d/%m/%Y")
                copy_file(obs_diff_file, self.temp_ws)
                obs_diffs_list.append(obs_diffs.copy())
            obs_file = self.model_ws / f"simulation_{ml_name}.csv"
            observations.Observations.to_csv(obs_file, date_format="%d/%m/%Y")
            copy_file(obs_file, self.temp_ws)
            obs_list.append(observations.copy())

            # smp style zero-weight head obs of full timeseries. For plotting ensemble hydrographs from pestpp-ies stack.
            # just monthly mean to avoid crazy big files; TODO: could make headsmp resampling an option for short sims.
            headsmp_obs = PestSolver._setup_base_obs(
                ml.simulate(
                    #tmin=ml.get_tmin(tmin=None, use_oseries=False, use_stresses=True),
                    #tmax=ml.get_tmax(tmax=None, use_oseries=False, use_stresses=True),
                ).resample("ME").mean(),
                obs_type="headsmp",
                weight=0.0,
                other_col_data={"model_name": ml_name},
            )
            headsmp_obs.index.name = "date"
            headsmp_obs_file = self.model_ws / f"simulation_{ml_name}.smp.csv"
            headsmp_obs.Observations.to_csv(headsmp_obs_file, date_format="%d/%m/%Y")
            copy_file(headsmp_obs_file, self.temp_ws)
            headsmp_obs_list.append(headsmp_obs.copy())

        # stress obs
        for sm_p in self.stressmodel_parameterisers:
            if sm_p.obs_data is not None:
                obs_stress_file = self.model_ws / f"{sm_p.stressmodel_name}.stress_obs.csv"
                sm_p.obs_data.to_csv(obs_stress_file, date_format=sm_p.date_format)
                copy_file(obs_stress_file, self.temp_ws)
                sm_p.obs_data.name = "Observations"
                sm_p_obs = sm_p.obs_data.to_frame()
                sm_p_obs.loc[:, "model_name"] = ml_name
                sm_p_obs.loc[:, "obs_type"] = "stress_obs"
                sm_p_obs.loc[:, "weight"] = 1.0
                stress_obs_list.append(sm_p_obs)

        self.stress_obs = pd.concat(
            stress_obs_list,
            ignore_index=False)  # can't concat this with self.observations because index differs

        self.observations = pd.concat(obs_list, ignore_index=False)
        self.observations.index.name = "date"

        if self.obs_diff:
            self.obs_diffs = pd.concat(obs_diffs_list, ignore_index=False)
            self.obs_diffs.index.name = "date"
            ml_obs_list = [] # we do this to re-order as pst_from.add_observations is called to build the pst file
            for ml_name, ml in self.models.items():
                ml_obs_list.append(
                    pd.concat(
                        [self.observations.loc[self.observations.model_name == ml_name],
                        self.obs_diffs.loc[self.obs_diffs.model_name == ml_name]],
                        ignore_index=False)
                )
            self.observations = pd.concat(ml_obs_list, ignore_index=False)

        # smp style zero-weight head obs of full timeseries
        headsmp_obs = pd.concat(headsmp_obs_list, ignore_index=False)
        self.observations = pd.concat([self.observations, headsmp_obs], ignore_index=False)

        # stressmodel contributions
        # ensure we remove stress_contribution_groups.csv so forward_run can use its existence
        # to define bool save_stress_contributions (non pypestworker runs)
        Path(self.model_ws / "stress_contribution_groups.csv").unlink(missing_ok=True)
        self.sm_contribs = pd.DataFrame()
        if self.save_stress_contributions:
            self.sm_contribs = self._get_stressmodel_contributions()
        self.stress_obs = pd.concat([self.stress_obs, self.sm_contribs], ignore_index=False)

        # setup parameters
        pars_list = []
        self.vary = []
        for ml_idx, (ml_name, ml) in enumerate(self.models.items()):
            ml.parameters.loc[:, "optimal"] = ml.parameters.loc[:, "initial"]
            self.vary_by_model[ml_name] = list(ml.parameters.vary.values.astype(bool))
            self.vary += self.vary_by_model[ml_name]
            parameters = ml.parameters[self.vary_by_model[ml_name]].copy()
            parameters.index = [
                p.replace("_A", "_g") if p.endswith("_A") else p for p in parameters.index
            ]
            parameters.index.name = "parnames"
            if "constant_d" in parameters.index:
                heads_mask = (self.observations.model_name == ml_name)
                heads_mask = heads_mask & (self.observations.obs_type == "head")
                observations = self.observations.Observations.loc[heads_mask]
                if np.isnan(parameters.at["constant_d", "pmin"]):
                    ml.set_parameter(
                        "constant_d",
                        pmin=np.min(observations.values) - np.std(observations.values),
                    )
                if np.isnan(parameters.at["constant_d", "pmax"]):
                    ml.set_parameter(
                        "constant_d",
                        pmax=np.max(observations.values) + np.std(observations.values),
                    )
            else:
                ml.settings["fit_constant"] = False
            parameters["model_name"] = ml_name
            # define pmin/pmax from pastas model for pst in setup_files below
            parameters["pmin"] = ml.parameters.loc[
                self.vary_by_model[ml_name], "pmin"
            ].values
            parameters["pmax"] = ml.parameters.loc[
                self.vary_by_model[ml_name], "pmax"
            ].values
            ml_code = ml.oseries.metadata["ml_code"]
            parameters.index = [f"{ml_code}{p}" for p in parameters.index]
            pars_list.append(parameters.copy())
        parameters = pd.concat(pars_list, ignore_index=False)
        parameters.index.name = "parnames"
        if parameters.index.str.rsplit("_").str[0].str.isupper().any():
            logger.error(
                "pestpp is case insensitive so any capitalized parameters (stress model names) can cause issues in the solver."
            )
        par_sel = parameters.loc[:, ["optimal"]]
        par_sel.to_csv(self.model_ws / "parameters_sel.csv")
        copy_file(self.model_ws / "parameters_sel.csv", self.temp_ws)
        self.par_sel = par_sel
        self.parameters = parameters

        # model
        for dir in (self.model_ws, self.temp_ws, self.master_ws):
            for p in Path(dir).glob("*.pas"):
                p.unlink()
        for ml_name, ml in self.models.items():
            ml_code = ml.oseries.metadata["ml_code"]
            ml_file = self.model_ws / f"model_{ml_code}.pas"
            self.models[ml_name].to_file(ml_file)
            copy_file(ml_file, self.temp_ws)

    def write_pst(self, pst: pyemu.Pst, version: int = 2) -> None:
        """Write pest control file

        Parameters
        ----------
        pst : pyemu.Pst
            Pyemu pest control file object.
        version : int, optional
            Version of the control file, by default version 2
        """
        pst.write(self.pf.new_d / "pest.pst", version=version)

    def _update_stress_obs_names(
            self,
            ml_name: str,
            obs_types: list,
            date_format="%d/%m/%Y",
            rsplit_column_name=True,
    ) -> None:
        """
        Add pest obsnme and obgnme to self.stress_obs dataframe. Designed to be called
        immediately after each self.pf.add_observations() call.
        """
        tmp_obs = self.pf.obs_dfs[-1].assign(
            date=self.pf.obs_dfs[-1].obsnme.apply(
                lambda x: pd.to_datetime(x.rsplit("_date:")[-1], format=date_format)),
            column_names=self.pf.obs_dfs[-1].obsnme.apply(
                lambda x: x.rsplit("_date:")[0].rsplit("_column_names:", 1)[-1]
            )
        )
        if rsplit_column_name:
            tmp_obs["column_names"] = tmp_obs.column_names.apply(
                lambda x: f"{x.rsplit('_', 1)[0]}_{x.rsplit('_', 1)[-1]}"
            ) # these are lower case because pest obsnme is (from which column_names is derived - above)
        omask = (self.stress_obs.model_name == ml_name) & \
                (self.stress_obs.obs_type.isin(obs_types))
        join_obs = self.stress_obs.loc[omask].copy()
        og_index0 = join_obs.index.levels[0]
        join_obs.index = join_obs.index.set_levels(
            join_obs.index.levels[0].str.lower(),
            level="column_names"
        ) # because pest obsnme-derived column_names is lower case
        try:
            join_obs = join_obs.drop(columns=["obsnme","obgnme"])
        except: # if the cols don't exist (on first use of this function) we get an exception
            pass
        join_obs = join_obs.join(
            tmp_obs[["column_names", "date", "obsnme", "obgnme"]].set_index(["column_names", "date"]), how="left"
        )
        join_obs.index = join_obs.index.set_levels(
            og_index0, level="column_names",
        ) # revert index column_names to og case
        self.stress_obs.loc[omask, ["obsnme","obgnme"]] = join_obs.loc[:,["obsnme","obgnme"]]
        self.pf.obs_dfs[-1].loc[:,"weight"] = self.stress_obs.loc[omask].set_index("obsnme").weight

    def _update_observations_names(
            self,
            ml_name: str,
            obs_types: list,
            date_format="%d/%m/%Y",
    ) -> None:
        """
        Add pest obsnme and obgnme to self.observations dataframe. Designed to be called
        immediately after each self.pf.add_observations() call.
        """
        tmp_obs = self.pf.obs_dfs[-1].assign(
            date=self.pf.obs_dfs[-1].obsnme.apply(
                lambda x: pd.to_datetime(x.rsplit("_date:")[-1], format=date_format))
        )
        omask = (self.observations.model_name == ml_name) & \
                (self.observations.obs_type.isin(obs_types))
        join_obs = self.observations.loc[omask]
        try:
            join_obs = join_obs.drop(columns=["obsnme","obgnme"])
        except: # if the cols don't exist (on first use of this function) we get an exception
            pass
        self.observations.loc[omask, ["obsnme","obgnme"]] = join_obs.join(
            tmp_obs[["date", "obsnme", "obgnme"]].set_index("date"), how="left"
        ).loc[:,["obsnme","obgnme"]] # joining by date works because this is used for head obs, and we do this per bore (per pastas.simulation.csv file)
        self.pf.obs_dfs[-1].loc[:,"weight"] = self.observations.loc[omask].set_index("obsnme").weight
        tmp_obs = None

    def setup_files(self, version: int = 2):
        """Setup PEST file structure for optimization

        Parameters
        ----------
        version : int, optional
            Version of the control file, by default version 2
        """

        # standard pastas model parameters
        pf_pars = self.pf.add_parameters(
            self.model_ws / "parameters_sel.csv",
            index_cols=[self.par_sel.index.name],
            use_cols=self.par_sel.columns.to_list(),
            par_type="grid",
            par_style="direct",
            transform=self.par_transform,
            # pargp=self.par_sel.columns.to_list(),
            # par_name_base=self.par_sel.columns.to_list(), #[x.split("_")[0] for x in self.par_sel.columns],
            # lower_bound=self.ml.parameters.loc[self.vary, "pmin"].values.tolist(),
            # upper_bound=self.ml.parameters.loc[self.vary, "pmax"].values.tolist(),
            # ult_lbound = self.ml.parameters.loc[self.vary, ["pmin"]].transpose().values.tolist(),
            # ult_ubound = self.ml.parameters.loc[self.vary, ["pmax"]].transpose().values.tolist(),
        )
        pastas_ml_pars = pf_pars.index

        # save pastas.model parameter and observation index for going back and forth between pastas and pest names
        self.parameter_index = dict(
            zip(pf_pars.index, self.par_sel.index) #self.ml.parameters[self.vary].index)
        )
        
        # and for translating from pastas model parameter names to pest names
        self.ml_parname_to_pst = dict(
            zip(self.parameter_index.values(), self.parameter_index.keys())
        ) # to be updated below with stressmodel_parameteriser pars

        # add custom stressmodel parameters
        if self.stressmodel_parameterisers:
            for fp in Path(self.temp_ws).glob("*.parameteriser.pkl.gz"):
                Path(fp).unlink(missing_ok=True)
            for sm_p in self.stressmodel_parameterisers:
                sm_p.solver = self
                sm_p.add_stress_parameters(par_name_base="sm")
                # pickle to disk for pest non-pypestworker workers
                fname = self.temp_ws / f"{sm_p.stressmodel.name}.parameteriser.pkl.gz"
                with gzip.open(fname, "wb") as f:
                    dill.dump(sm_p, f)  # pickle
                # add new parnmes to indexers (although there is no translation here, keys/values are same, but we need them to simplify later code in forward_run)
                parnmes = sm_p.source_points.parnme.values
                self.parameter_index.update(
                    dict(
                        zip(
                            parnmes, #sm_p.stress_pars.index.to_frame().iloc[:, 0],
                            sm_p.stress_pars.index.to_frame().iloc[:, 0],
                        )
                    )
                )
                self.ml_parname_to_pst.update(
                    dict(
                        zip(
                            sm_p.stress_pars.index.to_frame().iloc[:, 0],
                            parnmes, #sm_p.stress_pars.index.to_frame().iloc[:, 0],
                        )
                    )
                )

        # observations
        for ml_name, ml in self.models.items():
            # usual pastas head obs
            obsgp = f"head_{ml_name}"
            self.pf.add_observations(
                f"simulation_{ml_name}.csv",
                index_cols=[self.observations.index.name],
                use_cols=["Observations"],
                obsgp=obsgp,
            )
            # add pest obsnme and obgnme to self.observations for this last set of obs added to pst
            self._update_observations_names(
                ml_name=ml_name,
                obs_types=["head"],
            )

            # head diffs from previous if requested
            if self.obs_diff:
                obsgp = f"headdiff_{ml_name}"
                self.pf.add_observations(
                    f"simulation_head_diffs_{ml_name}.csv",
                    index_cols=[self.obs_diffs.index.name],
                    use_cols=["Observations"],
                    obsgp=obsgp,
                )
                # add pest obsnme and obgnme to self.observations for this last set of obs added to pst
                self._update_observations_names(
                    ml_name=ml_name,
                    obs_types=["headdiff"],
                )

            # smp-style zero-weight head obs
            # usual pastas head obs
            obsgp = f"headsmp_{ml_name}"
            self.pf.add_observations(
                f"simulation_{ml_name}.smp.csv",
                index_cols=[self.observations.index.name],
                use_cols=["Observations"],
                obsgp=obsgp,
            )
            # add pest obsnme and obgnme to self.observations for this last set of obs added to pst
            self._update_observations_names(
                ml_name=ml_name,
                obs_types=["headsmp"],
            )

            # stress contributions
            if self.save_stress_contributions:
                for sm_name, istress_groups in self.stress_contribution_groups.xs(ml_name, level=0).groupby(level=0):
                    obsgp = f"stress_contrib_{ml_name}"
                    self.pf.add_observations(
                        f"simulation_stress_contributions_{ml_name}.csv",
                        index_cols=["column_names","date"],
                        use_cols=["Observations"],
                        obsgp=obsgp,
                    )
                    # add pest obsnme and obgnme to self.observations for this last set of obs added to pst
                    self._update_stress_obs_names(
                        ml_name=ml_name,
                        obs_types=["stress_contribution"],
                        date_format="%d/%m/%Y",
                        rsplit_column_name=False,
                    )

        # stress obs
        for sm_p in self.stressmodel_parameterisers:
            if sm_p.obs_data is not None:
                obsgp = f"stress_{sm_p.stressmodel_name}"
                self.pf.add_observations(
                    f"{sm_p.stressmodel_name}.stress_obs.csv",
                    index_cols=sm_p.obs_data.index.names,
                    use_cols=["Observations"],
                    obsgp=obsgp,
                )
                # add pest obsnme and obgnme to self.stress_obs for this last set of obs added to pst
                self._update_stress_obs_names(
                    ml_name=ml_name,
                    obs_types=["stress_obs"],
                    date_format=sm_p.date_format,
                )

        # python scripts to run
        self.pf.add_py_function(self.run_function, "run()", is_pre_cmd=None)
        self.pf.mod_py_cmds.append("run()")

        # create control file
        pst = self.pf.build_pst(self.pf.new_d / "pest.pst", version=version)
        if "longname" not in pst.parameter_data.columns:
            pst.parameter_data["longname"] = pst.parameter_data.index.values

        # define factored obs weights if requested
        if isinstance(self, PestIesSolver) and self.phi_factors != {}:
            # ies phi factor file
            phi_factor_file = str(self.pf.new_d / "pest.phi_factors.csv")
            pd.DataFrame.from_dict(self.phi_factors, orient="index").to_csv(
                phi_factor_file, header=None
            )
            pst.pestpp_options.update({"ies_phi_factor_file": Path(phi_factor_file).name})
        elif self.phi_factors != {}:
            # TODO: manually edit weights based on initial simulation residuals (pest_hp / glm cases)
            logger.warning("Phi factor prior weighting for PEST-HP / PESTPP-GLM is not yet supported.")
            pass

        # pastas model parameter bounds
        pastas_pars_mask = pst.parameter_data.longname.isin(pastas_ml_pars.values)
        pst.parameter_data.loc[pastas_pars_mask, ["parlbnd"]] = self.parameters.loc[
            self.vary, "pmin"
        ].values
        pst.parameter_data.loc[pastas_pars_mask, ["parubnd"]] = self.parameters.loc[
            self.vary, "pmax"
        ].values

        # update stressmodel parameter bounds
        if self.stressmodel_parameterisers:
            for sm_p in self.stressmodel_parameterisers:
                sm_p.stress_pars = sm_p.stress_pars.reset_index(drop=False).set_index(
                    "index"
                )
                pst.parameter_data.loc[sm_p.stress_pars.index, ["parval1","partrans","parlbnd","parubnd"]] = (
                    sm_p.stress_pars.loc[:,["parval1","partrans","parlbnd","parubnd"]]
                )
                sm_p.stress_pars = sm_p.stress_pars.reset_index(drop=False).set_index(
                    ["column_names", "index_org"]
                )

        # Tie duplicate wellmodel pars to other wellmodels' pars.
        # We want the same params used across stressmodels where the same stress (eg pumping bore) / datetime is used.
        pst.parameter_data["parnme_common_base"] = pst.parameter_data.parnme.str.replace(r"_inst:\d+", "", regex=True)
        pst.parameter_data["inst_first"] = pst.parameter_data.groupby("parnme_common_base").inst.transform("min")
        tied_mask = (pst.parameter_data.inst > pst.parameter_data.inst_first) & (pst.parameter_data.partrans != "fixed")
        source_pars = pst.parameter_data.loc[
            (pst.parameter_data.inst == pst.parameter_data.inst_first)
        ].set_index("parnme_common_base")
        pst.parameter_data.loc[tied_mask, "partied"] = source_pars.loc[
            pst.parameter_data.loc[tied_mask].parnme_common_base
        ].parnme.values
        pst.parameter_data.loc[tied_mask, "partrans"] = "tied"

        # add parval/bound offsets as needed depending on par_transform and zero values at bounds
        pst = PestSolver.add_offsets(pst)

        log_update_mask = pst.parameter_data.partrans == "log"
        pst.parameter_data.loc[log_update_mask, ["parchglim"]] = "factor"
        pst.parameter_data.loc[~log_update_mask, ["parchglim"]] = "relative"
        pst.parameter_data.loc[pastas_pars_mask, ["pargp"]] = (
            self.par_sel.columns.to_list()
        )

        # apply provided parameter group settings (FORCEN, DERINC etc)
        if self.par_group_settings is not None:
            pst.rectify_pgroups()
            for pargp, kw_val_dict in self.par_group_settings.items():
                for pargp_kw, pargp_kw_value in kw_val_dict.items():
                    pst.parameter_groups.loc[pargp, pargp_kw] = pargp_kw_value

        pst.control_data.noptmax = self.noptmax  # optimization runs
        if self.control_data is not None:
            for key, value in self.control_data.items():
                if key == "noptmax":
                    logger.warning(
                        "noptmax is set as an attribute and can't be set using the `control_data` dictionary"
                    )
                else:
                    setattr(pst.control_data, key, value)

        # add tikhonov regularisation
        if self.add_tikhonov_reg:
            pyemu.helpers.zero_order_tikhonov(pst)

        # add preferred difference regularisation equations using the covariance for regularisation weight (only for hp/glm cases)
        if isinstance(self, (PestHpSolver, PestGlmSolver)):
            if self.stressmodel_parameterisers:
                for sm_p in self.stressmodel_parameterisers:
                    pyemu.helpers.first_order_pearson_tikhonov(
                        pst, sm_p.stress_parcov, reset=False
                    )

        # build a list of parcovs for IES
        if isinstance(self, PestIesSolver):
            if self.pcovs != {}:
                unc_str = ""
                for ml_idx, (ml_name, pcov) in enumerate(self.pcovs.items()):
                    new_pnames = [f"m{str(ml_idx).zfill(2)}{p}" for p in pcov.index]
                    pcov.columns = new_pnames
                    pcov.index = new_pnames
                    pastas_parcov = pyemu.Cov(
                        x=pcov.values,
                        names=pcov.columns,
                        isdiagonal=False,
                    )
                    covmat_fname = str(
                        self.temp_ws / f"pest.prior.{ml_name}.pastas_pars.jcb"
                    )
                    pastas_parcov.to_binary(covmat_fname)
                    unc_str += self._get_uncfile_str(
                        pastas_parcov,
                        covmat_file=covmat_fname,
                        var_mult=1.0,
                        # TODO probs need a user variable here, or define internally based on par range for this stressmodel par set
                        include_path=False,
                    )
            else:
                pastas_parcov = pyemu.Cov.from_parameter_data(
                    pst,
                    sigma_range=4.0,
                    scale_offset=False,
                    subset=pastas_ml_pars,  # .to_list(), pyemu doc says str, but has to be a Series/Index
                )  # returns a diagonal matrix
                # I think the wording in pyemu doc is wrong on scale_offset=True by default.
                # Here, parval1 is already scaled and offset...why add those before doing cov calcs?
                # definitely get log par errors. Maybe pyemu does the anti-scale/offset immediately
                # before pst.write (scary!), whereas here we have already done that.
                # I don't think it does though, as i always get par transform errors if I do not anti-scale/offset myself before pst.write.
                unc_str = self._get_uncfile_str(
                    pastas_parcov
                )  # default args are for diagonals
            pastas_parcov = None
            if self.stressmodel_parameterisers:
                covmat_fname = str(
                    self.temp_ws / f"pest.prior.{sm_p.stressmodel_name}.jcb"
                )
                sm_p.stress_parcov.to_binary(covmat_fname)
                for sm_p in self.stressmodel_parameterisers:
                    unc_str += self._get_uncfile_str(
                        sm_p.stress_parcov,
                        covmat_file=covmat_fname,
                        var_mult=1.0,  # TODO probs need a user variable here, or define internally based on par range for this stressmodel par set
                        include_path=False,
                    )
            with open(self.temp_ws / "pest.prior.unc", "w") as unc:
                unc.write(unc_str)

        self.write_pst(pst=pst, version=version)

        with (self.temp_ws / "parameter_index.json").open("w") as f:
            json.dump(obj=self.parameter_index, fp=f, default=str)

        # define an observation index for linking pst obsnme to observation / simulation data
        self.observation_index = pd.DataFrame(
            {
                "obsnme": self.observations.obsnme.values,
                "model_name": self.observations.model_name.values,
                "date": self.observations.index.values,
                "obgnme": self.observations.obgnme.values,
            }
        ).set_index(["model_name","obgnme","date"])
        self.observation_index.to_json(str(self.temp_ws / "observation_index.json"))

        self.stress_obs.to_csv(Path(self.model_ws / "stress_obs.csv"), date_format="%d/%m/%Y")
        copy_file(Path(self.model_ws / "stress_obs.csv"), self.temp_ws)
        self.observations.to_csv(Path(self.model_ws / "observations.csv"), date_format="%d/%m/%Y")
        copy_file(Path(self.model_ws / "observations.csv"), self.temp_ws)

        self.ppw_kwargs = {
            "timeout": self.timeout,  # PyPestWorker socket timeout in seconds
            "models": self.models,  # "ml": self.ml,  # "ml_dict": self.ml.to_dict(),
            "parameter_index": self.parameter_index,
            "observation_index": self.observation_index,
            "stressmodel_parameterisers": self.stressmodel_parameterisers,
            "stress_obs": self.stress_obs,
            "save_stress_contributions": self.save_stress_contributions,
            "stress_contribution_groups": self.stress_contribution_groups,
        }

    def run(self, arg_str: str = "", silent: bool = False):
        pyemu.os_utils.run(
            f"{self.exe_name.name} pest.pst{arg_str}", cwd=self.pf.new_d, verbose=silent
        )

    def initialize(self, version: int = 2) -> None:
        """Initialize the solver by setting up the model and files."""
        if len(self.models) == 0:
            raise ValueError("No Pastas model assigned to the solver.")
        if self.pf.pst is None:
            self.setup_model()
            self.setup_files(version=version)
        else:
            logger.info("Solver is already initialized.")

    @staticmethod
    def add_offsets(pst) -> pyemu.Pst:
        """
        Add offset for default log transform (where needed - parlbnd <= 0).
        Generally a good idea to log transform.
        Check for 0.0 parubnd for transform==none pars and add an offset so
        derinc can be calc'd by PEST_HP when pars are at 0.0

        Parameters
        ----------
        pst : pyemu.Pst
            Pyemu pest control file object.

        Returns
        -------
        Modified pyemu.Pst with parameter offsets applied
        """
        log_mask = (pst.parameter_data.partrans.str.lower() == "log") & (
            pst.parameter_data.parlbnd <= 0.0
        )
        par_offsets = pst.parameter_data.loc[log_mask].parlbnd - 0.1
        pst.parameter_data.loc[log_mask, ["offset"]] = par_offsets.values
        pst.parameter_data.loc[log_mask, ["parval1", "parlbnd", "parubnd"]] = (
            pst.parameter_data.loc[log_mask, ["parval1", "parlbnd", "parubnd"]]
            .add(par_offsets.abs(), axis=0)
            .values
        )

        # Check for 0.0 parbnd for untransformed pars and add an offset so derinc can be calc'd by PEST_HP when pars are at 0.0
        ubnd0_mask = (pst.parameter_data.partrans.str.lower() == "none") & (
            pst.parameter_data.loc[:, ["parubnd", "parlbnd"]] == 0.0
        ).any(axis=1)
        pst.parameter_data.loc[ubnd0_mask, ["offset"]] = 0.1
        pst.parameter_data.loc[ubnd0_mask, ["parval1", "parlbnd", "parubnd"]] = (
            pst.parameter_data.loc[ubnd0_mask, ["parval1", "parlbnd", "parubnd"]]
            .sub(0.1)
            .values
        )
        return pst

    def posterior_pcov_from_jco(
        self,
        jco_file: str,
        pastas_par_names: bool = True,
        **kwargs,
    ) -> pyemu.Cov:
        """
        Obtain the posterior parameter covariance matrix for pst file corresponding to jco_file

        Parameters
        ----------
        jco_file : str
            Filepath to Jacobian matrix from a PEST calibration exercise.
        pastas_par_names : bool, optional
            Whether to return pastas parameter names as row/col indices, or leave PEST par names as is (False). Default is True.
        **kwargs : dict
            Additional keyword arguments passed to pyemu.Schur

        Returns
        -------
        post_pcov : pyemu.Cov
            Posterior parameter ensemble for pst file corresponding to jco_file
        """
        if "scale_offset" not in kwargs.keys():
            # The prior must be constructed from offset par space for log-transformed pars, otherwise we will get nans in the prior covmat.
            # Because pyemu.Schur's prior is constructed from par bounds (unless the prior parcov is user-provided),
            # PestSolver is likely to have applied par offsets for log transformed pars and/or zero value parbnds.
            kwargs["scale_offset"] = False
        schur = pyemu.Schur(jco=jco_file, **kwargs)
        post_pcov = schur.posterior_parameter.df()
        if pastas_par_names:
            post_pcov.rename(
                index=self.parameter_index, columns=self.parameter_index, inplace=True
            )
        if self.par_transform == "log":
            logger.warning(
                'Posterior PestSolver.pcov is based on log-transformed parameter space, because PestSolver.par_transform is "log"'
            )
        return post_pcov


class PestGlmSolver(PestSolver):
    """PESTPP-GLM (Gauss-Levenberg-Marquardt) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pestpp-glm",
        model_ws: str | Path = Path("pastas_files"),
        temp_ws: str | Path = Path("pest"),
        master_ws: str | Path = Path("pest"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-GLM solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name or path to the PESTPP-GLM executable. Default is "pestpp-glm".
        model_ws : str | Path, optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : str | Path, optional
            The template workspace directory for PEST files. Default is "temp".
        master_ws : str | Path, optional
            The master working directory, by default Path("master") unless
            use_pypestworker is True, then master_ws is equal to temp_ws.
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : dict[str, Any] | None, optional
            Control data for the PEST solver. Default is None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
        nfev : int | None, optional
            The number of function evaluations. Default is None.
        port_number : int, optional
            The port number for communication. Default is 4004.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the PestSolver.

        Returns
        -------
        None
        """
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            master_ws=master_ws,
            noptmax=noptmax,
            control_data=control_data,
            pcov=pcov,
            nfev=nfev,
            port_number=port_number,
            use_pypestworker=use_pypestworker,
            long_names=True,
            **kwargs,
        )

    def solve(self, **kwargs) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """
        Solves the optimization problem using the pestpp-glm solver.
        This method sets up the model and necessary files, runs the solver, and
        retrieves the optimal parameters, their covariance, and the objective
        function value.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        success : bool
            Indicates whether the solver ran successfully.
        optimal : NDArray[np.float64]
            The optimal parameters obtained from the solver.
        stderr : NDArray[np.float64]
            The standard errors of the optimal parameters.
        """

        self.initialize(version=2)
        if self.use_pypestworker:
            pyemu.os_utils.start_workers(
                worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
                exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
                pst_rel_path="pest.pst",  # the control file to use with PEST
                num_workers=1,  # how many agents to deploy
                port=self.port_number,  # the port to use for communication
                worker_root=self.temp_ws.parent,  # where to deploy the agent directories; relative to where python is running
                master_dir=self.temp_ws,  # the manager directory
                reuse_master=self.reuse_master,
                ppw_function=self.ppw_function
                if self.use_pypestworker
                else None,  # the function to run in the agent
                ppw_kwargs=self.ppw_kwargs
                if self.use_pypestworker
                else {},  # the arguments to pass to the ppw_function
            )
        else:
            self.run()

        # optimal parameters
        ipar = pd.read_csv(self.temp_ws / "pest.ipar", index_col=0).transpose()
        ipar.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        self.nfev = ipar.columns[-1]
        optimal[self.vary] = ipar.loc[:, self.nfev].values

        # covariance
        pcov = pd.read_csv(
            self.temp_ws / f"pest.{self.nfev}.post.cov",
            sep=r"\s+",
            skiprows=[0],
            nrows=len(ipar.index),
            header=None,
        )
        pcov.index = ipar.index
        pcov.columns = ipar.index
        self.pcov = pcov
        stderr = np.full(len(optimal), np.nan)
        stderr[self.vary] = np.sqrt(np.diag(self.pcov.values))

        # objective function value (phi)
        iobj = pd.read_csv(self.temp_ws / "pest.iobj", index_col=0)
        self.obj_func = iobj.at[self.nfev, "total_phi"]
        success = True  # always :)
        return success, optimal, stderr


class PestHpSolver(PestSolver):
    """PEST_HP (highly parallelized) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pest_hp",
        exe_agent: str | Path = "agent_hp",
        model_ws: str | Path = Path("pastas_files"),
        temp_ws: str | Path = Path("pest"),
        master_ws: str | Path = Path("pest"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        num_workers: int | None = None,
        #TODO: use_pypestworker: bool = False, # need pest_hp version of pypestworker for this. TCP messaging differs.
        **kwargs,
    ) -> None:
        """
        Initialize the PEST_HP solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name or path to the PEST_HP executable. Default is "pest_hp".
        exe_agent : str | Path, optional
            The name or path to the agent_HP executable. Default is "agent_hp".
        model_ws : str | Path, optional
            The model workspace directory for Pastas files. Default is "model".
        temp_ws : str | Path, optional
            The template workspace directory for PEST files. Default is "temp".
        master_ws : str | Path, optional
            The master working directory, by default Path("master").
        noptmax : int, optional
            The maximum number of optimization iterations. Default is 0.
        control_data : dict[str, Any] | None, optional
            Control data for the PEST solver. Default is None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix. Default is None.
        nfev : int | None, optional
            The number of function evaluations. Default is None.
        port_number : int, optional
            The port number for communication. Default is 4004.
        num_workers : int | None, optional
            The number of worker processes, by default the number of physical CPU cores.
        **kwargs : dict
            Additional keyword arguments passed to the PestSolver.

        Returns
        -------
        None
        """
        #TODO:        use_pypestworker : bool, optional
        #    Whether to use the PyPestWorker for Python processing. Default is True.
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            master_ws=master_ws,
            pcov=pcov,
            nfev=nfev,
            long_names=False,
            noptmax=noptmax,
            control_data=control_data,
            port_number=port_number,
            use_pypestworker=False, # TODO: allow pypestworker with pest_hp. Need pest_hp version of pypestworker for this. TCP messaging differs.
            **kwargs,
        )
        self.exe_agent = Path(exe_agent)
        self.computername = get_computername()
        copy_file(self.exe_agent, self.temp_ws)  # copy agent executable
        self.num_workers = (
            cpu_count(logical=False) if num_workers is None else num_workers
        )

    def solve(
        self, silent: bool = False, **kwargs
    ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve the optimization problem using the pest_hp solver.

        This method sets up the model and necessary files, runs the solver, and
        retrieves the optimal parameters and the objective function value.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for the solver.

        Returns
        -------
        success : bool
            Indicates whether the solver ran successfully.
        optimal : NDArray[np.float64]
            The optimal parameters obtained from the solver.
        stderr : NDArray[np.float64]
            The standard errors of the optimal parameters.
        """
        self.initialize(version=1)
        pyemu.os_utils.start_workers(
            worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
            exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
            pst_rel_path="pest.pst",  # the control file to use with PEST
            num_workers=self.num_workers,  # how many agents to deploy
            worker_root=self.master_ws.parent,  # where to deploy the agent directories; relative to where python is running
            master_dir=self.master_ws,  # the manager directory
            port=self.port_number,  # the port to use for communication
            verbose=silent,
            silent_master=silent,
            reuse_master=self.reuse_master,
            ppw_function=self.ppw_function
            if self.use_pypestworker
            else None,  # the function to run in the agent
            ppw_kwargs=self.ppw_kwargs
            if self.use_pypestworker
            else {},  # the arguments to pass to the ppw_function
            cleanup=False,
        )

        par = pd.read_csv(
            self.master_ws / "pest.par",
            index_col=0,
            sep=r"\s+",
            skiprows=[0],
            header=None,
        )
        par.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        # load par * scale + offset --> pastas model par space
        optimal[self.vary] = (
            par.iloc[:, 0].values * par.iloc[:, 1].values + par.iloc[:, 2].values
        )

        ofr = pd.read_csv(
            self.master_ws / "pest.ofr", index_col=0, sep=r"\s+", skiprows=2
        )
        self.nfev = ofr.index[-1]
        self.obj_func = ofr.at[self.nfev, "total"]

        # get posterior par cov
        self.pcov = self.posterior_pcov_from_jco(str(self.master_ws / "pest.jco"))

        # TODO: Obtain stderror from pest.hp output
        stderr = np.full_like(optimal, np.nan)
        return True, optimal, stderr


class PestIesSolver(PestSolver):
    """PESTPP-IES (Iterative Ensemble Smoother) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pestpp-ies",
        model_ws: str | Path = Path("pastas_files"),
        temp_ws: str | Path = Path("pest"),
        master_ws: str | Path = Path("pest"),
        noptmax: int = 0,
        ies_num_reals: int = 50,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        num_workers: int | None = None,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-iES solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name of the executable to run, by default "pestpp-ies".
        model_ws : str | Path, optional
            The working directory for the model, by default Path("model").
        temp_ws : str | Path, optional
            The temporary working directory, by default Path("temp").
        master_ws : str | Path, optional
            The master working directory, by default Path("master") unless
            use_pypestworker is True, then master_ws is equal to temp_ws.
        noptmax : int, optional
            The maximum number of optimization iterations, by default 0.
        ies_num_reals : int, optional
            The number of realizations to draw in order to form parameter and observation ensembles, by default 50.
        control_data : dict[str, Any] | None, optional
            Additional control data for the solver, by default None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix, by default None.
        nfev : int | None, optional
            The number of function evaluations, by default None.
        port_number : int, optional
            The port number for communication, by default 4004.
        num_workers : int | None, optional
            The number of worker processes, by default the number of physical CPU cores.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs
            Additional keyword arguments passed to the base class initializer.

        Returns
        -------
        None
        """

        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            master_ws=master_ws,
            pcov=pcov,
            nfev=nfev,
            port_number=port_number,
            use_pypestworker=use_pypestworker,
            **kwargs,
        )

        self.noptmax = noptmax
        self.ies_num_reals = ies_num_reals
        self.control_data = control_data
        self.num_workers = (
            cpu_count(logical=False) if num_workers is None else num_workers
        )

    def run_ensembles(
        self,
        ies_add_base: bool = True,
        par_sigma_range: float = 4.0,
        observation_noise_standard_deviation: float = 0.0,
        observation_noise_correlation_coefficient: float = 0.0,
        ies_parameter_ensemble_method: Literal["norm", "truncnorm", "uniform"]
        | None = None,
        ies_parameter_ensemble: Optional[DataFrame | None] = None,
        noise_by_obsnme_tag: Optional[DataFrame | None] = None,
        pestpp_options: dict[str, Any] | None = None,
        custom_obs_weights: Optional[DataFrame | None] = None,
        silent: bool = False,
    ) -> None:
        """
        Run ensemble simulations using pestpp-ies.

        Parameters
        ----------
        ies_add_base : bool, optional
            Whether to add the base parameter set to the ensemble, by default
            True. The base ensemble uses the initial parameter values as
            provided by Pastas and does not add noise on the observations.
        par_sigma_range : float, optional
            The difference between a parameters upper and lower bounds
            expressed as standard deviations, by default 4.0.
        observation_noise_standard_deviation : float, optional
            The standard deviation of the observation noise, by default 0.0.
        observation_noise_correlation_coefficient : float, optional
            The correlation coefficient of the observation noise, by default 0.0.
        ies_parameter_ensemble_method : Literal["norm", "truncnorm", "uniform"] | None, optional
            The method to distribution of the prior for the parameter ensemble, by default None.
            If None the parameter distribution is drawn by pestpp-ies itself.
        ies_parameter_ensemble : DataFrame | None, optional
            Optional DataFrame of prior parameter ensemble.
            This par ens is passed to pestpp-ies via control file keyword ies_parameter_ensemble.
            Useful for batch running a pre-developed / optimised stack. Default is None.
        noise_by_obsnme_tag : [DataFrame | None], optional
            Dataframe of obs noise standard deviations. Indexed by obsnme partial str match tags,
            with two columns: 'value' and 'noise_type' ['absolute' or 'relative']. Default is None.
        pestpp_options : dict | None, optional
            Additional PEST++ options, by default None.
        custom_obs_weights : DataFrame | None, optional
            Custom observation weights indexed by model name, with columns of date_from, date_to, obs_type, and weight.
            obs_type can be "head", "stress_obs", or "headdiff"; these along with model name are used to filter obs group
            name (obgnme) for selective weight assignment between the specified dates.
            Default is None.
        Returns
        -------
        None
        """
        self.initialize(version=2)

        # change ies_num_reals
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        pst.pestpp_options["ies_num_reals"] = self.ies_num_reals
        pst.pestpp_options["ies_add_base"] = ies_add_base
        ies_save_binary = eval(str(pestpp_options.get("ies_save_binary", False)).title())
        ies_ens_ext = ".jcb" if ies_save_binary else ".csv"
        pst.pestpp_options["par_sigma_range"] = par_sigma_range

        if custom_obs_weights is not None:
            custom_obs_weights = custom_obs_weights.dropna(subset=["date_from", "date_to", "obs_type", "weight"])
            join_data = pd.concat(
                [
                    self.observations[["obsnme","obs_type"]].set_index("obsnme"),
                    self.stress_obs[["obsnme", "obs_type"]].set_index("obsnme"),
                    ], axis=0, ignore_index=False,
            )
            pst.observation_data = pst.observation_data.join(join_data, how="left")
            pst.observation_data.loc[:, "date"] = pd.to_datetime(pst.observation_data.date, format="%d/%m/%Y")
            for ml_name, row in custom_obs_weights.iterrows():
                mask = (pst.observation_data.obgnme.str.contains(f"{ml_name.lower()}", regex=True)) & \
                        (pst.observation_data.obgnme.str.contains(f"{row.obs_type}_", regex=True)) & \
                       (pst.observation_data.date.between(row.date_from, row.date_to))
                pst.observation_data.loc[mask, "weight"] = row.weight

        if observation_noise_standard_deviation == 0.0 and noise_by_obsnme_tag is None:
            pst.pestpp_options["ies_no_noise"] = True
        elif noise_by_obsnme_tag is not None:
            noise_file = self.write_ensemble_observation_noise_by_obsnme_tag(
                noise_by_obsnme_tag,
                pst.observation_data,
                ies_add_base=ies_add_base,
            )
            pst.pestpp_options["ies_observation_ensemble"] = noise_file
            pst.pestpp_options.pop("ies_no_noise", False)
        else:
            self.write_ensemble_observation_noise(
                standard_deviation=observation_noise_standard_deviation,
                correlation_coefficient=observation_noise_correlation_coefficient,
            )
            pst.pestpp_options["ies_observation_ensemble"] = (
                "pest_starting_obs_ensemble.csv"
            )
            pst.pestpp_options.pop("ies_no_noise", False)
        if ies_parameter_ensemble_method is not None and ies_parameter_ensemble is None:
            self.write_ensemble_parameter_distribution(
                method=ies_parameter_ensemble_method,
                par_sigma_range=par_sigma_range,
                ies_add_base=ies_add_base,
                ies_save_binary=ies_save_binary,
                ies_ens_ext=ies_ens_ext,
            )
            pst.pestpp_options["ies_parameter_ensemble"] = "pest_starting_par_ensemble.csv"
        if ies_parameter_ensemble is not None:
            if ies_parameter_ensemble_method is not None:
                logger.warning(
                    "ies_parameter_ensemble_method is not None, and neither is ies_parameter_ensemble.\n" +
                    "Provided ies_parameter_ensemble overrides ies_parameter_ensemble_method."
                )
            ies_parameter_ensemble = pyemu.ParameterEnsemble(
                pst=pst,
                df=ies_parameter_ensemble,
            )
            ies_par_ens_name = f"pest_starting_par_ensemble{ies_ens_ext}"
            if ies_save_binary:
                ies_parameter_ensemble.to_binary(Path(self.temp_ws / ies_par_ens_name))
            else:
                ies_parameter_ensemble.to_csv(Path(self.temp_ws / ies_par_ens_name))
            pst.pestpp_options["ies_parameter_ensemble"] = (
                ies_par_ens_name
            )

        # add a user-provided pcov (eg from an initial leastsquares solve)
        if self.pcov is not None:
            pst = self.parcov_to_uncfile(pst)

        pestpp_options = {} if pestpp_options is None else pestpp_options
        pst.pestpp_options.update(pestpp_options)

        self.write_pst(pst=pst, version=2)

        pyemu.os_utils.start_workers(
            worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
            exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
            pst_rel_path="pest.pst",  # the control file to use with PEST
            num_workers=self.num_workers,  # how many agents to deploy
            worker_root=self.master_ws.parent,  # where to deploy the agent directories; relative to where python is running
            master_dir=self.master_ws,  # the manager directory
            port=self.port_number,  # the port to use for communication
            verbose=silent,
            silent_master=silent,
            reuse_master=self.reuse_master,
            ppw_function=self.ppw_function
            if self.use_pypestworker
            else None,  # the function to run in the agent
            ppw_kwargs=self.ppw_kwargs
            if self.use_pypestworker
            else {},  # the arguments to pass to the ppw_function
            cleanup=True,
        )

        phidf = pd.read_csv(self.master_ws / "pest.phi.meas.csv", index_col=0)
        self.nfev = phidf.index[-1]
        if self.noptmax > 0:
            self.obj_func = phidf.at[
                self.nfev, "base"
            ]  # could also get mean of all ensembles?

    def parcov_to_uncfile(
        self,
        pst: pyemu.Pst,
    ) -> pyemu.Pst:
        """
        Modify a Pst control file object to include a ++parcov() keyword pointing to a .unc file
        containing a .mat file representation of the pcov dataframe provided to the solver.

        Parameters:
        -----------
        pst : pyemu.Pst object
            The Pst control file object to be modified to include a parcov via an input uncfile.

        Returns:
        --------
        pyemu.Pst
            Modified PEST control file object pointing to provided pcov.
        """
        ies_pcov = self.pcov.copy()
        # rename parcov parameter names to pest names (from pastas model parameter names)
        ies_pcov.rename(
            index=self.ml_parname_to_pst, columns=self.ml_parname_to_pst, inplace=True
        )

        # TODO: If PestSolver.par_transform=="log": Convert pcov from Pastas.Model.pcov untransformed space to log space. Is this even possible?
        #       Or modify Pastas.model.residuals / model.simulation to  allow log-transformation of parameters provided to LSQ.
        if self.par_transform == "log":
            logger.warning(
                "Provided pcov must pertain to log parameter space. This is currently unhandled."
            )

        # convert dataframe to pyemu.Cov object and write to disk, along with a .unc file pointing to it.
        ies_pcov = pyemu.Cov(x=ies_pcov.values, names=ies_pcov.columns)
        ies_pcov.to_ascii(self.model_ws / "pest.prior_parcov.mat")
        ies_pcov.to_uncfile(
            self.model_ws / "pest.prior.unc", covmat_file="pest.prior_parcov.mat"
        )
        # update pst file to read the parcov matrix for prior definition and parameter sampling.
        pst.pestpp_options["parcov"] = "pest.prior.unc"
        return pst

    @staticmethod
    def parameter_distribution(
        ies_num_reals: int,
        initial: float,
        pmin: float,
        pmax: float,
        par_sigma_range: float,
        method: Literal["norm", "truncnorm", "uniform"],
    ) -> NDArray[np.float64]:
        """Generate a distribution of parameter values based on the specified method.

        Parameters
        ----------
        ies_num_reals : int
            Number of ensembles/realizations.
        initial : float
            Initial parameter value.
        pmin : float
            Minimum parameter value.
        pmax : float
            Maximum parameter value.
        par_sigma_range : float
            Range for the parameter sigma.
        method : {'norm', 'truncnorm', 'uniform'}
            Method to use for generating the distribution. 'norm' generates a
            normal distribution, 'truncnorm' generates a truncated normal
            distribution, and 'uniform' generates a uniform distribution.

        Returns
        -------
        np.array
            Array of generated parameter values.
        """
        if method == "norm":
            scale = min(initial - pmin, pmax - initial) / (par_sigma_range / 2)
            rvs = np.sort(norm(loc=initial, scale=scale).rvs(size=ies_num_reals))
            rvs[rvs < pmin] = pmin
            rvs[rvs > pmax] = pmax
        elif method == "truncnorm":
            scale_left = (initial - pmin) / (par_sigma_range / 2)
            tnorm_left = truncnorm(
                a=(pmin - initial) / scale_left, b=0.0, loc=initial, scale=scale_left
            )
            scale_right = (pmax - initial) / (par_sigma_range / 2)
            tnorm_right = truncnorm(
                a=0.0, b=(pmax - initial) / scale_right, loc=initial, scale=scale_right
            )

            left_ies_num_reals = int(
                np.ceil((initial - pmin) / (pmax - pmin) * ies_num_reals)
            )
            right_ies_num_reals = int(
                np.ceil((pmax - initial) / (pmax - pmin) * ies_num_reals)
            )
            rvs_left = tnorm_left.rvs(size=left_ies_num_reals)
            rvs_right = tnorm_right.rvs(size=right_ies_num_reals)
            rvs = np.sort(np.append(rvs_left, rvs_right)[:ies_num_reals])
            rvs[rvs < pmin] = pmin
            rvs[rvs > pmax] = pmax
        elif method == "uniform":
            rvs = np.linspace(
                start=pmin, stop=pmax, num=ies_num_reals
            )  # linspace ensures pmin and pmax are in the ensembles
        else:
            raise ValueError(f"{method=} should be 'norm', 'truncnorm' or 'uniform'.")
        return rvs

    @staticmethod
    def generate_observation_noise(
        ies_num_reals: int,
        nobs: int,
        standard_deviation: float,
        correlation_coefficient: float = 0.0,
        seed: int = pyemu.en.SEED,
    ) -> NDArray[np.float64]:
        """Generate a matrix of normally distributed and optionally correlated noise

        Parameters
        ----------
        ies_num_reals : int
            Number of ensembles/realizations.
        nobs : int
            Number of observations (length of each noise series).
        standard_deviation : float
            Standard deviation of the noise.
        rho : float, optional
            Autoregressive coefficient. Default is 0.0 (pure white noise).
        seed : int, optional
            Random seed for reproducibility, by default pyemu.en.SEED.

        Returns
        -------
        NDArray[np.float64] (nobs, ies_num_reals) matrix
        """
        drng = np.random.default_rng(seed)

        x = drng.normal(loc=0.0, scale=standard_deviation, size=(nobs, ies_num_reals))
        if correlation_coefficient != 0.0:
            sige = np.sqrt(1 - correlation_coefficient**2) * standard_deviation
            e = drng.normal(loc=0.0, scale=sige, size=(nobs, ies_num_reals))
            for j in range(1, nobs):
                x[j] = correlation_coefficient * x[j - 1] + e[j]
        return x

    def write_ensemble_parameter_distribution(
        self,
        method: Literal["norm", "truncnorm", "uniform"] = "norm",
        par_sigma_range: float = 4.0,
        ies_add_base: bool = True,
        seed: int = pyemu.en.SEED,
    ) -> None:
        """
        Generate and write an ensemble of parameter distributions to a CSV file.

        Parameters
        ----------
        method : Literal["norm", "truncnorm", "uniform"], optional
            The method to use for generating the parameter distribution.
            Options are "norm" for normal distribution, "truncnorm" for
            truncated normal distribution, and "uniform" for uniform
            distribution. Default is "norm".
        par_sigma_range : float, optional
            The range of the parameter sigma for the distribution. Default is
            4.0.
        ies_add_base : bool, optional
            If True, add the base parameter values to the ensemble. Default is
            True.
        seed : int, optional
            Random seed for reproducibility, by default pyemu.en.SEED.

        Returns
        -------
        None
        """
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        par_df = pd.DataFrame(
            index=pd.Index(range(self.ies_num_reals)), columns=pst.parameter_data.index
        )
        for pname, pdata in pst.parameter_data.iterrows():
            rvs = PestIesSolver.parameter_distribution(
                ies_num_reals=self.ies_num_reals,
                initial=pdata.at["parval1"],
                pmin=pdata.at["parlbnd"],
                pmax=pdata.at["parubnd"],
                par_sigma_range=par_sigma_range,
                method=method,
            )
            par_df[pname] = rvs
        # shuffle each column with the initial parameters independently
        par_df.loc[:, :] = np.random.default_rng(seed=seed).permuted(
            par_df.values, axis=0
        )
        if ies_add_base:
            par_df.loc[self.ies_num_reals - 1] = pst.parameter_data.loc[
                :, "parval1"
            ].values
            par_df = par_df.rename(index={self.ies_num_reals - 1: "base"})
        par_df.to_csv(self.temp_ws / "pest_starting_par_ensemble.csv")

    def write_ensemble_observation_noise_by_obsnme_tag(
            self,
            noise_by_obsnme_tag: DataFrame,
            obs_data: DataFrame,
            ies_add_base: bool = True,
    ):
        """
        Generate and write an ensemble of observation noise to a CSV file, based on options contained in
        the noise_by_obsnme_tag DataFrame, which is indexed by obsnme partial string match tags.

        Parameters
        ----------
        noise_by_obsnme_tag: DataFrame
            DataFrame, which is indexed by obsnme partial string match tags.
            Columns are 'value', 'noise_type' ['absolute' or 'relative' (to mean of obs values)],
            'correlation coefficient' (of noise; 0-->1), 'minobsval' and 'maxobsval' (leave null if not wanted).
        obs_data : DataFrame
            Pyemu.Pst.observation_data DataFrame.
        ies_add_base : bool, optional
            If True, add the base observation values to the ensemble. Default
            is True.

        Returns
        -------
        noise_file : str
        The path to the noise file. Needs to be specified in the pst control file.
        """
        obs_plus_noise_list = []
        for otag, row in noise_by_obsnme_tag.iterrows():
            obsvals = obs_data.loc[obs_data.index.to_series().str.contains(otag)].obsval
            if row.noise_type == "relative":
                stdev = row.value * obsvals.mean()
            else: # absolute
                stdev = row.value
            noise = PestIesSolver.generate_observation_noise(
                ies_num_reals=self.ies_num_reals,
                nobs=len(obsvals.index),
                standard_deviation=stdev,
                correlation_coefficient=row.correlation_coefficient,
                seed=pyemu.en.SEED,
            )
            obs_plus_noise = obsvals.to_frame().values + noise
            if not np.isnan(row.minobsval):
                obs_plus_noise = np.maximum(obs_plus_noise, row.minobsval)
            if not np.isnan(row.maxobsval):
                obs_plus_noise = np.minimum(obs_plus_noise, row.maxobsval)
            obs_noise_df = pd.DataFrame(
                obs_plus_noise,
                index=obsvals.index,
                columns=pd.Index(range(self.ies_num_reals)),
            ).transpose()
            if ies_add_base:
                obs_noise_df.loc[self.ies_num_reals - 1] = obsvals
                obs_noise_df = obs_noise_df.rename(index={self.ies_num_reals - 1: "base"})
            obs_plus_noise_list.append(
                obs_noise_df.copy()
            )

        obs_noise_df = pd.concat(obs_plus_noise_list, axis=1, ignore_index=False)
        # check for obsnmes not in noise ens, and add those with zero noise
        missing_obsnmes = obs_data.index[~obs_data.index.isin(obs_noise_df.columns)]
        obs_noise_df.loc[:, missing_obsnmes] = obs_data.loc[missing_obsnmes].obsval.values
        obs_noise_df = obs_noise_df.loc[:, obs_data.index] # reorder per pst file
        # save the noise ensemble and add to control file
        noise_file = self.temp_ws / "pest_starting_obs_ensemble.csv"
        obs_noise_df.to_csv(noise_file)

        return noise_file

    def write_ensemble_observation_noise(
        self,
        standard_deviation: float = 0.0,
        correlation_coefficient: float = 0.0,
        ies_add_base: bool = True,
    ) -> None:
        """
        Generate and write an ensemble of observation noise to a CSV file.

        Parameters
        ----------
        standard_deviation : float, optional
            The standard deviation of the observation noise. Default is 0.0.
        correlation_coefficient : float, optional
            The correlation coefficient of the observation noise. Default is
            0.0.
        ies_add_base : bool, optional
            If True, add the base observation values to the ensemble. Default
            is True.

        Returns
        -------
        None
        """
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        noise = PestIesSolver.generate_observation_noise(
            ies_num_reals=self.ies_num_reals,
            nobs=len(pst.observation_data.index),
            standard_deviation=standard_deviation,
            correlation_coefficient=correlation_coefficient,
            seed=pyemu.en.SEED,
        )
        obs_data = pst.observation_data.loc[:, ["obsval"]].values
        obs_noise_df = pd.DataFrame(
            obs_data + noise,
            index=pst.observation_data.index,
            columns=pd.Index(range(self.ies_num_reals)),
        ).transpose()
        if ies_add_base:
            obs_noise_df.loc[self.ies_num_reals - 1] = obs_data.flatten()
            obs_noise_df = obs_noise_df.rename(index={self.ies_num_reals - 1: "base"})
        obs_noise_df.to_csv(self.temp_ws / "pest_starting_obs_ensemble.csv")

    def parameter_ensemble(self, iteration: int = 0) -> pyemu.ParameterEnsemble:
        """
        Read a parameter ensemble for a given iteration.

        Parameters:
        -----------
        iteration : int, optional
            The iteration number for which to read the parameter ensemble.
            Default is 0.

        Returns:
        --------
        pyemu.ParameterEnsemble
            The parameter ensemble for the specified iteration.
        """

        pst = pyemu.Pst(str(self.master_ws / "pest.pst"))
        pe = pyemu.ParameterEnsemble.from_csv(
            pst=pst, filename=self.master_ws / f"pest.{iteration}.par.csv"
        )
        return pe

    @lru_cache()
    def simulation_ensemble(
        self,
        iteration: int = 0,
        from_file: bool = False,
        tmin: TimestampType = None,
        tmax: TimestampType = None,
    ) -> pd.DataFrame:
        """
        Generate or read a simulation ensemble.

        Parameters
        ----------
        iteration : int, optional
            The iteration number for which to read the simulation ensemble.
            Default is 0.
        from_file : bool, optional
            If True, read the simulation ensemble from a file. If False,
            generate it with Pastas from the parameter ensemble. Default is
            False.
        tmin : TimestampType, optional
            The minimum timestamp for the simulation period. If None, use the
            model's tmin setting. Default is None.
        tmax : TimestampType, optional
            The maximum timestamp for the simulation period. If None, use the
            model's tmax setting. Default is None.

        Returns
        -------
        pd.DataFrame
            The simulation ensemble as a DataFrame.
        """
        if from_file:
            pst = pyemu.Pst(str(self.master_ws / "pest.pst"))
            se = (
                pyemu.ObservationEnsemble.from_csv(
                    pst=pst, filename=self.master_ws / f"pest.{iteration}.obs.csv"
                )
                .transpose()
                .set_index(self.ml.observations().index)
            )
        else:
            ipar = self.parameter_ensemble(iteration=iteration).transpose()
            ipar.index = self.ml.parameters.index[self.vary]

            tmin = self.ml.settings["tmin"] if tmin is None else pd.Timestamp(tmin)
            tmax = self.ml.settings["tmax"] if tmax is None else pd.Timestamp(tmax)
            freq = (
                "D"
                if self.ml.settings["freq"] is not None
                else self.ml.settings["freq"]
            )
            se = pd.DataFrame(
                np.nan,
                columns=ipar.columns,
                index=pd.date_range(start=tmin, end=tmax, freq=freq),
            )

            for idx in ipar.columns:
                self.ml.parameters.loc[ipar.index, "optimal"] = ipar.loc[:, idx].values
                se.loc[:, idx] = (
                    self.ml.simulate(tmin=tmin, tmax=tmax).loc[se.index].values
                )

        return se

    def observation_ensemble(self) -> pyemu.ObservationEnsemble:
        """
        Generate an observation ensemble from a CSV file. This method reads a
        PEST control file and a corresponding observation noise CSV file to
        create an observation ensemble.

        Returns
        -------
        pyemu.ObservationEnsemble
            The generated observation ensemble.
        """

        pst = pyemu.Pst(str(self.master_ws / "pest.pst"))
        oe = pyemu.ObservationEnsemble.from_csv(
            pst=pst, filename=self.master_ws / "pest.obs+noise.csv"
        )
        return oe

    def jacobian(self, iteration: int = 0) -> pd.DataFrame:
        """
        Calculate the Jacobian matrix for the given iteration.
        The Jacobian matrix is computed using the simulation ensemble and
        parameter ensemble for the specified iteration. The calculation
        involves normalizing the ensembles and using the pseudo-inverse
        of the parameter ensemble.

        Parameters:
        -----------
        iteration : int, optional
            The iteration number for which the Jacobian is calculated.
            Default is 0.

        Returns:
        --------
        pd.DataFrame
            A DataFrame representing the Jacobian matrix, with the same
            indices as the observation ensemble and the same columns as
            the parameter ensemble.
        """
        # jac_ies needs to be calculated manually
        obs_ies = self.simulation_ensemble(
            iteration=iteration, from_file=True
        ).transpose()
        par_ies = self.parameter_ensemble(iteration=iteration)
        jac = PestIesSolver.jacobian_emperical(obs_ies.values, par_ies.values)
        jac_ies = pd.DataFrame(jac, index=obs_ies.index, columns=par_ies.columns)
        return jac_ies

    @staticmethod
    def jacobian_emperical(
        simulation_ensembles: NDArray[np.float64],
        parameter_ensembles: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the approximate Jacobian matrix for the given ensembles.

        Parameters
        ----------
        simulation_ensembles : NDArray[np.float64]
            Ensembles of the simulated values of shape (nobs, nreals)
        parameter_ensembles : NDArray[np.float64]
            Ensembles of the paramters of shape (nreals, npar)

        Returns
        -------
        NDArray[np.float64]
            Approximate, empirical Jacobian matrix
        """
        _, ies_num_reals_sim = simulation_ensembles.shape
        ies_num_reals_par, _ = parameter_ensembles.shape
        if ies_num_reals_par != ies_num_reals_sim:
            raise AssertionError(
                f"Number of realizations in parameter {ies_num_reals_par} and"
                f"simulation {ies_num_reals_sim} ensembles must be equal"
            )
        else:
            ies_num_reals = ies_num_reals_sim

        deviations_sim = (
            simulation_ensembles.T - np.mean(simulation_ensembles, axis=1)
        ).T / np.sqrt(ies_num_reals - 1)
        deviations_par = (
            parameter_ensembles - np.mean(parameter_ensembles, axis=0)
        ).T / np.sqrt(ies_num_reals - 1)
        jac = deviations_sim @ np.linalg.pinv(deviations_par)
        return jac

    def solve(
        self, run_ensembles: bool = False, **kwargs
    ) -> tuple[bool, NDArray[np.float64], NDArray[np.float64]]:
        """
        Gets the base realization of the parameter ensemble.

        Parameters
        ----------
        run_ensembles : bool, optional
            If True, runs the ensembles with the provided keyword arguments (default is False).
        **kwargs : dict
            Additional keyword arguments to pass to the `run_ensembles` method.

        Returns
        -------
        tuple
            A tuple containing:
            - bool: Always returns True.
            - numpy.ndarray: The optimal parameters.
            - numpy.ndarray: The standard error of the parameters.
        """
        if "noise" in kwargs:
            del kwargs["noise"]  # remove noise from kwargs, not used in PestIesSolver
        if "weights" in kwargs:
            del kwargs["weights"]

        if run_ensembles:
            self.run_ensembles(**kwargs)

        # optimal parameters
        ipar = self.parameter_ensemble(iteration=self.nfev).transpose()
        ipar.index = self.ml.parameters.index[self.vary]
        optimal = self.ml.parameters["initial"].copy().values
        optimal[self.vary] = ipar.loc[:, "base"].values

        # standard error (could be totally the wrong way to think about/calculate this)
        stderr = np.full_like(optimal, np.nan)
        stderr[self.vary] = ipar.std(axis=1) / np.sqrt(len(ipar.columns))

        return True, optimal, stderr


class PestSenSolver(PestSolver):
    """PESTPP-SEN (Global Sensitivity Analysis) solver"""

    def __init__(
        self,
        exe_name: str | Path = "pestpp-sen",
        model_ws: str | Path = Path("pastas_files"),
        temp_ws: str | Path = Path("pest"),
        master_ws: str | Path = Path("pest"),
        noptmax: int = 0,
        control_data: dict[str, Any] | None = None,
        pcov: DataFrame | None = None,
        nfev: int | None = None,
        port_number: int = 4004,
        num_workers: int | None = None,
        use_pypestworker: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize the PESTPP-SEN class. This class is used to run the
        PESTPP-SEN analysis and is not really a solver.

        Parameters
        ----------
        exe_name : str | Path, optional
            The name of the executable to run, by default "pestpp-sen".
        model_ws : str | Path, optional
            The working directory for the model, by default Path("model").
        temp_ws : str | Path, optional
            The temporary working directory, by default Path("temp").
        master_ws : str | Path, optional
            The master working directory, by default Path("master") unless
            use_pypestworker is True, then master_ws is equal to temp_ws.
        noptmax : int, optional
            The maximum number of optimization iterations, by default 0.
        control_data : dict[str, Any] | None, optional
            Control data for the solver, by default None.
        pcov : DataFrame | None, optional
            The parameter covariance matrix, by default None.
        nfev : int | None, optional
            The number of function evaluations, by default None.
        port_number : int, optional
            The port number for communication, by default 4004.
        num_workers : int | None, optional
            The number of worker processes, by default the number of physical CPU cores.
        use_pypestworker : bool, optional
            Whether to use the PyPestWorker for Python processing. Default is True.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        PestSolver.__init__(
            self,
            exe_name=exe_name,
            model_ws=model_ws,
            temp_ws=temp_ws,
            master_ws=master_ws,
            pcov=pcov,
            nfev=nfev,
            port_number=port_number,
            use_pypestworker=use_pypestworker,
            **kwargs,
        )
        self.noptmax = noptmax
        self.control_data = control_data
        self.num_workers = (
            cpu_count(logical=False) if num_workers is None else num_workers
        )

    def start(
        self, pestpp_options: dict[str, Any] | None = None, silent: bool = False
    ) -> None:
        """
        Start the PESTPP-SEN analysis.

        This method sets up the model and files, updates the PEST control file with
        the provided PEST++ options, and starts the PESTPP-SEN workers.

        Parameters
        ----------
        pestpp_options : dict[str, Any], optional
            Additional PEST++ options to update in the PEST control file, by default None.

        Returns
        -------
        None
        """

        self.initialize(version=2)

        # change ies_num_reals
        pst = pyemu.Pst(str(self.temp_ws / "pest.pst"))
        pestpp_options = {} if pestpp_options is None else pestpp_options
        pst.pestpp_options.update(pestpp_options)

        self.write_pst(pst=pst, version=2)

        pyemu.os_utils.start_workers(
            worker_dir=self.temp_ws,  # the folder which contains the "template" PEST dataset
            exe_rel_path=self.exe_name.name,  # the PEST software version we want to run
            pst_rel_path="pest.pst",  # the control file to use with PEST
            num_workers=self.num_workers,  # how many agents to deploy
            worker_root=self.master_ws.parent,  # where to deploy the agent directories; relative to where python is running
            port=self.port_number,  # the port to use for communication
            master_dir=self.master_ws,  # the manager directory
            reuse_master=self.reuse_master,
            verbose=silent,
            silent_master=silent,
            ppw_function=self.ppw_function
            if self.use_pypestworker
            else None,  # the function to run in the agent
            ppw_kwargs=self.ppw_kwargs
            if self.use_pypestworker
            else {},  # the arguments to pass to the ppw_function
        )

    def solve() -> None:
        raise NotImplementedError(
            "PestSenSolver does not have a solve method. Run the sensitivity"
            "analysis using the `start` method."
        )
