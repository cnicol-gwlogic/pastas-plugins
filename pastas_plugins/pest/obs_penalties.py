import logging, dill, gzip
import numpy as np
import pandas as pd
import geopandas as gpd

from typing import Optional, Callable
from shutil import copy as copy_file
from pathlib import Path
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


logger = logging.getLogger(__name__)

@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StressContribPenaltySettings:
    """
    Settings for ColocatedStressContribPenalties and BetweenStressContribPenalties

    Parameters
    ----------
    colocated_penalty_obs: bool
        Default is True
    colocated_penalty_max_separation_distance: float
        Default is 250.0
    colocated_penalty_max_difference_percent: float
        Default is 10.0
    colocated_penalty_obs_phi_factor: float
        Default is 0.1
    colocated_penalty_stress_contribution_groups: list | None
        Stress contribution groups for which penalties are applied based on the group's well locations' centroid.
        All names must be in the colnme field of solver.sm_contribs.
        Default is an empty list, in which case all common stress contribution groups are used.
    between_penalty_obs: bool
        Default is True
    between_penalty_stress_contribution_groups: list | None
        Stress contribution groups for which penalties are applied based on the group's well locations' centroid.
        All names must be in the colnme field of solver.sm_contribs.
        Default is an empty list, in which case all common stress contribution groups are used.
    between_penalty_max_distance_from_connecting_line: float
        Default is 1000.0
    between_penalty_obs_phi_factor: float
        Default is 0.1
    solver: PestSolver | None
    """
    colocated_penalty_obs: Optional[bool] = True
    colocated_penalty_max_separation_distance: Optional[float] = 250.0
    colocated_penalty_max_difference_percent: Optional[float] = 10.0
    colocated_penalty_obs_phi_factor: Optional[float] = 0.1
    colocated_penalty_stress_contribution_groups: Optional[list] = Field(default_factory=list)
    between_penalty_obs: Optional[bool] = True
    between_penalty_stress_contribution_groups: Optional[list] = Field(default_factory=list)
    between_penalty_max_distance_from_connecting_line: Optional[float] = 1000.0
    between_penalty_obs_phi_factor: Optional[float] = 0.1
    solver = None
    if colocated_penalty_obs or between_penalty_obs:
        logger.warning(
            "User beware! stress_contribution_penalty_obs require that model.oseries.metadata['x'] "
            "and model.oseries.metadata['y'] must be provided for all Pastas models"
        )
        # warn user of critical point
        logger.warning("IMPORTANT NOTE: All models in solver.models must use the SAME stress direction (up OR down) "
                       "for penalty_obs to work correctly.")

def sanitise_differences(differences: pd.DataFrame) -> pd.DataFrame:
    """
    Replace index names with pyemu-friendly names, and force/check date index is datetime dtype.

    Parameters
    ----------
    differences: pd.DataFrame
        colocated or between_bore stress constribution differences from either
        ColocatedStressContribPenalties.get_colocated_differences() or
        BetweenStressContribPenalties.get_colocated_differences()

    Returns
    -------
    differences: pd.DataFrame
        As per input, except replaced index names with pyemu-friendly names, and forced date index to datetime dtype.
    """
    # make a copy of the original (probably too-long for pyemu/pest) columns
    #orig_index_names = differences.index.names
    #differences.loc[:, differences.index.names] = differences.index.values
    # replace index name _ with . for pest/pyemu - underscores mess with pyemus obsnme convention / identifying indexes etc
    differences.index.names = [str(n).replace('_', '.') for n in differences.index.names]
    penalty_index_names = differences.index.names
    # ensure dt index - seems flaky
    differences.reset_index(drop=False, inplace=True)
    differences["date"] = pd.to_datetime(differences.date)
    differences.set_index(penalty_index_names, inplace=True)
    # consistent sort by penalty_index_names
    differences.sort_index(inplace=True)
    return differences

class ColocatedStressContribPenalties:
    """
    Make PEST penalty obs for stress contribution differences between colocated models (obs bores).
    Forward run makes these penalty obs after all pastas (obs bore) models are run.
    Initial obs values are colocated_penalty_max_difference_percent, and they are less_than type.
    """
    OUTPUT_PENALTY_FILE = "sim_str_cont_coloc_pen.csv"
    def __init__(
            self,
            settings: StressContribPenaltySettings,
    ):
        self.solver = settings.solver
        self.solver.assign_model_coords(force_update=True)
        self.max_separation_distance = settings.colocated_penalty_max_separation_distance
        self.max_difference_percent = settings.colocated_penalty_max_difference_percent
        self.stress_contribution_groups = settings.colocated_penalty_stress_contribution_groups

        self.colocated_bores = self._get_colocated_bores()
        self.solver.colocated_bores = self.colocated_bores
        self.obsgp = f"less_than_stress_contrib_penalty_colocated"
        self.difference_obs = self._calc_differences(self.solver.sm_contribs)
        # warn user of critical point
        logger.warning("IMPORTANT NOTE: All models in solver.models must use the SAME stress direction (up OR down) "
                       "for colocated_penalty_obs to work correctly.")

    def _get_colocated_bores(self) -> gpd.GeoDataFrame:
        """
        Find obs bores (models) within max_separation_distance of every model (obs bore) in the pestsolver.

        Parameters
        ----------

        Returns
        -------
        colocated_bores: gpd.GeoDataFrame
            Multi-indexed by (ml_name, ml_name_r) with values being geometry and near geometry (geometry_r).
            "ml_name_r" is the colocated (nearby, within max_separation_distance of ml_name) obs bore.
        """
        ml_gdf = self.solver.ml_gdf.reset_index(drop=False)
        colocated_bores = gpd.sjoin_nearest(
            ml_gdf, ml_gdf, how='inner',
            lsuffix="", rsuffix="r",
            max_distance=self.max_separation_distance,
            exclusive=True,
        )
        colocated_bores.rename(columns={"ml_name_": "ml_name"}, inplace=True)
        # drop duplicate pairs
        sorted_pairs = np.sort(colocated_bores[["ml_name", "ml_name_r"]], axis=1)
        colocated_bores = colocated_bores.loc[pd.DataFrame(sorted_pairs).duplicated(keep='first').values]
        colocated_bores.set_index(["ml_name", "ml_name_r"], inplace=True)
        colocated_bores.loc[:, "max_difference_percent"] = self.max_difference_percent
        # save the data
        self.colocated_bores_file = Path(self.solver.model_ws / f"sim_stress_contrib_colocated_bores.csv")
        colocated_bores.to_csv(self.colocated_bores_file)
        copy_file(self.colocated_bores_file, self.solver.temp_ws)
        return colocated_bores

    @staticmethod
    def get_colocated_differences(
            colocated_bores: gpd.GeoDataFrame,
            sm_contribs: pd.DataFrame,
            set_to_max_difference_percent: bool,
            max_difference_percent: float,
            min_abs_value_assess: float=0.001,
            stress_contribution_groups: Optional[list | None] = None,
    ) -> pd.Series:
        """
        Calculate stress_contribution differences between colocated models (obs bores).

        Parameters
        ----------
        colocated_bores: gpd.GeoDataFrame
            Multi-indexed by (ml_name, ml_name_r) with values being geometry and near geometry (geometry_r).
            "ml_name_r" is the colocated (nearby, within max_separation_distance of ml_name) obs bore.
        sm_contribs: pd.DataFrame
            Stressmodel contributions
        set_to_max_difference_percent: bool
            Set values to max_difference_percent (e.g. for observation target creation).
        max_difference_percent: float
            Maximum % difference between stress contributions for colocated bores. Percent calculated based on the
            element-wise maximum of the two series being compared.
        min_abs_value_assess: float
            Minimum value across both stress contribution series, below which we do not compare % differences. Avoids
            precision / not-meaningful effects.
        stress_contribution_groups: Optional[list | None] = None
            Default is None, in which case all common stress contribution groups are used.

        Returns
        ----------
        differences: pd.Series
            Stressmodel contribution differences, MultiIndexed by ('ml_name', 'ml_name_r', 'colnme', 'date')
        """
        penalty_index_names = ['ml_name', 'ml_name_r', 'colnme', 'date']
        differences = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=penalty_index_names),
            columns=['Observations'],
        )
        for ml_name, df in colocated_bores.groupby(level='ml_name'):
            for ml_name_r, row in df.xs(ml_name).iterrows():
                # get contribs df indexed by (colnme, date)
                mask = (sm_contribs.model_name == ml_name)
                rmask = (sm_contribs.model_name == ml_name_r)
                if stress_contribution_groups is not None:
                    mask = mask & (sm_contribs.index.get_level_values("colnme").isin(stress_contribution_groups))
                    rmask = rmask & (sm_contribs.index.get_level_values("colnme").isin(stress_contribution_groups))
                ml_contribs = sm_contribs.loc[mask, "Observations"].abs().clip(lower=min_abs_value_assess)
                ml_contribs_r = sm_contribs.loc[rmask, "Observations"].abs().clip(lower=min_abs_value_assess)
                # subtract one from the other and assign to (ml_name, ml_name_r)
                # Below, we use .round(6) on divisors: Although rounding tends to be dangerous for pest,
                # it can cause huge meaningless % results below...which is more dangerous.
                diff = (ml_contribs - ml_contribs_r).abs().round(6)  # assume same stress direction (user beware)
                # convert to % of larger contribution
                max = ml_contribs.combine(ml_contribs_r, np.maximum, fill_value=np.nan).astype(float).round(6)
                uncommon_entries_mask = max.isna()
                diff = diff[~uncommon_entries_mask]
                max = max[~uncommon_entries_mask]
                diff = (diff / max) * 100.0  # reverted .dropna() and added np.nan to replace below.
                diff = diff.replace([np.nan, np.inf, -np.inf], 0.0)
                diff = pd.concat([diff], keys=[(ml_name, ml_name_r)], names=['ml_name', 'ml_name_r'])
                if set_to_max_difference_percent:
                    diff.loc[
                        :, :] = max_difference_percent  # reset values to max_difference_percent - for now we want a target obs diff of < max_difference_percent
                # keep only those ml_name / ml_name_r / colnme / date not already in differences
                drop_mask = diff.index.isin(differences.index.values)
                differences = pd.concat([differences, diff.loc[~drop_mask]], ignore_index=False)
                # consistent sort by penalty_index_names
                differences.sort_index(inplace=True)
        return differences

    def _calc_differences(
            self,
            sm_contribs: pd.DataFrame,
            set_to_max_difference_percent: bool = True,
    ) -> pd.Series:
        """
        Calculate stress_contribution differences between colocated models (obs bores).

        Parameters
        ----------
        sm_contribs: pd.DataFrame
            Stressmodel contributions
        set_to_max_difference_percent: bool
            Set values to max_difference_percent (e.g. for observation target creation).
            Default is True.

        Returns
        ----------
        differences: pd.Series
            Stressmodel contribution differences, MultiIndexed by ('ml_name', 'ml_name_r', 'colnme', 'date')
        """
        differences = ColocatedStressContribPenalties.get_colocated_differences(
            self.colocated_bores, sm_contribs, set_to_max_difference_percent,
            self.max_difference_percent, min_abs_value_assess=0.001,
            stress_contribution_groups=self.stress_contribution_groups
        )
        differences = sanitise_differences(differences)
        self.penalty_index_names = differences.index.names
        # save the data
        self.penalty_file = Path(self.solver.model_ws / ColocatedStressContribPenalties.OUTPUT_PENALTY_FILE)
        differences.to_csv(self.penalty_file, date_format=self.solver.date_format)
        copy_file(self.penalty_file, self.solver.temp_ws)
        return differences

    def make_difference_obs(self) -> pd.DataFrame:
        """
        Makes PEST observation data for stress_contribution differences.

        Returns
        ----------
        pst_from_obs_df: pd.DataFrame
        """
        self.solver.pf.add_observations(
            self.penalty_file.name,
            index_cols=self.penalty_index_names,
            use_cols=["Observations"],
            obsgp=self.obsgp,
        )
        self.solver.pf.obs_dfs[-1]["weight"] = 1.0
        return self.solver.pf.obs_dfs[-1]


class BetweenStressContribPenalties:
    """
    Find obs bores (models) within max_distance_from_connecting_line to specified stressor(s) in
    settings.between_penalty_stress_contribution_groups for every model (obs bore) in the pestsolver.
    Add less_than difference obs (ie contrib at bigger distances < contrib at closer distances)
    for each bore pair, starting with the most distal one / next most distal,
    then moving in closer to stressor.
    Exclude bore pairs within settings.colocated_penalty_max_separation_distance.
    Forward run makes these penalty obs after all pastas (obs bore) models are run.
    Initial obs values are zero, and they are less_than type, so distal bores have less
    stress_contribution than closer bores.

    Attributes:
    -----------
    sm_contrib_group_centroids: gpd.GeoDataFrame
        Centroid of each stress_contribution_group in settings.between_penalty_stress_contribution_groups.
        MultiIndexed by ["stress_contribution_group", "ml_name"], with values being a Point of the
        centroid of the given stress_contribution_group for the given model (ml_name).
    between_bores: gpd.GeoDataFrame
        Obs bores within settings.between_penalty_max_distance_from_connecting_line (from model to sm_contrib centroid)
        for each stress_contribution_group in settings.between_penalty_stress_contribution_groups.
        MultiIndexed by (stress_contribution_group, buffer_name, ml_name) with values being (obs bore)
        distance_to_sm_contrib, sm_contrib_centroid, and geometry (Point of origin model loc).
        Needed to identify (for a given model (obs bore)) those other obs bores (models) located between the model and
        a given stressmodel contribution centroid, so that PEST obs penalties can be applied where stress contributions
        are larger at obs bores at greater distance from the stressor.
    between_bore_pairs: gpd.GeoDataFrame
        Adjacent obs bore pairs from between_bores (ml_name, ml_name_r), listed sequentially from those furthest from
        the stressor.
    difference_obs: gpd.GeoDataFrame
    """
    SETTINGS_PICKLE_FILE = "sim_str_cont_bw_pen_settings.pkl.gz"
    OUTPUT_PENALTY_FILE = "sim_str_cont_bw_pen.csv"
    def __init__(
            self,
            settings: StressContribPenaltySettings
    ):
        # prep required solver attributes
        self.solver = settings.solver
        self.solver.assign_model_coords(force_update=True)
        self.solver.assign_stressmodel_coords(force_update=True)

        # assign class parameters needed from settings
        self.stress_contribution_groups = settings.between_penalty_stress_contribution_groups
        self.max_distance_from_connecting_line = settings.between_penalty_max_distance_from_connecting_line
        self.exclude_sep_distance = settings.colocated_penalty_max_separation_distance

        # penalty obs prep work
        self.sm_contrib_group_centroids = self._get_sm_contrib_group_centroids()
        self.between_bores = self._get_between_bores()
        self.between_bore_pairs = self._get_between_bore_pairs()

        # make difference penalty obs for between_bore_pairs
        self.obsgp = f"less_than_stress_contrib_penalty_between"
        self.difference_obs = self._calc_differences(self.solver.sm_contribs)

        # save settings for pest forward run
        with gzip.open(Path(self.solver.model_ws / BetweenStressContribPenalties.SETTINGS_PICKLE_FILE), "wb") as f:
            dill.dump(settings, f)  # pickle
        copy_file(Path(self.solver.model_ws / BetweenStressContribPenalties.SETTINGS_PICKLE_FILE), self.solver.temp_ws)

        # warn user of critical point
        logger.warning("IMPORTANT NOTE: All models in solver.models must use the SAME stress direction (up OR down) "
                       "for between_penalty_obs to work correctly.")

    def _get_sm_contrib_group_centroids(self) -> gpd.GeoDataFrame:
        """
        Get centroids of all between_penalty_stress_contribution_groups for every model in PestSolver.models

        Returns
        -------
        sm_contrib_group_centroids: gpd.GeoDataFrame
            MultiIndexed by ["stress_contribution_group", "ml_name"], with values being a Point of the
            centroid of the given stress_contribution_group for the given model (ml_name).
        """
        mux = index=pd.MultiIndex.from_tuples([], names=["stress_contribution_group", "ml_name"])
        sm_contrib_group_centroids = pd.DataFrame(columns=["geometry"], index=mux)
        for stress_contribution_group in self.stress_contribution_groups:
            for ml_name,ml in self.solver.models.items():
                contrib_group_istresses = self.solver.stress_contribution_groups.loc[
                    (ml_name, stress_contribution_group)
                ].istress_names.values
                coords = self.solver.sm_gdf.loc[
                    (ml_name, slice(None), contrib_group_istresses)
                ].reset_index(drop=False).drop(columns=["ml_name","sm_name"]).dissolve(by="istress_name", aggfunc='mean') # sm_gdf has a block of istress_names for every model/stressmodel)
                sm_contrib_group_centroids.loc[(stress_contribution_group, ml_name), "geometry"] = \
                    coords.union_all().centroid

        return gpd.GeoDataFrame(sm_contrib_group_centroids, geometry="geometry")

    def _get_between_bores(self) -> gpd.GeoDataFrame:
        """
        For each given stress_contribution_group, find obs bores (models) within max_distance_from_connecting_line
        to specified stressor(s) for every model (obs bore) in the pestsolver.

        Returns
        -------
        between_bores: gpd.GeoDataFrame
            Obs bores (models) within max_distance_from_connecting_line to specified stressor(s)
            (stress_contribution_group(s)) for every model (obs bore) in the pestsolver.
            MultiIndexed by (stress_contribution_group, buffer_name, ml_name) with values being distance_to_sm_contrib,
            sm_contrib_centroid, and geometry (model location to which the line buffer / contained other obs bores
            is assessed).
        """
        mux = index=pd.MultiIndex.from_tuples([], names=["stress_contribution_group", "buffer_name", "ml_name"])
        _models_in_buffers = gpd.GeoDataFrame(columns=["geometry"], index=mux)
        for stress_contribution_group in self.stress_contribution_groups:
            sm_contrib_centroids = self.sm_contrib_group_centroids.xs(stress_contribution_group).geometry # indexed by ml_name
            ml_locs = self.solver.ml_gdf.geometry # indexed by ml_name
            lines = ml_locs.shortest_line(sm_contrib_centroids, align=True)
            buffers = gpd.GeoDataFrame(
                index=lines.index,
                geometry=lines.buffer(self.max_distance_from_connecting_line, cap_style='flat')
            )
            buffers["stress_contribution_group"] = stress_contribution_group
            buffers["sm_contrib_centroid"] = sm_contrib_centroids.geometry
            # shrink buffers where models are within self.max_distance_from_connecting_line of stressor - DEACTIVATED FOR NOW - CONSIDER
            #mask = (buffers.distance_to_sm_contrib < self.max_distance_from_connecting_line)
            #buffers.loc[mask, "geometry"] = lines.loc[mask].buffer(buffers.loc[mask].distance_to_sm_contrib, cap_style='flat')
            buffers["buffer_name"] = buffers.index
            models_in_buffers = gpd.sjoin(
                self.solver.ml_gdf.reset_index(drop=False).loc[:,["ml_name","geometry"]],
                gpd.GeoDataFrame(data=buffers, geometry="geometry"),
                how="inner", predicate="covered_by"
            )
            models_in_buffers["ml_left_geom"] = self.solver.ml_gdf.loc[models_in_buffers.ml_name_left].geometry.values
            models_in_buffers["distance_to_sm_contrib"] = models_in_buffers.sm_contrib_centroid.distance(
                models_in_buffers.ml_left_geom
            )
            models_in_buffers["sm_contrib_centroid.x"] = models_in_buffers.sm_contrib_centroid.x
            models_in_buffers["sm_contrib_centroid.y"] = models_in_buffers.sm_contrib_centroid.y
            models_in_buffers = models_in_buffers.drop(
                columns=["ml_name_right", "ml_left_geom", "sm_contrib_centroid"]
            ).rename(columns={"ml_name_left": "ml_name"})
            models_in_buffers.set_index(["stress_contribution_group", "buffer_name", "ml_name"], inplace=True) # can have lots of models in a given buffer
            _models_in_buffers = pd.concat([_models_in_buffers, models_in_buffers], ignore_index=False)
        between_bores = _models_in_buffers
        # sort by distance to stressor (descending far to near)
        between_bores.sort_values(
            by=["stress_contribution_group", "buffer_name", "distance_to_sm_contrib"],
            ascending=[True, True, False],
            inplace=True
        )
        self.between_bores_file = Path(self.solver.model_ws / f"sim_stress_contrib_between_bores.gpkg")
        between_bores.to_file(self.between_bores_file)
        copy_file(self.between_bores_file, self.solver.temp_ws)
        return between_bores

    def _get_between_bore_pairs(self):
        """
        Identify bore pairs within each buffer polygon (around line connecting model with sm_contrib centroid).
        Bore pairs are defined sequentially: starting from most distal model, find the next most distal model (that is
        further from the model than self.exclude_sep_distance).

        Returns
        -------
        model_pairs_in_buffers: gpd.GeoDataFrame
            MultiIndexed by (stress_contribution_group, buffer_name, ml_name); column ml_name_r is the next closest
            model to ml_name and separation_distance is the separation distance between them.
            In other words, adjacent bore pairs within a given model's line buffer from the model to the
            stressor are each [ml_name, ml_name_r] combo.
        """
        model_pairs_in_buffers = self.between_bores.copy()
        # make bore pairs
        model_pairs_in_buffers.reset_index(drop=False, inplace=True)
        model_pairs_in_buffers["ml_name_r"] = model_pairs_in_buffers.groupby(
            by=["stress_contribution_group", "buffer_name"], group_keys=False
        ).apply(lambda x: x.ml_name.shift(-1), include_groups=False)
        # drop those with no closer obs bore to stressor
        model_pairs_in_buffers.dropna(subset=["ml_name_r"], inplace=True)
        # separation distance
        model_pairs_in_buffers["geometry_right"] = self.solver.ml_gdf.loc[
            model_pairs_in_buffers.ml_name_r].geometry.values
        model_pairs_in_buffers["separation_distance"] = model_pairs_in_buffers.geometry.distance(
            model_pairs_in_buffers.geometry_right
        )
        # drop bore pairs within self.exclude_sep_distance of one another
        keep_mask = (model_pairs_in_buffers.separation_distance > self.exclude_sep_distance)
        model_pairs_in_buffers = model_pairs_in_buffers.loc[keep_mask].copy()
        model_pairs_in_buffers.set_index(["stress_contribution_group", "buffer_name", "ml_name"], inplace=True)
        # save the data
        self.between_bore_pairs_file = Path(self.solver.model_ws / f"sim_stress_contrib_between_bore_pairs.csv")
        model_pairs_in_buffers.to_csv(self.between_bore_pairs_file)
        copy_file(self.between_bore_pairs_file, self.solver.temp_ws)
        return model_pairs_in_buffers

    @staticmethod
    def dt_series_to_stat(
            dt_series: pd.Series,
            stat: Optional[str | Callable]="max",
    ) -> pd.Series:
        """
        Convert a datetime Series to a Statistic Series.

        Parameters
        ----------
        dt_series: pd.Series
            datetime series with numeric values. Can be multiindexed.
        stat: Optional[str | Callable]
            Pandas statistic keyword or callable function used to calculate the returned statistic.
            Default is 'max'.

        Returns
        -------
        s_stat: pd.Series
            Single element Series with mean date of datetime index
        """
        value_name = dt_series.name
        if isinstance(dt_series.index, pd.MultiIndex):
            dt_index_name = [
                n for n in dt_series.index.names
                if pd.api.types.is_datetime64_any_dtype(dt_series.index.get_level_values(n))
            ][0]
            non_date_idx_names = [n for n in dt_series.index.names if n != dt_index_name]
        else:
            dt_index_name = dt_series.index.name
        s_stat = dt_series.reset_index(drop=False)
        if isinstance(dt_series.index, pd.MultiIndex):
            s_stat = s_stat.groupby(by=non_date_idx_names).agg({dt_index_name: "mean", value_name: stat})
            s_stat = s_stat.reset_index(drop=False).set_index(non_date_idx_names + [dt_index_name]).loc[:, value_name]
        else:
            s_stat["dummy"] = "group"
            s_stat = s_stat.groupby(by="group").agg({dt_index_name: "mean", value_name: stat})
            s_stat = s_stat.reset_index(drop=True).set_index(dt_index_name).loc[:, value_name]
        return s_stat

    @staticmethod
    def get_between_bores_differences(
            between_bore_pairs: pd.DataFrame,
            sm_contribs: pd.DataFrame,
            set_to_zero: bool,
            max_only: Optional[bool] = True,
            stress_contribution_groups: Optional[list | None] = None,
    ) -> pd.Series:
        """

        Parameters
        ----------
        between_bore_pairs: pd.DataFrame
            MultiIndexed by (stress_contribution_group, buffer_name, ml_name); column ml_name_r is the next closest
            model to ml_name and separation_distance is the separation distance between them.
            In other words, adjacent bore pairs within a given model's line buffer from the model to the
            stressor are each [ml_name, ml_name_r] combo.
        sm_contribs: pd.DataFrame
            Stressmodel contributions
        set_to_zero: bool
            Set values to zero (e.g. for observation target creation).
        max_only: Optional[bool]
            Flag whether to include full difference timeseries, or define difference as the difference between maximum
             timeseries values. May expose this option later - can end up with huge obs files...
        stress_contribution_groups: Optional[list | None] = None
            Default is None, in which case all common stress contribution groups are used.

        Returns
        -------
        differences: pd.Series
            Stressmodel contribution differences, MultiIndexed by ('ml_name', 'ml_name_r', 'colnme', 'date')
        """
        penalty_index_names = ['ml_name', 'ml_name_r', 'colnme', 'date']
        differences = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[]] * len(penalty_index_names), names=penalty_index_names)
        )
        for (stress_contribution_group, buffer_name, ml_name), adjacent_bores in between_bore_pairs.groupby(
                level=["stress_contribution_group", "buffer_name", "ml_name"]
        ):
            # ml_name / ml_name_r are the adjacent bore (model) pairs for which a given stress contribution should be
            # less for ml_name than ml_name_r
            for ml_name_r, row in adjacent_bores.set_index("ml_name_r").iterrows():
                # get contribs df indexed by (colnme, date) - .abs() is to overcome pastas stress up vs down flag
                mask = (sm_contribs.model_name == ml_name) & \
                       (sm_contribs.index.get_level_values("colnme") == stress_contribution_group)
                ml_contribs = sm_contribs.loc[mask, "Observations"].abs()
                mask_r = (sm_contribs.model_name == ml_name_r) & \
                       (sm_contribs.index.get_level_values("colnme") == stress_contribution_group)
                ml_contribs_r = sm_contribs.loc[mask_r, "Observations"].abs()
                if max_only:
                    ml_contribs = BetweenStressContribPenalties.dt_series_to_stat(ml_contribs, stat="max")
                    ml_contribs_r = BetweenStressContribPenalties.dt_series_to_stat(ml_contribs_r, stat="max")
                # subtract one from the other and assign to (ml_name, ml_name_r)
                # need to be careful here depending on pastas model stress direction 'up' vs 'down'. Hence .abs() above.
                # here we make a dangerous assumption that all models in solver.models use the same stress direction (up OR down)
                diff = (ml_contribs - ml_contribs_r)  # ml_contribs_r contrib should be > ml_contribs
                diff = pd.concat([diff], keys=[(ml_name, ml_name_r)], names=['ml_name', 'ml_name_r'])
                if set_to_zero:
                    diff.loc[:, :] = 0.0  # reset values to 0.0 - for now we want a target obs diff of < 0.0
                # keep only those ml_name / ml_name_r / colnme / date not already in differences
                drop_mask = diff.index.isin(differences.index.values)
                differences = pd.concat([differences, diff.loc[~drop_mask]], ignore_index=False)
            differences.dropna(subset=["Observations"], inplace=True)  # keep only common stress contribution groups
            if stress_contribution_groups != []:
                differences = differences.loc[
                    differences.index.get_level_values("colnme").isin(stress_contribution_groups), :
                ]
        return differences

    def _calc_differences(self, sm_contribs: pd.DataFrame, set_to_zero: bool=True) -> pd.Series:
        """
        Calculate stress_contribution differences between colocated models (obs bores).

        Parameters
        ----------
        sm_contribs: pd.DataFrame
            Stressmodel contributions
        set_to_zero: bool
            Set values to zero (e.g. for observation target creation). Default is True.

        Returns
        ----------
        differences: pd.Series
            Stressmodel contribution differences, MultiIndexed by ('ml_name', 'ml_name_r', 'colnme', 'date')
        """
        differences = BetweenStressContribPenalties.get_between_bores_differences(
            between_bore_pairs=self.between_bore_pairs, sm_contribs=sm_contribs, set_to_zero=set_to_zero,
            max_only=True, stress_contribution_groups=self.stress_contribution_groups
        )
        differences = sanitise_differences(differences)
        self.penalty_index_names = differences.index.names
        # save the data
        self.penalty_file = Path(self.solver.model_ws / BetweenStressContribPenalties.OUTPUT_PENALTY_FILE)
        differences.to_csv(self.penalty_file, date_format=self.solver.date_format)
        copy_file(self.penalty_file, self.solver.temp_ws)
        return differences

    def make_difference_obs(self) -> pd.DataFrame:
        """
        Makes PEST observation data for stress_contribution differences.

        Returns
        ----------
        pst_from_obs_df: pd.DataFrame
        """
        self.solver.pf.add_observations(
            self.penalty_file.name,
            index_cols=self.penalty_index_names,
            use_cols=["Observations"],
            obsgp=self.obsgp,
        )
        self.solver.pf.obs_dfs[-1]["weight"] = 1.0
        return self.solver.pf.obs_dfs[-1]
