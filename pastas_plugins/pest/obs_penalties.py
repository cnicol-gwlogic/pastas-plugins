import logging
import numpy as np
import pandas as pd
import geopandas as gpd

from typing import Optional
from shutil import copy as copy_file
from pathlib import Path
from pydantic.dataclasses import dataclass
from shapely.geometry import Point, LineString
from pastas_plugins.pest.solver import PestSolver


logger = logging.getLogger(__name__)

@dataclass
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
    between_penalty_obs: bool
        Default is True
    between_penalty_stress_contribution_groups: list
        Stress contribution groups for which penalties are applied based on the group's well locations' centroid.
        All names must be in the column_names field of solver.sm_contribs.
        Default is an empty list.
    between_penalty_max_distance_from_connecting_line: float
        Default is 1000.0
    between_penalty_obs_phi_factor: float
        Default is 0.1
    solver: Optional[PestSolver | None]
        Default is None
    """
    colocated_penalty_obs: Optional[bool] = True
    colocated_penalty_max_separation_distance: Optional[float] = 250.0
    colocated_penalty_max_difference_percent: Optional[float] = 10.0
    colocated_penalty_obs_phi_factor: Optional[float] = 0.1
    between_penalty_obs: Optional[bool] = True
    between_penalty_stress_contribution_groups: Optional[list] = []
    between_penalty_max_distance_from_connecting_line: Optional[float] = 1000.0
    between_penalty_obs_phi_factor: Optional[float] = 0.1
    solver: Optional[PestSolver | None] = None
    if colocated_penalty_obs or between_penalty_obs:
        logger.warning(
            "User beware! stress_contribution_penalty_obs require that model.oseries.metadata['x'] "
            "and model.oseries.metadata['y'] must be provided for all Pastas models"
        )

class ColocatedStressContribPenalties:
    """
    Make PEST penalty obs for stress contribution differences between colocated models (obs bores).
    Forward run makes these penalty obs after all pastas (obs bore) models are run.
    Initial obs values are colocated_penalty_max_difference_percent, and they are less_than type.
    """
    def __init__(
            self,
            settings: StressContribPenaltySettings
    ):
        self.solver = settings.solver
        self.solver.assign_model_coords(force_update=True)
        self.max_separation_distance = settings.colocated_penalty_max_separation_distance
        self.max_difference_percent = settings.colocated_penalty_max_difference_percent

        self.colocated_bores = self._get_colocated_bores(models=self.solver.models)
        self.solver.colocated_bores = self.colocated_bores
        self.difference_obs = self._calc_differences(self.solver.sm_contribs)

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
        nearest = gpd.sjoin_nearest(
            ml_gdf, ml_gdf, how='inner',
            lsuffix="", rsuffix="_r",
            max_distance=self.max_separation_distance
        )
        colocated_bores = nearest.loc[nearest.ml_name != nearest["ml_name_r"]]
        colocated_bores.set_index(["ml_name", "ml_name_r"], inplace=True)

        return colocated_bores

    def _calc_differences(self, sm_contribs: pd.DataFrame, set_to_max_difference_percent: bool=True) -> pd.Series:
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
            Stressmodel contribution differences, MultiIndexed by ('ml_name', 'ml_name_r', 'column_names', 'date')
        """
        self.penalty_index_names = ['ml_name', 'ml_name_r', 'column_names', 'date']
        differences = pd.DataFrame(index=pd.MultiIndex.from_arrays([[]]*4, names=self.penalty_index_names))
        for ml_name, colocated_bores in self.colocated_bores.groupby(level='ml_name'):
            for ml_name_r, df in colocated_bores:
                # get contribs df indexed by (column_names, date)
                ml_contribs = sm_contribs.loc[sm_contribs.model_name == ml_name, "Observations"]
                ml_contribs_r = sm_contribs.loc[sm_contribs.model_name == ml_name_r, "Observations"]
                # subtract one from the other and assign to (ml_name, ml_name_r)
                if (ml_name_r, ml_name) not in differences.index:
                    diff = (ml_contribs - ml_contribs_r)
                    # convert to % of larger contribution
                    max = ml_contribs.combine(ml_contribs_r, np.maximum, fill_value=np.nan)
                    diff = (diff.abs() / max).dropna() * 100.0 # na() entries are uncommon stress contribution names (column_names)
                    diff = pd.concat([diff], keys=[(ml_name, ml_name_r)], names=['ml_name', 'ml_name_r'])
                    if set_to_max_difference_percent:
                        diff.loc[:,:] = self.max_difference_percent # reset values to max_difference_percent - for now we want a target obs diff of < max_difference_percent
                    differences = pd.concat([differences, diff], ignore_index=False)
        # save the data
        self.penalty_file = Path(self.solver.model_ws / f"sim_stress_contrib_colocated_penalties.csv")
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
        obsgp = f"less_than_stress_contrib_penalty_colocated"
        self.solver.pf.add_observations(
            self.penalty_file.name,
            index_cols=self.penalty_index_names,
            use_cols=["Observations"],
            obsgp=obsgp,
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
    """
    def __init__(
            self,
            settings: StressContribPenaltySettings
    ):
        self.solver = settings.solver
        self.solver.assign_model_coords(force_update=True)
        self.solver.assign_stressmodel_coords(force_update=True)

        self.stress_contribution_groups = settings.between_penalty_stress_contribution_groups
        self.max_distance_from_connecting_line = settings.between_penalty_max_distance_from_connecting_line
        self.exclude_sep_distance = settings.colocated_penalty_max_separation_distance

        self.sm_contrib_group_centroids = self._get_sm_contrib_group_centroid()
        self.between_bores = self._get_between_bores(models=self.solver.models)
        self.solver.colocated_bores = self.colocated_bores
        self.difference_obs = self._calc_differences(self.solver.sm_contribs)

    def _get_sm_contrib_group_centroids(
            self,
    ) -> gpd.GeoDataFrame:
        """
        Get centroids of all between_penalty_stress_contribution_groups for every model in PestSolver.models

        Returns
        -------
        sm_contrib_group_centroids: gpd.GeoDataFrame
            MultiIndexed by ["stress_contribution_group", "ml_name"], with values being a Point of the
            centroid of the given stress_contribution_group for the given model (ml_name).
        """
        mux = index=pd.MultiIndex.from_product([[]]*2, names=["stress_contribution_group", "ml_name"])
        sm_contrib_group_centroids = pd.DataFrame(mux)
        for stress_contribution_group in self.stress_contribution_groups:
            ml_sm_contrib_centroids = {}
            for ml_name,ml in self.solver.models.items():
                contrib_group_istresses = self.solver.stress_contribution_groups.loc[
                    (ml_name, stress_contribution_group)
                ].istress_names.values
                coords = self.solver.sm_gdf.loc[
                    (ml_name, slice(None), contrib_group_istresses)
                ].reset_index(drop=False).dissolve(by="istress_names", aggfunc='mean') # sm_df has a block of istress_names for every model/stressmodel)
                sm_contrib_group_centroids.loc[(stress_contribution_group, ml_name), "geometry"] = coords.centroid

        return gpd.GeoDataFrame(sm_contrib_group_centroids, geometry="geometry")

    def _get_between_bores(self, stress_contribution_group: str) -> gpd.GeoDataFrame:
        """
        For a given stress_contribution_group, find obs bores (models) within max_distance_from_connecting_line
        to specified stressor(s) for every model (obs bore) in the pestsolver.

        Parameters
        ----------
        stress_contribution_group: str

        Returns
        -------
        models_in_buffers: gpd.GeoDataFrame
            MultiIndexed by (buffer_name, ml_name) with values being distance_to_sm_contrib, sm_contrib_centroid, and
            geometry (Point of ml_loc).
        """
        sm_contrib_centroids = self.sm_contrib_group_centroids.xs(stress_contribution_group).geometry # indexed by ml_name
        ml_locs = self.solver.ml_gdf.geometry # indexed by ml_name
        lines = ml_locs.shortest_line(sm_contrib_centroids, align=True)
        buffers = lines.buffer(self.max_distance_from_connecting_line)
        buffers["sm_contrib_centroid"] = sm_contrib_centroids.geometry
        buffers["distance_to_sm_contrib"] = lines.geometry.length
        buffers["buffer_name"] = buffers.index
        models_in_buffers = gpd.sjoin(
            gpd.GeoDataFrame(data=dict(ml_name=self.solver.ml_gdf.index.values), geometry=ml_locs),
            gpd.GeoDataFrame(data=buffers, geometry="geometry"),
            how="inner", predicate="within")
        models_in_buffers.set_index(["buffer_name", "ml_name"], inplace=True) # can have lots of models in a given buffer
        return models_in_buffers

    def _get_between_bore_pairs(
            self,
            models_in_buffers: gpd.GeoDataFrame
    ):
        """
        Identify bore pairs within each buffer polygon (around line connecting model with sm_contrib centroid).
        Bore pairs are defined sequentially: starting from most distal model, find the next most distal model (that is
        further from the model than self.exclude_sep_distance).
        Note that models_in_buffers relates to a single stress contribution group, so this function is expected to be
        called as many times as there are stress contribution groups.

        Parameters
        ----------
        models_in_buffers: gpd.GeoDataFrame
            MultiIndexed by (buffer_name, ml_name) with values being distance_to_sm_contrib, sm_contrib_centroid, and
            geometry (Point of ml_loc).

        Returns
        -------
        model_pairs_in_buffers: gpd.GeoDataFrame
            MultiIndexed by (buffer_name, ml_name) with values being ml_name_r (the next closest model)
        """
        # TODO FINISH THIS


    def _calc_differences(self, sm_contribs: pd.DataFrame, set_to_max_difference_percent: bool=True) -> pd.Series:
        """
        Calculate stress_contribution differences between colocated models (obs bores).

        Parameters
        ----------
        sm_contribs: pd.DataFrame
            Stressmodel contributions
        set_to_max_difference_percent: bool
            Set values to max_difference_percent (e.g. for observation target creation). Default is True.

        Returns
        ----------
        differences: pd.Series
            Stressmodel contribution differences, MultiIndexed by ('ml_name', 'ml_name_r', 'column_names', 'date')
        """
        self.penalty_index_names = ['ml_name', 'ml_name_r', 'column_names', 'date']
        differences = pd.DataFrame(index=pd.MultiIndex.from_arrays([[]]*4, names=self.penalty_index_names))
        for ml_name, colocated_bores in self.colocated_bores.groupby(level='ml_name'):
            for ml_name_r, df in colocated_bores:
                # get contribs df indexed by (column_names, date)
                ml_contribs = sm_contribs.loc[sm_contribs.model_name == ml_name, "Observations"]
                ml_contribs_r = sm_contribs.loc[sm_contribs.model_name == ml_name_r, "Observations"]
                # subtract one from the other and assign to (ml_name, ml_name_r)
                if (ml_name_r, ml_name) not in differences.index:
                    diff = (ml_contribs - ml_contribs_r)
                    # convert to % of larger contribution
                    max = ml_contribs.combine(ml_contribs_r, np.maximum, fill_value=np.nan)
                    diff = (diff.abs() / max).dropna() * 100.0 # na() entries are uncommon stress contribution names (column_names)
                    diff = pd.concat([diff], keys=[(ml_name, ml_name_r)], names=['ml_name', 'ml_name_r'])
                    if set_to_max_difference_percent:
                        diff.loc[:,:] = self.max_difference_percent # reset values to max_difference_percent - for now we want a target obs diff of < max_difference_percent
                    differences = pd.concat([differences, diff], ignore_index=False)
        # save the data
        self.penalty_file = Path(self.solver.model_ws / f"sim_stress_contrib_colocated_penalties.csv")
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
        obsgp = f"less_than_stress_contrib_penalty_colocated"
        self.solver.pf.add_observations(
            self.penalty_file.name,
            index_cols=self.penalty_index_names,
            use_cols=["Observations"],
            obsgp=obsgp,
        )
        self.solver.pf.obs_dfs[-1]["weight"] = 1.0
        return self.solver.pf.obs_dfs[-1]
