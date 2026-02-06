import numpy as np
import pandas as pd
import geopandas as gpd
from shutil import copy as copy_file
from pathlib import Path
from pastas_plugins.pest.solver import PestSolver, PestGlmSolver


class ColocatedStressContribPenalties():
    """
    Forward run makes these penalty obs after all pastas (obs bore) models are run.
    Initial obs values are zero, and they are greater_than
    """
    def __init__(
            self,
            solver: PestSolver,
            max_separation_distance=0.0,
            max_difference_percent=10.0,
    ):
        self.solver = solver
        self.max_separation_distance = max_separation_distance
        self.max_difference_percent = max_difference_percent
        self.colocated_bores = self._get_collocated_bores(models=solver.models)
        solver.colocated_bores = self.colocated_bores
        self.difference_obs = self._calc_differences(solver.sm_contribs)

    def _get_collocated_bores(self, models: dict) -> gpd.GeoDataFrame:
        """
        Find obs bores (models) within max_separation_distance of every model (obs bore) in the pestsolver.

        Parameters
        ----------
        models: dict
            PestSolver.models dictionary

        Returns
        -------
        collocated_bores: gpd.GeoDataFrame
            Multi-indexed by (ml_name, ml_name_r) with values being geometry and near geometry (geometry_r).
            "ml_name_r" is the colocated (nearby, within max_separation_distance of ml_name) obs bore.
        """
        self.solver._assign_model_coords()
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
        self.pf.obs_dfs[-1]["weight"] = 1.0
        return self.pf.obs_dfs[-1]
