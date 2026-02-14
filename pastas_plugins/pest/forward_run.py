import pyemu
from pandas import DataFrame, Series


def run() -> None:
    # load packages
    from pathlib import Path

    from dill import load as dill_load  # pickle
    from gzip import open as gz_open
    from pandas import read_csv, concat, date_range, DataFrame
    from pandas.tseries.offsets import MonthEnd
    from pastas.io.base import load as load_model

    from pastas_plugins.pest.parameterisers import Parameteriser   # noqa: F401
    import pastas_plugins.pest.obs_penalties as obs_pen
    from pastas_plugins.pest.obs_penalties import (
        ColocatedStressContribPenalties, BetweenStressContribPenalties
    ) # noqa: F401

    # base path
    fpath = Path(__file__).parent

    # load pastas model
    models = [load_model(m) for m in Path(fpath).glob("model_*.pas")]

    # load save_stress_contributions
    save_stress_contributions = False
    sim_stress_contrib_colocated_bores = None
    sim_stress_contrib_between_bore_pairs = None
    if Path("stress_contribution_groups.csv").exists():
        save_stress_contributions = True
        stress_contribution_groups = read_csv("stress_contribution_groups.csv", index_col=[0,1])
        if Path(f"sim_stress_contrib_colocated_bores.csv").exists():
            sim_stress_contrib_colocated_bores = read_csv(
                "sim_stress_contrib_colocated_bores.csv", index_col=["ml_name", "ml_name_r"]
            )
        if Path(f"sim_stress_contrib_between_bore_pairs.csv").exists():
            sim_stress_contrib_between_bore_pairs = read_csv(
                "sim_stress_contrib_between_bore_pairs.csv",
                index_col=["stress_contribution_group", "buffer_name","ml_name"],
            )

    # update standard pastas model parameters
    parameters = read_csv(fpath / "parameters_sel.csv", index_col=0)
    for ml in models:
        ml_code = ml.oseries.metadata["ml_code"]
        for pname, val in parameters.loc[:, "optimal"].items():
            pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
            if pname[len(ml_code):] in ml.parameters.index.values and ml_code == pname[:len(ml_code)]:
                ml.set_parameter(pname[len(ml_code):], optimal=val)

    # update custom stressmodel parameters
    pickles = Path(fpath).glob("*.parameteriser.pkl.gz")
    stressmodel_parameterisers = [
        dill_load(gz_open(sm_p)) for sm_p in pickles
    ]  # pickle.load(
    for sm_p in stressmodel_parameterisers:
        # update stress TimeSeries
        for ml in models:
            sm_snames = {sm_name: DataFrame(ml.stressmodels.get(sm_name).get_stress(squeeze=False)).columns for sm_name in ml.stressmodels}
            for sm_name, snames in sm_snames.items():
                smodel = ml.stressmodels.get(sm_name)
                do_update = (ml.name in sm_p.model_names) & (sm_name in sm_p.stressmodel_names) & (smodel is not None)
                snames_to_update = [sname for sname in snames if sname in sm_p.stress_names]
                if (len(snames_to_update) > 0) & do_update:
                    # get df of updated (parameterised and interpolated) stress TimeSeries for model
                    updated_stress_df = sm_p.interpolate_stresses(**sm_p.interp_kwargs)
                    for stress_series in smodel.stress:
                        if stress_series.name in sm_p.stress_names:
                            stress_series.series_original = updated_stress_df.loc[
                                :, stress_series.name
                            ]
                    ml.stressmodels[sm_name] = smodel
    # ^^ one sm_p even for many pastas models in one pest cal will work ok - we just update the stress rates,
    # while pumping well distances from each model (obs bore) remain as originally defined per model.
    # Pest-calibrated rates are the same across all pastas models, but distances of q wells from obs bores vary. Yay.

    def _get_monthend_interpolant(ml, data):
        """Interpolate from one datetime-indexed series or df to another at monthend intervals"""
        smp_index = date_range(
            ml.settings["tmin"], ml.settings["tmax"] + MonthEnd(0), freq="ME"
        )
        data = data.reindex(data.index.union(smp_index)).interpolate(method="time")
        data = data.loc[smp_index]
        return data

    # simulate
    stress_obs_done = []  # This is a list of stress obs indices we have already processed in an earlier pastas model in the below loop.
    # We only want to process stress_obs once, not repeatedly for every model (the same stresses (stress Series names) may be reused across all models)
    for ml in models:
        ml_name = ml.name
        simulation = ml.simulate()
        sim_obs_idx = simulation.index.union(ml.observations().index)
        sim_obs = simulation.reindex(sim_obs_idx).interpolate(method="time")
        sim_obs = sim_obs.loc[ml.observations().index]
        sim_obs.to_csv(fpath / f"simulation_{ml_name}.csv", date_format="%d/%m/%Y", float_format='%.16f')

        # save head_diffs too
        head_diffs = (sim_obs - sim_obs.shift().values).dropna()
        head_diffs.to_csv(fpath / f"simulation_head_diffs_{ml_name}.csv", date_format="%d/%m/%Y", float_format='%.16f')

        # smp-style zero-weight obs
        sim_smp = _get_monthend_interpolant(ml, simulation)
        sim_smp.to_csv(fpath / f"simulation_{ml_name}.smp.csv", date_format="%d/%m/%Y", float_format='%.16f')

        # stress obs
        for sm_p in stressmodel_parameterisers:
            if (sm_p.obs_data is not None) and (ml_name in sm_p.model_names) and \
                    (sm_p.parameteriser_name not in stress_obs_done):
                stress_mod = sm_p.mod2obs()
                stress_mod.to_csv(f"{sm_p.parameteriser_name}.stress_obs.csv", date_format=sm_p.date_format, float_format='%.16f')
                stress_obs_done += [sm_p.parameteriser_name]  # one set of obs per parameteriser

        # stress contributions
        if save_stress_contributions:
            contribs_all = ml.get_contributions(split=True) # all contributions
            contribs_all = [_get_monthend_interpolant(ml, s) for s in contribs_all] # downsample from daily. Should make this an option...
            contribs_all = concat(contribs_all, axis=1, ignore_index=False)
            for label, istress_names in stress_contribution_groups.loc[(ml_name, slice(None)), :].groupby(level=1):
                names = istress_names.istress_names.values.flatten()
                # aggregate selected groups of istress contributions
                contribs_all.loc[:,label] = contribs_all.loc[:, names].sum(axis=1)
            # drop unspecific istress_names / labels from the df
            if (stress_contribution_groups.xs(ml_name).save_all == False).any():
                contribs_all = contribs_all.loc[:,
                contribs_all.columns.isin(stress_contribution_groups.xs(ml_name).index.get_level_values("label"))
                ]
            contribs_all.index.name = "date"
            contribs_all = contribs_all.reset_index(drop=False).melt(
                id_vars="date",
                value_vars=contribs_all.columns,
                var_name="colnme",
                value_name="Observations",
            ).set_index(["colnme","date"])
            contribs_all.to_csv(fpath / f"sim_stress_contribs_{ml_name}.csv", date_format="%d/%m/%Y", float_format='%.16f')

            # stress contribution penalties
            if sim_stress_contrib_colocated_bores is not None:
                colocated_differences = ColocatedStressContribPenalties.get_colocated_differences(
                    colocated_bores=sim_stress_contrib_colocated_bores,
                    sm_contribs=contribs_all,
                    set_to_max_difference_percent=False,
                    max_difference_percent=sim_stress_contrib_colocated_bores.iloc[0].max_difference_percent # future upgrades might allow different max diffs per bore
                )
                colocated_differences = obs_pen.sanitise_differences(colocated_differences)
                colocated_differences.to_csv(
                    ColocatedStressContribPenalties.OUTPUT_PENALTY_FILE, date_format="%d/%m/%Y", float_format='%.16f'
                )
            if sim_stress_contrib_between_bore_pairs is not None:
                between_bore_differences = BetweenStressContribPenalties.get_between_bores_differences(
                    between_bore_pairs=sim_stress_contrib_between_bore_pairs, 
                    sm_contribs=contribs_all, set_to_zero=False,
                )
                between_bore_differences = obs_pen.sanitise_differences(between_bore_differences)
                between_bore_differences.to_csv(
                    BetweenStressContribPenalties.OUTPUT_PENALTY_FILE, date_format="%d/%m/%Y", float_format='%.16f'
                )

def run_pypestworker(
    pst: str | pyemu.Pst,
    host: int,
    port: int,
    timeout: float,
    models: dict,
    parameter_index: dict,
    observation_index: DataFrame,
    stressmodel_parameterisers: list = [],
    stress_obs: DataFrame | None = None,
    save_stress_contributions: bool = False,
    stress_contribution_groups: Series | None = None,
    sim_stress_contrib_colocated_bores: DataFrame | None = None,
    sim_stress_contrib_between_bore_pairs: DataFrame | None = None,
) -> None:
    from logging import getLogger

    from pandas import concat, date_range
    from pandas.tseries.offsets import MonthEnd
    from pastas_plugins.pest.parameterisers import Parameteriser  # noqa: F401
    from pastas_plugins.pest.obs_penalties import (
        ColocatedStressContribPenalties, BetweenStressContribPenalties, sanitise_differences
    )  # noqa: F401

    ppw = pyemu.os_utils.PyPestWorker(
        pst=pst,
        host=host,
        port=port,
        timeout=timeout,
        verbose=False,
    )

    pvals = ppw.get_parameters()
    if pvals is None:
        return None

    def _get_monthend_interpolant(ml, data):
        """Interpolate from one datetime-indexed series or df to another at monthend intervals"""
        smp_index = date_range(
            ml.settings["tmin"], ml.settings["tmax"] + MonthEnd(0), freq="ME"
        )
        data = data.reindex(data.index.union(smp_index)).interpolate(method="time")
        data = data.loc[smp_index]
        return data

    while True:

        obsvals_list, obs_diffs_list, stress_obs_list, headsmp_list, contribs_all_list = [], [], [], [], []
        head_obsgps = [
            og for og in observation_index.index.get_level_values("obgnme").unique() \
            if og.find("head_") >= 0
        ]
        headsmp_obsgps = [
            og for og in observation_index.index.get_level_values("obgnme").unique() \
            if og.find("headsmp_") >= 0
        ]
        head_diff_obsgps = [
            og for og in observation_index.index.get_level_values("obgnme").unique() \
            if og.find("headdiff") >= 0
        ]
        stress_obs_done = [] # This is a list of stress obs indices we have already processed in an earlier pastas model in the below loop.
        # We only want to process stress_obs once, not repeatedly for every model (the same stresses (stress Series names) may be reused across all models)
        for ml_name, ml in models.items():
            #ml.settings["tmax"] = None
            ml_code = ml.oseries.metadata["ml_code"]
            # update standard pastas model parameters
            for pname, val in pvals.items():
                pname = parameter_index[pname]
                pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
                if pname[len(ml_code):] in ml.parameters.index.values and pname[:len(ml_code)] == ml_code:
                    ml.set_parameter(pname[len(ml_code):], optimal=val)
            # update stress TimeSeries
            for sm_p in stressmodel_parameterisers:
                sm_snames = {sm_name: DataFrame(ml.stressmodels.get(sm_name).get_stress(squeeze=False)).columns for sm_name in ml.stressmodels}
                for sm_name, snames in sm_snames.items():
                    smodel = ml.stressmodels.get(sm_name)
                    do_update = (ml.name in sm_p.model_names) & (sm_name in sm_p.stressmodel_names) & (smodel is not None)
                    snames_to_update = [sname for sname in snames if sname in sm_p.stress_names]
                    if (len(snames_to_update) > 0) & do_update:
                        sm_p_parnames = sm_p.stress_pars.parnme
                        new_par_values = pvals.loc[sm_p_parnames].values
                        # get df of updated (parameterised and interpolated) stress TimeSeries for model
                        interp_kwargs = sm_p.interp_kwargs
                        interp_kwargs["updated_sourcevals"] = new_par_values
                        updated_stress_df = sm_p.interpolate_stresses(**interp_kwargs)
                        for stress_series in smodel.stress:
                            if stress_series.name in sm_p.stress_names:
                                stress_series.series_original = updated_stress_df.loc[
                                    :, stress_series.name
                                ]
                        ml.stressmodels[sm_name] = smodel

                        # stress obs
                        stress_obs_rates = stress_obs.loc[stress_obs.obs_type == "stress_obs"]
                        if (not stress_obs_rates.empty) and (sm_p.obs_data is not None):
                            stress_mod = sm_p.mod2obs()
                            # reindex with pest obsnme
                            obsnmes = stress_obs_rates.loc[stress_mod.index].obsnme
                            stress_mod.index = obsnmes.values
                            stress_mod = stress_mod.loc[~stress_mod.index.isin(stress_obs_done)]  # avoid repeated processing of obs
                            # store the series
                            if not stress_mod.empty:
                                stress_obs_list.append(stress_mod)
                                stress_obs_done += stress_mod.index.to_list()

            # run simulation
            sim = ml.simulate()

            # get head obs
            obs = observation_index.xs(ml.name)
            obs = obs.loc[obs.index.get_level_values("obgnme").isin(head_obsgps)].droplevel("obgnme") # xs-->df indexed by date. Values are just obsnme
            sim_obs_idx = sim.index.union(obs.index)
            sim_obs = sim.reindex(sim_obs_idx).interpolate(method="time")
            obsvals = sim_obs.loc[obs.index.values]
            sim_obs = None
            # save head_diffs in case needed below (before we replace datetime index with obsnme index)
            head_diffs = (obsvals - obsvals.shift().values).dropna()
            onames = obs.obsnme
            obsvals.index = onames
            obsvals_list.append(obsvals)

            # head difference from first head obs
            if len(head_diff_obsgps) > 0:
                obs_diffs = observation_index.xs(ml.name)  # xs-->df indexed by [obgnme,date]. Values are just obsnme
                head_diff_obsgps2 = [og for og in head_diff_obsgps if og in obs_diffs.index.get_level_values("obgnme")]
                obs_diffs = obs_diffs.loc[
                    obs_diffs.index.get_level_values("obgnme").isin(head_diff_obsgps2)
                ].droplevel("obgnme")  # -->df indexed by date. Values are just obsnme
                onames = obs_diffs.obsnme
                head_diffs.index = onames
                obs_diffs_list.append(head_diffs)

            # smp-style zero-weight obs
            sim_dateidx_obsnme = observation_index.xs(ml.name).loc[
                observation_index.xs(ml.name).index.get_level_values("obgnme").isin(headsmp_obsgps)
            ].droplevel("obgnme") # xs-->df indexed by date. Values are just obsnme
            sim_smp = _get_monthend_interpolant(ml, sim)
            sim_smp_vals = (sim_smp.loc[sim_dateidx_obsnme.index])
            sim_smp_vals.index = sim_dateidx_obsnme.obsnme
            headsmp_list.append(sim_smp_vals)

            # stress contributions
            if save_stress_contributions:
                stress_obs_contribs = stress_obs.loc[
                    (stress_obs.obs_type=="stress_contribution") & \
                    (stress_obs.model_name == ml.name)
                    ]
                contribs_all = ml.get_contributions(split=True)  # all contributions
                contribs_all = [_get_monthend_interpolant(ml, s) for s in contribs_all]  # reindex to monthend via time interp. Should make this an option...
                contribs_all = concat(contribs_all, axis=1, ignore_index=False)
                for label, istress_names in stress_contribution_groups.loc[(ml_name, slice(None)), :].groupby(level=1):
                    names = istress_names.istress_names.values.flatten()
                    # aggregate selected groups of istress contributions
                    contribs_all.loc[:, label] = contribs_all.loc[:, names].sum(axis=1)
                # drop unspecified istress_names / labels from the df
                if (stress_contribution_groups.xs(ml_name).save_all == False).any():
                    contribs_all = contribs_all.loc[:,
                    contribs_all.columns.isin(stress_contribution_groups.xs(ml_name).index.get_level_values("label"))
                    ]
                # melt from xtab to flat array and save
                contribs_all.index.name = "date"
                contribs_all = contribs_all.reset_index(drop=False).melt(
                    id_vars="date",
                    value_vars=contribs_all.columns,
                    var_name="colnme",
                    value_name="Observations",
                ).set_index(["colnme", "date"])
                obsnmes = stress_obs_contribs.loc[contribs_all.index].obsnme
                contribs_all.index = obsnmes.values
                # store the series
                contribs_all_list.append(contribs_all.Observations)

                # stress contribution penalties
                if sim_stress_contrib_colocated_bores is not None:
                    colocated_differences = ColocatedStressContribPenalties.get_colocated_differences(
                        colocated_bores=sim_stress_contrib_colocated_bores,
                        sm_contribs=contribs_all,
                        set_to_max_difference_percent=False,
                        max_difference_percent=sim_stress_contrib_colocated_bores.iloc[0].max_difference_percent # future upgrades might allow different max diffs per bore
                    ).loc[:,"Observations"]
                    colocated_differences = sanitise_differences(colocated_differences)
                    # TODO IMPLEMENT THIS OBSNME BIT
                    #obsnmes = stress_obs_contribs.loc[contribs_all.index].obsnme
                    #colocated_differences.index = obsnmes.values
                if sim_stress_contrib_between_bore_pairs is not None:
                    between_bore_differences = BetweenStressContribPenalties.get_between_bores_differences(
                        between_bore_pairs=sim_stress_contrib_between_bore_pairs,
                        sm_contribs=contribs_all, set_to_zero=False,
                    ).loc[:,"Observations"]
                    between_bore_differences = sanitise_differences(between_bore_differences)
                    # TODO IMPLEMENT THIS OBSNME BIT
                    #obsnmes = stress_obs_contribs.loc[contribs_all.index].obsnme
                    #colocated_differences.index = obsnmes.values

        obsvals_all = concat(obsvals_list, axis=0, ignore_index=False)
        sim_smp_vals = concat(headsmp_list, axis=0, ignore_index=False)
        obsvals_all = concat([obsvals_all, sim_smp_vals], axis=0, ignore_index=False)
        if len(obs_diffs_list) > 0:
            obs_diffs = concat(obs_diffs_list, axis=0, ignore_index=False)
            obsvals_all = concat([obsvals_all, obs_diffs], axis=0, ignore_index=False)
        if len(stress_obs_list) > 0:
            stress_obs_all = concat(stress_obs_list, axis=0, ignore_index=False)
            obsvals_all = concat([obsvals_all, stress_obs_all], axis=0, ignore_index=False)
        if len(contribs_all_list) > 0:
            contribs_all = concat(contribs_all_list, axis=0, ignore_index=False)
            obsvals_all = concat([obsvals_all, contribs_all], axis=0, ignore_index=False)

        ppw.send_observations(obsvals=obsvals_all)

        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    run()
