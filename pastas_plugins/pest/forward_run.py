import pyemu
from pandas import DataFrame, Series


def run() -> None:
    # load packages
    from pathlib import Path

    from dill import load as dill_load  # pickle
    from gzip import open as gz_open
    from pandas import read_csv, concat
    from pastas.io.base import load as load_model

    from pastas_plugins.pest.parameterisers import (  # noqa: F401
        BaseParameteriser,
        WellModelParameteriser,
    )

    # base path
    fpath = Path(__file__).parent

    # load pastas model
    models = [load_model(m) for m in Path(fpath).glob("model_*.pas")]

    # load save_stress_contributions
    save_stress_contributions = False
    if Path("stress_contribution_groups.csv").exists():
        save_stress_contributions = True
        stress_contribution_groups = read_csv("stress_contribution_groups.csv", index_col=[0,1,2])

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
            smodel = ml.stressmodels.get(sm_p.stressmodel_name)
            if smodel is not None:
                # get df of updated (parameterised and interpolated) stress TimeSeries for model
                updated_stress_df = sm_p.interpolate_stresses(**sm_p.interp_kwargs)
                for stress_series in smodel.stress:
                    if stress_series.name in sm_p.stress_names:
                        stress_series.series_original = updated_stress_df.loc[
                            :, stress_series.name
                        ]
                ml.stressmodels[sm_p.stressmodel_name] = smodel
    # ^^ one sm_p even for many pastas models in one pest cal will work ok - we just update the stress rates,
    # while pumping well distances from each model (obs bore) remain as originally defined per model.
    # Pest-calibrated rates are the same across all pastas models, but distances of q wells from obs bores vary. Yay.

    # simulate
    for ml in models:
        ml_name = ml.name
        #ml.settings["tmax"] = None
        simulation = ml.simulate(
            #tmin=ml.get_tmin(tmin=None, use_oseries=False, use_stresses=True),
            #tmax=ml.get_tmax(tmax=None, use_oseries=False, use_stresses=True),
        )
        simulation.loc[ml.observations().index].to_csv(fpath / f"simulation_{ml_name}.csv", date_format="%d/%m/%Y")

        # save head_diffs too
        head_diffs = (simulation - simulation.shift().values).dropna()
        head_diffs.to_csv(fpath / f"simulation_head_diffs_{ml_name}.csv", date_format="%d/%m/%Y")

        # smp-style zero-weight obs
        sim_smp = simulation.resample("ME").mean()
        sim_smp.to_csv(fpath / f"simulation_{ml_name}.smp.csv", date_format="%d/%m/%Y")

        # stress obs
        for sm_p in stressmodel_parameterisers:
            if (sm_p.obs_data is not None) and (sm_p.model.name == ml_name):
                stress_mod = sm_p.mod2obs()
                stress_mod.to_csv(f"{sm_p.stressmodel_name}.stress_obs.csv", date_format=sm_p.date_format)

        # stress contributions
        if save_stress_contributions:
            contribs_all = ml.get_contributions(
                split=True,
                #tmin=ml.get_tmin(tmin=None, use_oseries=False, use_stresses=True),
                #tmax=ml.get_tmax(tmax=None, use_oseries=False, use_stresses=True),
            ) # all contributions
            contribs_all = [s.resample("ME").mean() for s in contribs_all] # downsample from daily. Should make this an option...
            contribs_all = concat(contribs_all, axis=1, ignore_index=False)
            ml_stress_groups = stress_contribution_groups.xs(ml_name)
            for sm_name, istress_groups in ml_stress_groups.groupby(level="sm_name"):
                istress_groups = ml_stress_groups.xs(sm_name)
                for label, istress_names in istress_groups.groupby(level=0):
                    names = istress_groups.xs(label)[["istress_names"]].values.flatten()
                    # aggregate selected groups of istress contributions
                    contribs_all.loc[:,label] = contribs_all.loc[:, names].sum(axis=1)
            # drop unspecific istress_names / labels from the df
            if (stress_contribution_groups.xs(ml_name).save_all == False).any():
                contribs_all = contribs_all.loc[:,
                contribs_all.columns.isin(ml_stress_groups.index.get_level_values("label"))
                ]
            contribs_all.index.name = "date"
            contribs_all = contribs_all.reset_index(drop=False).melt(
                id_vars="date",
                value_vars=contribs_all.columns,
                var_name="column_names",
                value_name="Observations",
            ).set_index(["column_names","date"])
            contribs_all.to_csv(fpath / f"simulation_stress_contributions_{ml_name}.csv", date_format="%d/%m/%Y")

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
) -> None:
    from logging import getLogger

    from pastas_plugins.pest.parameterisers import (  # noqa: F401
        BaseParameteriser,
        WellModelParameteriser,
    )
    from pandas import concat

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
                smodel = ml.stressmodels.get(sm_p.stressmodel_name)
                if smodel is not None:
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
                    ml.stressmodels[sm_p.stressmodel_name] = smodel
            # run simulation
            sim = ml.simulate(
                #tmin=ml.get_tmin(tmin=None, use_oseries=False, use_stresses=True),
                #tmax=ml.get_tmax(tmax=None, use_oseries=False, use_stresses=True),
            )

            # get head obs
            obs = observation_index.xs(ml.name)
            obs = obs.loc[obs.index.get_level_values("obgnme").isin(head_obsgps)].droplevel("obgnme") # xs-->df indexed by date. Values are just obsnme
            obsvals = sim.loc[obs.index.values]
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
            sim_smp = sim.resample("ME").mean()
            sim_smp_vals = (sim_smp.loc[sim_dateidx_obsnme.index])
            sim_smp_vals.index = sim_dateidx_obsnme.obsnme
            headsmp_list.append(sim_smp_vals) #f"simulation_{ml_name}.smp.csv"

            # stress obs
            stress_obs_rates = stress_obs.loc[stress_obs.obs_type == "stress_obs"]
            if stress_obs is not None:
                for sm_p in stressmodel_parameterisers:
                    if (sm_p.obs_data is not None) and (sm_p.model.name == ml_name):
                        stress_mod = sm_p.mod2obs()
                        # reindex with pest obsnme
                        obsnmes = stress_obs_rates.loc[stress_mod.index].obsnme
                        stress_mod.index = obsnmes.values
                        # store the series
                        stress_obs_list.append(stress_mod)

            # stress contributions
            if save_stress_contributions:
                stress_obs_contribs = stress_obs.loc[
                    (stress_obs.obs_type=="stress_contribution") & \
                    (stress_obs.model_name == ml.name)
                    ]
                contribs_all = ml.get_contributions(
                    split=True,
                    #tmin=ml.get_tmin(tmin=None, use_oseries=False, use_stresses=True),
                    #tmax=ml.get_tmax(tmax=None, use_oseries=False, use_stresses=True),
                )  # all contributions
                contribs_all = [s.resample("ME").mean() for s in
                                contribs_all]  # downsample from daily. Should make this an option...
                contribs_all = concat(contribs_all, axis=1, ignore_index=False)
                ml_stress_groups = stress_contribution_groups.xs(ml_name)
                for sm_name, istress_groups in ml_stress_groups.groupby(level="sm_name"):
                    istress_groups = ml_stress_groups.xs(sm_name)
                    for label, istress_names in istress_groups.groupby(level=0):
                        names = istress_groups.xs(label)[["istress_names"]].values.flatten()
                        # aggregate selected groups of istress contributions
                        contribs_all.loc[:, label] = contribs_all.loc[:, names].sum(axis=1)
                # drop unspecific istress_names / labels from the df
                if (stress_contribution_groups.xs(ml_name).save_all == False).any():
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
                ).set_index(["column_names", "date"])
                obsnmes = stress_obs_contribs.loc[contribs_all.index].obsnme
                contribs_all.index = obsnmes.values
                # store the series
                contribs_all_list.append(contribs_all.Observations)

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
