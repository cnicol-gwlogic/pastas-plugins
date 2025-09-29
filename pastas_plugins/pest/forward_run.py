import pyemu
from pandas import DataFrame
from pastas import Model


def run() -> None:
    # load packages
    import glob
    from pathlib import Path

    import dill  # pickle
    from pandas import read_csv
    from pastas.io.base import load as load_model

    from pastas_plugins.pest.parameterisers import (  # noqa: F401
        BaseParameteriser,
        WellModelParameteriser,
    )

    # base path
    fpath = Path(__file__).parent

    # load pastas model
    models = [load_model(m) for m in glob.glob(str(fpath / "model_*.pas"))]

    # update standard pastas model parameters
    parameters = read_csv(fpath / "parameters_sel.csv", index_col=0)
    for ml in models:
        for pname, val in parameters.loc[:, "optimal"].items():
            pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
            ml.set_parameter(pname[3:], optimal=val)
    # update custom stressmodel parameters
    pickles = glob.glob(str(fpath / "*.parameteriser.pkl"))
    stressmodel_parameterisers = [
        dill.load(open(sm_p, "rb")) for sm_p in pickles
    ]  # pickle.load(
    for sm_p in stressmodel_parameterisers:
        # get df of updated (parameterised and interpolated) stress TimeSeries for model
        updated_stress_df = sm_p.interpolate_stresses(**sm_p.interp_kwargs)
        # update stress TimeSeries
        for ml in models:
            smodel = ml.stressmodels.get(sm_p.stressmodel_name)
            for stress_series in smodel.stress:
                if stress_series.name in sm_p.stress_names:
                    stress_series.series_original = updated_stress_df.loc[
                        :, stress_series.name
                    ]
    # ^^ one sm_p even for many pastas models in one pest cal will work ok - we just update the stress rates,
    # while pumping well distances from each model (obs bore) remain as originally defined per model.
    # Pest-calibrated rates are the same across all pastas models, but distances of q wells from obs bores vary. Yay.

    # simulate
    for ml in models:
        ml_name = ml.name
        simulation = ml.simulate()
        simulation.loc[ml.observations().index].to_csv(fpath / f"simulation_{ml_name}.csv")

        # save head_diffs too
        head_diffs = simulation - simulation.loc[simulation.index.min()]
        head_diffs.loc[ml.observations().index].to_csv(fpath / f"simulation_head_diffs_{ml_name}.csv")


def run_pypestworker(
    pst: str | pyemu.Pst,
    host: int,
    port: int,
    #ml: Model,  # ml_dict: dict,
    models: dict,
    parameter_index: dict,
    observation_index: DataFrame,
    stressmodel_parameterisers: list = [],
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
        verbose=False,
    )

    # load pastas model
    # ml = _load_model(ml_dict)  # load_model(ml_file)

    pvals = ppw.get_parameters()
    if pvals is None:
        return None

    # reactivate the model logger - it was deactivated before provision
    # as an arg to this module.
    # (multiprocesing uses pickling (of ml in this case), and pickle
    # can't pickle open file handle logger instances)
    #ml.logger = getLogger(ml.name)

    while True:
        # update custom stressmodel parameters
        for sm_p in stressmodel_parameterisers:
            sm_p_parnames = sm_p.stress_pars.parnme
            new_par_values = pvals.loc[sm_p_parnames].values
            # get df of updated (parameterised and interpolated) stress TimeSeries for model
            interp_kwargs = sm_p.interp_kwargs
            interp_kwargs["updated_sourcevals"] = new_par_values
            updated_stress_df = sm_p.interpolate_stresses(**interp_kwargs)

        obsvals_list, obs_diffs_list = [], []
        head_obsgps = [
            og for og in observation_index.index.get_level_values("obgnme").unique() \
            if og.find("headdiff") < 0
        ]
        head_diff_obsgps = [
            og for og in observation_index.index.get_level_values("obgnme").unique() \
            if og.find("headdiff") >= 0
        ]
        for ml_name,ml in models.items():
            ml.settings["tmin"] = None
            ml.settings["tmax"] = None
            # update standard pastas model parameters
            """pvals.to_csv(f"{ml.name}.{ppw.net_pack.runid}.pvals.temp.csv")
            updated_stress_df.to_csv(f"{ml.name}.{ppw.net_pack.runid}.updated_stress_df.temp.csv")"""
            for pname, val in pvals.items():
                pname = parameter_index[pname]
                pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
                if pname[3:] in ml.parameters.index.values:
                    ml.set_parameter(pname[3:], optimal=val)
            # update stress TimeSeries
            for sm_p in stressmodel_parameterisers:
                smodel = ml.stressmodels.get(sm_p.stressmodel_name)
                for stress_series in smodel.stress:
                    if stress_series.name in sm_p.stress_names:
                        stress_series.series_original = updated_stress_df.loc[
                            :, stress_series.name
                        ]
                        """ml.stressmodels[
                            sm_p.stressmodel_name
                        ].stress[idx].series_original = updated_stress_df.loc[
                            :, stress_series.name
                        ]"""
            sim = ml.simulate()
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

        obsvals = concat(obsvals_list, axis=0, ignore_index=False)
        obs_diffs = concat(obs_diffs_list, axis=0, ignore_index=False)
        obsvals_all = concat([obsvals, obs_diffs], axis=0, ignore_index=False)
        ppw.send_observations(obsvals=obsvals_all)

        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    run()
