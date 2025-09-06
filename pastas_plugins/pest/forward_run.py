import pyemu
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
    ml = load_model(fpath / "model.pas")

    # update standard pastas model parameters
    parameters = read_csv(fpath / "parameters_sel.csv", index_col=0)
    for pname, val in parameters.loc[:, "optimal"].items():
        pname = pname.replace("_g", "_A") if pname.endswith("_g") else pname
        ml.set_parameter(pname, optimal=val)
    # update custom stressmodel parameters
    pickles = glob.glob(fpath / "*.parameteriser.pkl")
    stressmodel_parameterisers = [
        dill.load(open(sm_p, "rb")) for sm_p in pickles
    ]  # pickle.load(
    for sm_p in stressmodel_parameterisers:
        # get df of updated (parameterised and interpolated) stress TimeSeries for model
        updated_stress_df = sm_p.interpolate_stresses(**sm_p.interp_kwargs)
        # update stress TimeSeries
        smodel = ml.stressmodels.get(sm_p.stressmodel_name)
        for stress_series in smodel.stress:
            stress_series.series_original = updated_stress_df.loc[:, stress_series.name]

    # simulate
    simulation = ml.simulate()
    simulation.loc[ml.observations().index].to_csv(fpath / "simulation.csv")


def run_pypestworker(
    pst: str | pyemu.Pst,
    host: int,
    port: int,
    ml: Model,  # ml_dict: dict,
    parameter_index: dict,
    stressmodel_parameterisers: list = [],
) -> None:
    from logging import getLogger

    from pastas_plugins.pest.parameterisers import (  # noqa: F401
        BaseParameteriser,
        WellModelParameteriser,
    )

    ppw = pyemu.os_utils.PyPestWorker(
        pst=pst,
        host=host,
        port=port,
        verbose=False,
    )

    # load pastas model
    # ml = _load_model(ml_dict)  # load_model(ml_file)

    # reactivate the model logger - it was deactivated before provision
    # as an arg to this module.
    # (multiprocesing uses pickling (of ml in this case), and pickle
    # can't pickle open file handle logger instances)
    ml.logger = getLogger(ml.__name__)

    pvals = ppw.get_parameters()
    if pvals is None:
        return None

    while True:
        # update standard pastas model parameters
        for pname, val in pvals.items():
            pname = parameter_index[pname]
            if pname in ml.parameters.keys():
                ml.set_parameter(pname, optimal=val)
        # update custom stressmodel parameters
        for sm_p in stressmodel_parameterisers:
            sm_p_parnames = sm_p.stress_pars.parnme
            new_par_values = pvals.loc[sm_p_parnames].values
            # get df of updated (parameterised and interpolated) stress TimeSeries for model
            interp_kwargs = sm_p.interp_kwargs
            interp_kwargs["updated_sourcevals"] = new_par_values
            updated_stress_df = sm_p.interpolate_stresses(**interp_kwargs)
            # update stress TimeSeries
            smodel = ml.stressmodels.get(sm_p.stressmodel_name)
            # TODO update this to deal with flat (melted) rather than pivoted stress timeseries df - dead easy where column_names==smodel
            for stress_series in smodel.stress:
                stress_series.series_original = updated_stress_df.loc[
                    :, stress_series.name
                ]

        sim = ml.simulate()
        obsvals = sim.loc[ml.observations().index]
        obsvals.index = ppw._pst.observation_data.index
        ppw.send_observations(obsvals=obsvals)
        pvals = ppw.get_parameters()
        if pvals is None:
            break


if __name__ == "__main__":
    run()
