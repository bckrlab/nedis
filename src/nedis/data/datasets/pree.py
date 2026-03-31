import os
from typing import Union
import numpy as np
import pandas as pd
import rpy2.robjects as ro

from nedis.data.utils import default_data_path, select_ref, format_reference


def load_all(data_path=default_data_path):
    return [
        load_task_pree___immunesystem_timepoint_none(
            reference="all", data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            reference=1, data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            reference=3, data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            reference=("all", "control"), data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            reference=(1, "control"), data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            reference=(3, "control"), data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            group="pree",
            reference="all", data_path=data_path),
        load_task_pree___immunesystem_timepoint_none(
            group="healthy",
            reference="all", data_path=data_path),
    ]


def load_task_pree___immunesystem_timepoint_none(
        group="all",
        reference: Union[str, int, tuple] = "all",
        data_path=default_data_path):

    data_pree = load_data(data_path=data_path)
    if group == "all":
        pass
    elif group == "pree":
        data_pree = data_pree[data_pree.meta.preeclampsia_bool]
    elif group == "healthy":
        data_pree = data_pree[~data_pree.meta.preeclampsia_bool]
    else:
        raise ValueError(f"Unknown group: {group}")

    data = data_pree

    data_features = data.features.values
    data_feature_names = data.features.columns.values

    data_timepoints = data.meta.timepoint.values
    data_timepoints_continuous = data.meta.ga.values
    data_groups = np.array(["pree" if pree else "control" for pree in data.meta.preeclampsia_bool.values])
    data_entities = data.meta.patient_id.values

    reference, data_reference = select_ref(reference, data_features, data_timepoints, data_groups)

    return dict(
        name=f"Preeclampsia{'' if group == 'all' else ' (' + group + ')'}; ref: {format_reference(reference)}",
        data=data,
        features=data_features,
        feature_names=data_feature_names,
        timepoints=data_timepoints,
        timepoints_continuous=data_timepoints_continuous,
        groups=data_groups,
        entities=data_entities,
        select_reference=data_reference)


def load_data(data_path=default_data_path):
    import numpy as np

    ro.r["load"](os.path.join(data_path, "raw/preeclampsia/preeclampsia.rda"))

    data = np.array(ro.r["data"])
    features = ro.r["colnames"](ro.r["data"])
    patient_ids = ro.r["rownames"](ro.r["data"])
    pe = np.array(ro.r["pe"])
    GA = np.array(ro.r["GA"])

    data_pree = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples([("features", f) for f in features]))
    data_pree[("meta", "patient_id")] = patient_ids
    data_pree[("meta", "preeclampsia")] = pe.astype(int)
    data_pree[("meta", "preeclampsia_bool")] = pe.astype(int) == 2
    data_pree[("meta", "ga")] = GA.astype(int)
    data_pree[("meta", "timepoint")] = 0
    data_pree.loc[(data_pree[("meta", "ga")] >= 1) & (data_pree[("meta", "ga")] <= 14), ("meta", "timepoint")] = 1
    data_pree.loc[(data_pree[("meta", "ga")] >= 15) & (data_pree[("meta", "ga")] <= 21), ("meta", "timepoint")] = 2
    data_pree.loc[(data_pree[("meta", "ga")] >= 22) & (data_pree[("meta", "ga")] <= 999), ("meta", "timepoint")] = 3

    return data_pree
