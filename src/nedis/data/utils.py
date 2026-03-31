import numpy as np


external_data_path = "../../nalab-data/data"
default_data_path = "./data"


def select_ref(reference, data_features, data_timepoints, data_groups):

    if not isinstance(reference, tuple):
        if reference == "all":
            reference = ("all", "all")
        elif reference in data_timepoints:
            reference = (reference, "all")
        elif isinstance(reference, int):
            reference = (np.unique(data_timepoints)[reference], "all")
        else:
            raise ValueError(f"Unknown reference identifier: '{reference}'")

    ref_tp, ref_g = reference

    if ref_tp == "all":
        select_timepoints = np.ones(data_features.shape[0], dtype=bool)
    else:
        select_timepoints = data_timepoints == ref_tp

    if ref_g == "all":
        select_groups = np.ones(data_features.shape[0], dtype=bool)
    else:
        select_groups = data_groups == ref_g

    return reference, select_timepoints & select_groups


def format_reference(reference):
    return str(reference).replace("'", "")