import os
import re
from typing import Union
import pandas as pd

from rpy2 import robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


from nedis.data.utils import default_data_path, select_ref, format_reference


def load_all(data_path=default_data_path):

    return [
        load_task_multiomics___immunesystem_trimester(
            reference="all", data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference=1, data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference=3, data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference=4, data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference=1, pp_mode='drop', data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference="all", pp_mode='reverse', data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference="all", feature_groups=["immune_system", "plasma_somalogic"], data_path=data_path),
        load_task_multiomics___immunesystem_trimester(
            reference="all", pp_mode='reverse', feature_groups=["immune_system", "plasma_somalogic"], data_path=data_path),
    ]


def load_task_multiomics___immunesystem_trimester(
        reference: Union[str, int, tuple] = "all",
        pp_mode='default',
        feature_groups="immune_system",
        data_path=default_data_path):
 
    data_multiomics = load_pregnancy_multiomics_data(data_dir=data_path)
    if pp_mode == 'drop':
        data_multiomics = data_multiomics.iloc[data_multiomics.timepoint.values != 4, :]
        pp_prefix = ' no PP'
    elif pp_mode == "reverse":
        data_multiomics.loc[data_multiomics.timepoint.values == 4, "timepoint"] = 0
        pp_prefix = ' PP first'
    elif pp_mode == "default":
        pp_prefix = ''
        pass
    else:
        raise ValueError(f"Unknown post partum mode: {pp_mode}")
    # data_multiomics_val = load_multiomics(validation=True, data_dir=data_path)

    if isinstance(feature_groups, str):
        feature_groups = [feature_groups]
    
    data = data_multiomics
    data_features = data[feature_groups].values
    
    data_feature_names = data[feature_groups].columns.values
    if len(feature_groups) == 1:
        data_feature_names = [f[1] for f in data_feature_names]
    else:
        data_feature_names = ["_".join(f) for f in data_feature_names]

    data_timepoints = data.timepoint.values
    data_timepoints_continuous = None
    
    data_groups = None
    data_entities = data["study_id"].values

    reference, data_reference = select_ref(reference, data_features, data_timepoints, data_groups)

    label = ", ".join([f.replace("_", " ") for f in feature_groups])
    return dict(
        name=f"Multiomics ({label}){pp_prefix}; ref: ({format_reference(reference)})",
        data=data,
        features=data_features,
        feature_names=data_feature_names,
        timepoints=data_timepoints,
        timepoints_continuous=data_timepoints_continuous,
        groups=data_groups,
        entities=data_entities,
        select_reference=data_reference
    )



def load_pregnancy_multiomics_data(flatten_index=False, data_dir="data"):
    
    # loading Cellfree RNA','PlasmaLuminex','SerumLuminex','Microbiome','ImmuneSystem','Metabolomics', 'PlasmaSomalogic'
    # see README
    robjects.r['load'](os.path.join(data_dir, "raw/multiomics/multiomics.rda"))
    serum_luminex =    _load_matrix(robjects.r["InputData"][2])
    plasma_luminex =   _load_matrix(robjects.r["InputData"][1])
    microbiome =       _load_matrix(robjects.r["InputData"][3])
    cellfree_rna =     _load_matrix(robjects.r["InputData"][0])
    immune_system =    _load_matrix(robjects.r["InputData"][4])
    # metabolomics =     _load_matrix(robjects.r["InputData"][5], skip_colnames=True) # non-utf colnames; should probably fix them somehow
    metabolomics =     _load_matrix(robjects.r["InputData"][5])
    plasma_somalogic = _load_matrix(robjects.r["InputData"][6])
    
    # combine datasets
    data = pd.concat(
        [
            cellfree_rna, 
            plasma_luminex, 
            serum_luminex, 
            microbiome, 
            immune_system, 
            metabolomics, 
            plasma_somalogic
        ], 
        keys=[
            "cellfree_rna", 
            "plasma_luminex", 
            "serum_luminex", 
            "microbiome",           # check
            "immune_system", 
            "metabolomics", 
            "plasma_somalogic"],    # check
        axis=1)
    
    # divide study id and term
    data.insert(0, "study_id", [re.sub(r"_.*", "", i) for i in data.index])
    data.insert(1, "timepoint", [int(re.sub(r".*_", "", i)) for i in data.index])
    
#     # gestational age for timepoint 4 is broken (weird numbers) ...
# #     data.insert(2, "gestational_age", robjects.r["featureweeks"])
#     # ... so we use an external file to add that information
#     # THIS IS DRIVING ME CRAZY, CAN'T WE JUST HAVE _ONE_ FILE WITH EVERYTHING?!?!?!?
#     df = pd.read_csv(os.path.join(data_dir, "default/multiomics/Study.csv"))
#     df[4] = df["GA"] + df["4 (PP, wks)"]
#     df = df[["Study #", "1", "2", "3", 4]]\
#         .rename(columns={"Study #": "study_id", "1": 1, "2": 2, "3": 3})\
#         .set_index("study_id").stack().reset_index()\
#         .rename(columns={ "level_1": "timepoint", 0: "gestational_age" })
#     df.columns = pd.MultiIndex.from_tuples([(c, "") for c in df.columns ])
#     data = df.merge(data, on=["study_id", "timepoint"])
    
    if flatten_index:            
        data.columns = ["_".join([entry for entry in column_tuple if len(entry) > 0])
                        for column_tuple in data.columns.tolist()]
    
    return data


def _load_matrix(r_matrix, skip_colnames=False, number=True):

    df = load_r_matrix(r_matrix, read_colnames=not skip_colnames, colname_formatter="number" if number else None)

    # cleanup rownames
    if any(['_BL' in n for n in df.index]):
        df.index = [r.replace("_3", "_4").replace("_2", "_3").replace("_1", "_2").replace("_BL", "_1")
                    for r in df.index]

    return df


def load_r_matrix(
        r_matrix, read_rownames=True, read_colnames=True,
        rowname_formatter=None, colname_formatter=None, name_encoding="latin1"):

    if isinstance(r_matrix, str):
        r_matrix = robjects.r[r_matrix]

    if rowname_formatter == "number":
        def rowname_formatter(i, rowname):
            if rowname is None:
                return str(i)
            else:
                return "{}_{}".format(i, rowname)

    if colname_formatter == "number":
        def colname_formatter(i, colname):
            if colname is None:
                return str(i)
            else:
                return "{}_{}".format(i, colname)

    # handle rownames
    if read_rownames:
        rownames = robjects.r["rownames"](r_matrix)
        if name_encoding is not None:
            rownames = robjects.r("iconv")(rownames, name_encoding, "UTF-8")
        if rowname_formatter is not None:
            rownames = [rowname_formatter(i, r) for i, r in enumerate(rownames)]
    else:
        if rowname_formatter is not None:
            rownames = [rowname_formatter(i, None) for i in range(robjects.r["dim"](r_matrix)[0])]
        else:
            rownames = None

    # handle colnames
    if read_colnames:
        colnames = robjects.r["colnames"](r_matrix)
        if name_encoding is not None:
            colnames = robjects.r("iconv")(colnames, name_encoding, "UTF-8", sub='???')
        if colname_formatter is not None:
            colnames = [colname_formatter(i, colname) for i, colname in enumerate(colnames)]
    else:
        if colname_formatter is not None:
            colnames = [colname_formatter(i, None) for i in range(robjects.r["dim"](r_matrix)[1])]
        else:
            colnames = None

    with localconverter(robjects.default_converter + pandas2ri.converter):
        return pd.DataFrame(robjects.conversion.rpy2py(r_matrix), index=rownames, columns=colnames)
