from nedis.utils import slugify, select
from nedis.data.utils import external_data_path, default_data_path 


def load_task_list(
        data_path=default_data_path,
        include=None,
        exclude=None):

    task_list = list()
    print(data_path)

    # multiomics
    if select("multiomics", include=include, exclude=exclude):
        print("* multiomics")
        from nedis.data.datasets.multiomics import load_all
        task_list += load_all(data_path=data_path)

    # pree
    if select("pree", include=include, exclude=exclude):
        print("* pree")
        from nedis.data.datasets.pree import load_all
        task_list += load_all(data_path=data_path)
    
    for t in task_list:
        t["id"] = slugify(t['name'])

    return task_list
