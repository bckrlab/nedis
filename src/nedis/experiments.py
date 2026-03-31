import pickle
import pathlib

import scipy
import sklearn.manifold
import sklearn.metrics

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import nedis.cordis.clustering
from nedis.utils import slugify
from nedis.visualization import plot_cordis_cluster
from nedis.base import calculate_correlation_disruption_matrix_cv
from nedis.cordis.utils import calculate_disruption_values_for_cluster, prepare_y


def log_cluster_experiment(
        exp_name, 
        X, y, entities, feature_names, 
        transformer, 
        groups=None,
        multi_sample=False,
        y_map=None,
        cluster_summary_visualizations=None,
        overwrite=False,
        output_dir="../_out",
        topk=6, 
        include_stats=None, 
        exclude_stats=None, 
        random_state=42,
        random_state_tsne=None,
        show_plots=False,
        visualize=True):

    if random_state_tsne is None:
        random_state_tsne = random_state

    output_path = pathlib.Path(output_dir) / exp_name

    def check_tags(*tags):
        check = include_stats is None or np.any([t in tags for t in include_stats])
        check &= exclude_stats is None or np.all([t not in tags for t in exclude_stats])
        return check

    def load(component, func):
        return load_component(
            component, func=func, overwrite=overwrite, output_dir=output_path)

    # log input
    transformer = load("transformer", lambda: transformer)
    X = load("X", lambda: X)
    y = load("y", lambda: y)
    groups = load("groups", lambda: groups)
    entities = load("entities", lambda: entities)
    feature_names = load("feature_names", lambda: feature_names)
    feature_names = np.array(feature_names)
    
    if not multi_sample:
        samples = None
    else:
        samples = entities
    
    # prepare analysis

    y_unique = np.unique(y)

    def f_correlation_matrices_dict():
        correlation_matrices_dict = {}
        for yy in y_unique: 
            cor = np.corrcoef(X[y == yy, :], rowvar=False)
            correlation_matrices_dict[yy] = cor
        return correlation_matrices_dict

    correlation_matrices_dict = load(
        "correlation_matrices_dict", 
        func=f_correlation_matrices_dict)

    def f_disruption_matrices_dict():
        disruption_matrices_dict = {}
        for yy in y_unique:
            print("Calculating disruption values for reference:", yy)
            disruption_matrices_dict[yy] = calculate_correlation_disruption_matrix_cv(
                X, idx_ref=(y == yy), groups=entities if groups else None, samples=samples)
        return disruption_matrices_dict
    
    disruption_matrices_dict = load(
        "disruption_matrices_dict", 
        func=f_disruption_matrices_dict)
    
    def f_coordinates_dict():
        coordinates_dict = {}
        for yy, correlation_matrix in correlation_matrices_dict.items(): 
            tsne = sklearn.manifold.TSNE(n_components=2, random_state=random_state_tsne)
            coo = tsne.fit_transform(abs(correlation_matrix))
            coordinates_dict[yy] = coo
        return coordinates_dict

    coordinates_dict = load(
        "coordinates_dict", 
        func=f_coordinates_dict)

    ###
    # visualize overall stats for task
    ###

    # plot correlation matrices
    if check_tags("overall"):
        fig, axes = plt.subplots(1, len(y_unique), figsize=(6 * len(y_unique), 4))
        for i, yy in enumerate(y_unique):
            ax = axes[i]
            cor = correlation_matrices_dict[yy]
            nedis.visualization.visualize_feature_clusters(
                cor, ax=ax, heatmap_kwargs=dict(cmap="vlag", center=0, vmin=-1, vmax=1))

    # plot correlation networks
    if check_tags("overall"):
        fig, axes = plt.subplots(1, len(y_unique), figsize=(4 * 2 * len(y_unique), 4 * 2))
        for i, yy in enumerate(y_unique):
            ax = axes[i]
            ax.axis("off")
            ax.set_title(yy)
            plot_cordis_cluster(
                None, coordinates_dict[y_unique[0]], correlation_matrices_dict[yy], ax=ax)

    if visualize:

        ###
        # visualize clusters
        ###

        clusters = sorted(transformer.clusters_, key=lambda x: x["reference_score"])
        clusters_selected = list(reversed(clusters[-topk:]))

        def visualize_clusters(target="correlation", target_visualization="tsne"):

            fig_path = output_path / f"figures"
            fig_path.mkdir(parents=True, exist_ok=True)


            for i_cluster, c in enumerate(clusters_selected):

                filename = fig_path / (f"cluster-{i_cluster:03d}___" + "-".join(str(cc) for cc in c["id"]) + "_rows.txt")
                with open(filename, 'w') as f:
                    for feature in feature_names[c["rows"]]:
                        f.write(f"{feature}\n")

                filename = fig_path / (f"cluster-{i_cluster:03d}___" + "-".join(str(cc) for cc in c["id"]) + "_columns.txt")
                with open(filename, 'w') as f:
                    for feature in feature_names[c["columns"]]:
                        f.write(f"{feature}\n")

                visualize_cluster(
                    c, y, entities, samples, feature_names, 
                    coordinates_dict, correlation_matrices_dict, disruption_matrices_dict, 
                    cluster_number=i_cluster,
                    y_map=y_map,
                    target=target, 
                    target_visualization=target_visualization,
                    summary_visualizations=cluster_summary_visualizations,
                    output_path=output_path)
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()

        if check_tags("cluster", "tsne", "correlation"):
            visualize_clusters(target="correlation", target_visualization="tsne")

        if check_tags("cluster", "tsne", "disruption"):
            visualize_clusters(target="disruption", target_visualization="tsne")
                
        if check_tags("cluster", "circle", "correlation"):
            visualize_clusters(target="correlation", target_visualization="circle")
                
        if check_tags("cluster", "circle", "disruption"):
            visualize_clusters(target="disruption", target_visualization="circle")
                
        if check_tags("cluster", "heatmap", "correlation"):
            visualize_clusters(target="correlation", target_visualization="heatmap")
            
        if check_tags("cluster", "heatmap", "disruption"):
            visualize_clusters(target="disruption", target_visualization="heatmap")

        if check_tags("cluster", "correlation-disruption"):
            visualize_clusters(target="correlation-disruption", target_visualization="heatmap")

    return transformer
       
        
def visualize_cluster(
        cluster, 
        y,
        entities,
        samples,
        feature_names,
        coordinates_dict,
        correlation_matrices_dict, 
        disruption_matrices_dict, 
        y_map=None,
        cluster_number=None, 
        output_path=None, 
        target="correlation",
        target_visualization="heatmap",
        summary_visualizations=None):
    
    # TODO: clean up this mess!

    if cluster_number is None:
        cluster_number = np.random.randint(0, 999)

    y_unique = np.unique(y)
    y_unique_named = [(y_map[yy] if y_map is not None else y) for yy in y_unique]
    
    values, _ = calculate_disruption_values_for_cluster(
        cluster, 
        disruption_matrices_dict[cluster["reference_label"]], 
        disruption_aggregation="mean")
    
    # clustering; I should not abuse sns for this :
    g = sns.clustermap(correlation_matrices_dict[cluster["reference_label"]][cluster["rows"],:][:,cluster["columns"]])
    # g = sns.clustermap(functools.reduce(operator.add, disruption_matrices_dict.values()).mean(axis=0)[cluster["rows"],:][:,cluster["columns"]])
    idx_rows, idx_cols = g.dendrogram_row.reordered_ind, g.dendrogram_col.reordered_ind
    plt.close()

    # initialize figure
    n_rows = 2 

    n_plots = len(y_unique)
    if summary_visualizations is None:
        n_plots += 5
    else:
        n_plots += int(len(summary_visualizations) / n_rows)
        
    scale = 4
    fig, axes = plt.subplots(
        n_rows, n_plots, 
        figsize=(4 * n_plots * scale, 3 * n_rows * scale),
        squeeze=False)
    
    ref_label = y_map[cluster['reference_label']] if y_map is not None else cluster['reference_label'] 
    fig.suptitle(f"Cluster {cluster_number}: Reference data={ref_label}, Id={cluster['id']}", fontsize=35)
    
    # summary plots
    
    n_summary_plots = 0
    
    # kde plot
    if summary_visualizations is None or "kde" in summary_visualizations:
        ax = axes[0, n_summary_plots]
        sns.kdeplot(x=values, hue=prepare_y(y, samples), ax=ax, palette=sns.color_palette(n_colors=len(y_unique)))
        if len(y_unique) == 2:
            score_label = f"AUC: {sklearn.metrics.roc_auc_score(prepare_y(y, samples), values):.02f}"
        else:
            score_label = f"Spearman: {scipy.stats.spearmanr(values, prepare_y(y, samples))[0]:.02f}"  
        ax.set_title(score_label)
        ax.annotate(score_label, (0,1), xycoords="axes fraction", ha='left', va='top', fontsize=55)

        if axes.shape[0] == 0:
            # TODO: this is horrible and needs to be cleaned up
            n_summary_plots += 1
              
    # line plot  
    if summary_visualizations is None or "line" in summary_visualizations:
        ax = axes[0, n_summary_plots]
        x_rank = scipy.stats.rankdata(prepare_y(y, samples), method="dense")
        sns.lineplot(x=x_rank, y=values, hue=entities, color="blue", alpha=0.1, ax=ax)
        sns.lineplot(x=x_rank, y=values, ax=ax)
        ax.set_xticks(np.unique(x_rank))
        ax.set_xticklabels(y_unique_named)
        ax.get_legend().remove()
        if len(y_unique) == 2:
            score_label = f"AUC: {sklearn.metrics.roc_auc_score(prepare_y(y, samples), values):.02f}"
        else:
            score_label = f"Spearman: {scipy.stats.spearmanr(values, prepare_y(y, samples))[0]:.02f}"  
        
        if axes.shape[0] == 0:
            n_summary_plots += 1
        
        ax.set_title(score_label)
        ax.annotate(score_label, (0,1), xycoords="axes fraction", ha='left', va='top', fontsize=55)
        
    # box plot  
    if summary_visualizations is None or "box" in summary_visualizations:
        if axes.shape[0] > 1:
            # TODO: this is horrible and needs to be cleaned up
            ax = axes[1, n_summary_plots]    
        else:
            ax = axes[0, n_summary_plots]
        
        n_summary_plots += 1
        sns.boxplot(x=prepare_y(y, samples), y=values, ax=ax)
        sns.swarmplot(x=prepare_y(y, samples), y=values, color="black", ax=ax)
        
        if len(y_unique) == 2:
            ax.set_title(f"AUC: {sklearn.metrics.roc_auc_score(prepare_y(y, samples), values):.04f}")
        else:
            ax.set_title(f"Spearman: {scipy.stats.spearmanr(values, prepare_y(y, samples))[0]:.04f}")  
        ax.set_xticklabels(y_unique_named)
        
    # individual correlation profiles
    if summary_visualizations is None or "correlation" in summary_visualizations:
        ax = axes[0, n_summary_plots]
        correlation_values = np.abs(np.concatenate([
            correlation_matrices_dict[yy][cluster["edges"].nonzero()] 
            for yy in y_unique]).flatten())
        correlation_y = np.repeat(np.arange(len(y_unique)), len(cluster["edges"].nonzero()[0]))
        correlation_hue = np.tile(np.arange(len(cluster["edges"].nonzero()[0])), len(y_unique))
        
        sns.lineplot(x=correlation_y, y=correlation_values, hue=correlation_hue, color="blue", alpha=0.3, ax=ax)
        ax.set_xticks(np.arange(len(y_unique)))
        ax.set_xticklabels(y_unique_named)
        ax.get_legend().remove()
        n_summary_plots += 1
    
    # individual disruption profiles
    if summary_visualizations is None or "disruption" in summary_visualizations:
        ax = axes[0, n_summary_plots]
        yy = y_unique[0]
        d_values = np.concatenate([
            np.mean(np.concatenate([
                disruption_matrices_dict[yy][i][cluster["edges"].nonzero()].reshape(1,-1)
                for i in range(disruption_matrices_dict[y_unique[0]].shape[0])], axis=0), axis=0)
            for yy in y_unique
        ])
        
        print(d_values.shape)
        d_y = np.repeat(np.arange(len(y_unique)), len(cluster["edges"].nonzero()[0]))
        d_hue = np.tile(np.arange(len(cluster["edges"].nonzero()[0])), len(y_unique))
        
        sns.lineplot(x=d_y, y=d_values, hue=d_hue, color="blue", alpha=0.3, ax=ax)
        ax.set_xticks(np.arange(len(y_unique)))
        ax.set_xticklabels(y_unique_named)
        ax.get_legend().remove()
        n_summary_plots += 1
    
    # conditional plots
    
    if target_visualization == "heatmap":
    
        if target == "correlation":
            for i, yy in enumerate(y_unique):
                
                ax = axes[0, i + n_summary_plots]
                
                if target_visualization == "heatmap":
                    
                    heatmap_values = correlation_matrices_dict[yy]
                    heatmap_values = heatmap_values[cluster["rows"],:][:,cluster["columns"]]
                    heatmap_values = heatmap_values[idx_rows,:][:,idx_cols]
                    sns.heatmap(
                        heatmap_values, 
                        vmin=-1, vmax=1, center=0,
                        cmap="vlag",
                        xticklabels=feature_names[cluster["columns"]][idx_cols],
                        yticklabels=feature_names[cluster["rows"]][idx_rows],
                        mask=np.triu(heatmap_values),
                        ax=ax)
                
                ax.set(title=f"Subset: {y_map[yy] if y_map is not None else yy} (id={yy})")
        
        elif target == "disruption":
            
            # figure out min and max values for mean disruptions
            vmin, vmax = float('inf'), float('-inf')
            for i, yy in enumerate(y_unique):
                plot_values = np.mean(disruption_matrices_dict[cluster["reference_label"]][prepare_y(y, samples) == yy], axis=0)
                plot_values = plot_values[cluster["rows"],:][:,cluster["columns"]][idx_rows,:][:,idx_cols]
                vmin = min(vmin, np.min(plot_values))
                vmax = max(vmax, np.max(plot_values))
            
            for i, yy in enumerate(y_unique):
                
                ax = axes[0, i + n_summary_plots]
                
                plot_values = np.mean(disruption_matrices_dict[cluster["reference_label"]][prepare_y(y, samples) == yy], axis=0)
                plot_values = plot_values[cluster["rows"],:][:,cluster["columns"]][idx_rows,:][:,idx_cols]
                
                sns.heatmap(
                    plot_values,
                    center=0, vmin=vmin, vmax=vmax,
                    cmap="vlag",
                    xticklabels=feature_names[cluster["columns"]][idx_cols],
                    yticklabels=feature_names[cluster["rows"]][idx_rows],
                    mask=np.triu(plot_values),
                    ax=ax)
                ax.set(title=f"Subset: {y_map[yy] if y_map is not None else yy} (id={yy})")
                

        elif target == "correlation-disruption":
            # TODO: clean up!!!

            for i, yy in enumerate(y_unique):
                
                ax = axes[0, i + n_summary_plots]
                
                if target_visualization == "heatmap":
                    
                    heatmap_values = correlation_matrices_dict[yy]
                    heatmap_values = heatmap_values[cluster["rows"],:][:,cluster["columns"]]
                    heatmap_values = heatmap_values[idx_rows,:][:,idx_cols]
                    sns.heatmap(
                        heatmap_values, 
                        vmin=-1, vmax=1, center=0,
                        cmap="vlag",
                        xticklabels=feature_names[cluster["columns"]][idx_cols],
                        yticklabels=feature_names[cluster["rows"]][idx_rows],
                        mask=np.triu(heatmap_values),
                        ax=ax)
                
                ax.set_title(f"Subset: {y_map[yy] if y_map is not None else yy} (id={yy})", fontsize=35)
        
            
            # figure out min and max values for mean disruptions
            vmin, vmax = float('inf'), float('-inf')
            for i, yy in enumerate(y_unique):
                plot_values = np.mean(disruption_matrices_dict[cluster["reference_label"]][prepare_y(y, samples) == yy], axis=0)
                plot_values = plot_values[cluster["rows"],:][:,cluster["columns"]][idx_rows,:][:,idx_cols]
                vmin = min(vmin, np.min(plot_values))
                vmax = max(vmax, np.max(plot_values))
            
            for i, yy in enumerate(y_unique):
                
                ax = axes[1, i + n_summary_plots]
                
                plot_values = np.mean(disruption_matrices_dict[cluster["reference_label"]][prepare_y(y, samples) == yy], axis=0)
                plot_values = plot_values[cluster["rows"],:][:,cluster["columns"]][idx_rows,:][:,idx_cols]
                
                sns.heatmap(
                    plot_values,
                    center=0, vmin=vmin, vmax=vmax,
                    cmap="vlag",
                    xticklabels=feature_names[cluster["columns"]][idx_cols],
                    yticklabels=feature_names[cluster["rows"]][idx_rows],
                    mask=np.triu(plot_values),
                    ax=ax)
                ax.set_title(f"Subset: {y_map[yy] if y_map is not None else yy} (id={yy})", fontsize=35)
                

    elif target_visualization == "tsne":
        for i, yy in enumerate(y_unique):
            ax = axes[0, i + n_summary_plots]
            plot_cordis_cluster(
                cluster, 
                coordinates_dict[y_unique[0]],  # keep coordinates stable: TODO: make configurable? 
                correlation_matrices_dict[yy], 
                disruption_matrices_dict[cluster["reference_label"]][prepare_y(y, samples) == yy],
                labels=feature_names,
                nodes_mode="all nodes",
                edges_mode=target,
                ax=ax)
            ax.set(title=f"Subset: {y_map[yy] if y_map is not None else yy} (id={yy})")
            
    elif target_visualization == "circle":
        for i, yy in enumerate(y_unique):
            ax = axes[0, i + n_summary_plots]
            plot_cordis_cluster(
                cluster, 
                coordinates_dict[y_unique[0]],  # keep coordinates stable: TODO: make configurable? 
                correlation_matrices_dict[yy], 
                disruption_matrices_dict[cluster["reference_label"]][prepare_y(y, samples) == yy],
                labels=feature_names,
                nodes_mode="cluster circle",
                edges_mode=target,
                ax=ax)            
            ax.set(title=f"Subset: {y_map[yy] if y_map is not None else yy} (id={yy})")


    if output_path is not None:
        
        fig_path = output_path / f"figures/cluster_{slugify(target_visualization)}_{target}"
        fig_path.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            fig_path / (f"cluster-{cluster_number:03d}___" + "-".join(str(cc) for cc in cluster["id"]) + ".pdf"), 
            bbox_inches="tight")

    return fig, axes


def load_component(component, func=None, overwrite=False, output_dir="../_out", verbose=True):
    
    output_path = pathlib.Path(output_dir)
    component_file = output_path / f"{component}.pickle"
    if component_file.exists() and not overwrite:
        if verbose:
            print(f"Loading component '{component}' from file: '{component_file}'")
        return pickle.load(open(component_file, "rb"))
    elif func is not None:
        if verbose:
           print(f"Calculating component '{component}' and saving to file: '{component_file}'")
        component = func()
        if not component_file.exists() or overwrite:
            component_file.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(component, open(component_file, "wb"))
        return component
    else:
        if verbose:
            print(f"Component '{component}' does not exist and no calculation instructions are given.")
        return None
 