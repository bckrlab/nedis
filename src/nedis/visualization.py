import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import sklearn.manifold
import sklearn.preprocessing
import scipy.spatial
import numpy as np

import scipy.interpolate


def visualize_feature_clusters(
        values, clustering=None, heatmap_kwargs=None, axline_kwargs=None, ax=None):
    
    if ax is None:
        ax = plt.gca()

    if clustering is None:
        row_labels = np.zeros(values.shape[0])
        col_labels = np.zeros(values.shape[1])

    elif hasattr(clustering, "row_labels_"):
        row_labels = clustering.row_labels_
        col_labels = clustering.column_labels_
        
    elif hasattr(clustering, "labels_"):
        row_labels = clustering.labels_
        col_labels = clustering.labels_
        
    elif isinstance(clustering, tuple):
        row_labels, col_labels = clustering

    else:
        row_labels = clustering
        col_labels = clustering

    if heatmap_kwargs is None:
        heatmap_kwargs = dict(center=0, vmin=-1, vmax=1)
        
    if axline_kwargs is None:
        axline_kwargs = dict(color="lightgrey")
    
    if clustering is None:
        order_row = np.arange(values.shape[0])
        order_col = np.arange(values.shape[0])
        
    else:
        order_row = np.argsort(row_labels)
        order_col = np.argsort(col_labels)
        
    sns.heatmap(values[order_row, :][:, order_col], ax=ax, **heatmap_kwargs)

    row = 0
    for i, l in enumerate(row_labels[order_row]):
        if l > row:
            ax.axhline(i, **axline_kwargs)
            row = l

    col = 0
    for i, l in enumerate(col_labels[order_col]):
        if l > col:
            ax.axvline(i, **axline_kwargs)
            col = l


def plot_cordis_cluster(
        cluster, 
        nodes_pos, 
        correlation_matrix, 
        disruption_matrices=None,
        labels=None,
        correlation_threshold=-1, 
        nodes_mode="all nodes",
        edges_mode="correlation",
        disruption_norm=None,
        ax=None, 
        verbose=0):
    
    if cluster is None:
        cluster = {
            "rows":range(nodes_pos.shape[0]), 
            "columns":range(nodes_pos.shape[0])
        }
    
    if ax is None:
        ax = plt.gca()

    # correlation_matrix = np.mean(correlation_matrices, axis=0)
#     correlation_matrix = correlation_matrices[cluster["reference_data"]]
#     nodes_pos = coordinates[cluster["reference_data"]]
    
    if "edges" in cluster:
        edges = [
            (src, dst) for src, dst in zip(*cluster["edges"].nonzero())
            if src != dst and 
                abs(correlation_matrix[src, dst]) >= correlation_threshold]
        rows = list({src for src, _ in edges})
        columns = list({dst for _, dst in edges})
    else:
        rows = cluster["rows"]
        columns = cluster["columns"]
        edges = [
            (src,dst) 
            for src in range(nodes_pos.shape[0]) 
            for dst in range(nodes_pos.shape[0])
            if \
                src != dst and \
                ((src in rows and dst in columns) \
                or (src in columns and dst in rows)) \
                and abs(correlation_matrix[src, dst]) >= correlation_threshold
        ]
    
    if nodes_mode == "all nodes":
        nodes_pos = {
            i:p 
            for i, p in enumerate(nodes_pos)}
    elif nodes_mode == "cluster nodes":
        nodes_pos = {
            i:p 
            for i, p in enumerate(nodes_pos) 
            if i in cluster["rows"] or i in cluster["columns"]}
    elif nodes_mode == "cluster circle":
        g = nx.Graph()
        g.add_nodes_from(cluster["rows"])
        nodes_pos = nx.circular_layout(g)
    else:
        raise ValueError(f"Unknown nodes mode: {nodes_mode}")

    if verbose > 0:
        print(f"Number of edges: {len(edges)}")
 
    def nodes_node_size(g, n, d):
        if n in rows and n in columns:
            return 40
        elif n in rows:
            return 40
        elif n in columns:
            return 40
        else:
            return 10

    def nodes_node_color(g, n, d):
        if n in rows and n in columns:
            return "black"
        elif n in rows:
            return "red"
        elif n in columns:
            return "blue"
        else:
            return "grey"
    
    if edges_mode == "correlation":
        def edges_width(g, src, dst, d):
            abscor = abs(correlation_matrix[src, dst])
            if src == dst and not np.isclose(correlation_matrix[src,dst], 1):
                print(correlation_matrix[src,dst])
            t = 0.1
            if abscor < t:
                return 0
            else:
                return (abscor - t) / (1 - t) * 5

        def edges_edge_color(g, src, dst, d):
            cmap = sns.color_palette(palette="vlag", as_cmap=True)
            cor = correlation_matrix[src, dst]
            c = (cor + 1) / 2
            color = cmap(c)
            color = (*color[:-1], 0.7)
            return color
        
    elif edges_mode == "disruption":
        
        if disruption_norm is None:
            dis = np.max(np.abs(np.median(disruption_matrices, axis=0)))
            disruption_norm = plt.Normalize(-dis, dis)
        
        def edges_width(g, src, dst, d):
            dis = np.median(disruption_matrices[:, src, dst])
            return np.abs((disruption_norm(dis) - 0.5)) * 2 * 10

        def edges_edge_color(g, src, dst, d):
            cmap = sns.color_palette(palette="vlag", as_cmap=True)
            dis = np.median(disruption_matrices[:, src, dst])
            color = cmap(disruption_norm(dis))
            color = (*color[:-1], np.abs((disruption_norm(dis) - 0.5)) * 2)
            return color
        
    else:
        raise ValueError(f"Unknown edges mode: {edges_mode}") 

    nx_plot(
        # nodes=np.arange(nodes_pos.shape[0]),
        nodes_pos=nodes_pos,
        nodes_args=dict(
            node_size=nodes_node_size,
            node_color=nodes_node_color,
        ),
        edges=edges,
        edges_args=dict(
            width=edges_width,
            edge_color=edges_edge_color,
        ),
        nodes_labels=labels,
        nodes_labels_args={
            "font_size": 2
        },
        ax=ax)

    ax.axis("off")
        

def visualize_data(
        X, y, entities, 
        coordinates=0, 
        correlation_matrices=None, 
        correlation_threshold=0.2,
        tsne_perplexity=30,
        tsne_learning_rate=200,
        mode="network", random_state=None):
    
    y_unique = np.unique(y)
    

    if correlation_matrices is None:
        print("Computing correlation matrices ...")
        correlation_matrices = {}
        for yy in y_unique: 
            cor = np.corrcoef(X[y == yy,:], rowvar=False)
            correlation_matrices[yy] = cor

    if mode == "heatmap":
        print("Visualizing heatmaps ...")

        # order correlation matrices by hierarchical clustering
        cm = correlation_matrices[y_unique[coordinates]]
        g = sns.clustermap(abs(cm))
        plt.close()
        reordered_columns = g.dendrogram_col.reordered_ind
        reordered_rows = g.dendrogram_row.reordered_ind

        correlation_matrices_ordered = {}
        for yy in y_unique:
            cor = correlation_matrices[yy]
            cor_ordered = cor[reordered_rows, :][:, reordered_columns]
            correlation_matrices_ordered[yy] = cor_ordered

        fig, axes = plt.subplots(1, len(y_unique), figsize=(6 * len(y_unique), 4))
        for i, yy in enumerate(y_unique):
            ax = axes[i]
            cor = correlation_matrices_ordered[yy]
            visualize_feature_clusters(
                cor, ax=ax, heatmap_kwargs=dict(cmap="vlag", center=0, vmin=-1, vmax=1))
        return fig, axes, correlation_matrices, (reordered_rows, reordered_columns)

    elif mode == "network":

        if isinstance(coordinates, int):
            print("Computing coordinates ...")
            correlation_matrix = correlation_matrices[y_unique[coordinates]]
            tsne = sklearn.manifold.TSNE(
                n_components=2, random_state=random_state, init="pca", perplexity=tsne_perplexity, learning_rate=tsne_learning_rate)
            coordinates = tsne.fit_transform(abs(correlation_matrix))


        print("Visualizing networks ...")
        fig, axes = plt.subplots(
            1, len(y_unique), figsize=(4 * 2 * len(y_unique), 4 * 2))
        for i, yy in enumerate(y_unique):
            ax = axes[i]
            ax.axis("off")
            ax.set_title(yy)
            plot_cordis_cluster(
                None, 
                coordinates, 
                correlation_matrices[yy], 
                correlation_threshold=correlation_threshold,
                ax=ax)
        return fig, axes, correlation_matrices, coordinates

    else:
        raise ValueError(f"Unknown mode: {mode}")
    

def nx_plot(
        graph=None,
        nodes=None,
        nodes_pos=None,
        nodes_args=None,
        nodes_labels=None,
        nodes_labels_pos=None,
        nodes_labels_args=None,
        edges=None,
        edges_pos=None,
        edges_args=None,
        edges_labels=None,
        edges_labels_pos=None,
        edges_labels_args=None,
        ax=None):
    """More complete version of`networkx` plotting. 

    Notes:
    * Alpha for individual edges: set alpha in color using `edge_color`

    * NetworkX = 2.6.3
        * Self-loops are plotted all of a sudden but their formatting doesn't work as expected. 
          Luckily this does not influence non-self-loops:
          Link: https://github.com/networkx/networkx/issues/5106
    * NetworkX < 2.4:
        * Weirdness to make **edges transparent**:
            Set edge alpha to a array or a function (value does not matter),
            and then the set edge colors to RGBA:
            `edges_args=dict(edge_color=lambda g,src,dst,d: return (1,0,0,d['alpha']), alpha=lambda g,src,dst,d: 1)`
        * When using matplotlib.pyplot.subplots networkx adjusted axes by calling `plt.tick_params` causing unwanted
            behavior. This was fixed in networkx 2.4.

    Parameters
    ----------
    graph : [type]
        [description]
    nodes : [type]
        [description]
    nodes_pos : [type]
        [description]
    nodes_args : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels_pos : [type], optional
        [description] (the default is None, which [default_description])
    nodes_labels_args : [type], optional
        [description] (the default is None, which [default_description])
    edges : [type], optional
        [description] (the default is None, which [default_description])
    edges_pos : [type], optional
        [description] (the default is None, which [default_description])
    edges_args : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels_pos : [type], optional
        [description] (the default is None, which [default_description])
    edges_labels_args : [type], optional
        [description] (the default is None, which [default_description])
    
    Returns
    -------
    [type]
        [description]
    """

    # get axis
    if ax is None:
        ax = plt.gca()

    # init graph

    if graph is None or isinstance(graph, str):
        if graph == "bi" or graph == "di":
            g = nx.DiGraph()
        else:
            g = nx.Graph()
    else:
        g = graph

    # init nodes

    if graph is None and nodes is None:
        if nodes_pos is not None:
            if isinstance(nodes_pos, dict):
                nodes = nodes_pos.keys()
            else: 
                nodes = np.arange(nodes_pos.shape[0])
        else:
            raise Exception(
                "Either `graph` or `nodes` must be given. Both were `None`.")

    if nodes is not None:
        if isinstance(nodes, dict):
            nodes = [(k, v) for k, v in nodes.items()]
        g.add_nodes_from(nodes)

    # init edges

    if edges is not None:
        if isinstance(edges, dict):
            edges = [(*k, v) for k, v in edges.items()]
        g.add_edges_from(edges)

    # init positions

    def init_pos(pos):
        if pos is None:
            return nodes_pos
        elif callable(pos):
            return {n: pos(g, n, d) for n, d in g.nodes(data=True)}
        elif isinstance(pos, dict):
            return pos
        else:
            return {n: p for n, p in zip(g.nodes(), pos)}

    nodes_pos = init_pos(nodes_pos)
    nodes_labels_pos = init_pos(nodes_labels_pos)
    edges_pos = init_pos(edges_pos)
    edges_labels_pos = init_pos(edges_labels_pos)

    # init labels

    def init_nodes_labels(labels):
        if callable(labels):
            return {n: labels(g, n, d) for n, d in g.nodes(data=True)}
        else:
            return labels
    nodes_labels = init_nodes_labels(nodes_labels)

    def init_edges_labels(labels):
        if callable(labels):
            tmp = {
                (src, dst): labels(g, src, dst, d) 
                for src, dst, d in g.edges(data=True)}
            tmp = {k: v for k, v in tmp.items() if v is not None}  # filter "None" labels
            return tmp
        else:
            return labels
    edges_labels = init_edges_labels(edges_labels)

    # init layout arguments

    def init_node_args(args):
        if args is None:
            args = {}
        else:
            args = args.copy()
            for k, v in args.items():
                if callable(v):
                    args[k] = [v(g, n, d) for n, d in g.nodes(data=True)]
        if "ax" not in args:
            args["ax"] = ax
        return args

    nodes_args = init_node_args(nodes_args)
    nodes_labels_args = init_node_args(nodes_labels_args)

    def init_edges_args(args):
        if args is None:
            args = {}
        else:
            args = args.copy()
            for k, v in args.items():
                if callable(v):
                    args[k] = [v(g, src, dst, d) for src, dst, d in g.edges(data=True)]
        if "ax" not in args:
            args["ax"] = ax
        return args

    edges_args = init_edges_args(edges_args)
    edges_labels_args = init_edges_args(edges_labels_args)

    # draw nodes (allow for several of shapes for nodes)
    if "node_shape" in nodes_args and type(nodes_args["node_shape"]) is list:

        shapes = list(zip(range(len(g.nodes())), nodes_args["node_shape"]))
        unique_shapes = np.unique(nodes_args["node_shape"])

        for shape in unique_shapes:

            shape_idx = [i for i, s in shapes if s == shape]

            nodes = list(g.nodes())
            nodelist = [nodes[i] for i in shape_idx]

            shape_args = nodes_args.copy()
            del shape_args["node_shape"]

            for arg, _ in shape_args.items():
                if type(shape_args[arg]) is list:
                    shape_args[arg] = [shape_args[arg][i] for i in shape_idx]

            nx.draw_networkx_nodes(
                g, nodes_pos, nodelist=nodelist, node_shape=shape, **shape_args)
    else:
        nx.draw_networkx_nodes(g, nodes_pos, **nodes_args)

    # draw edges
    # print(g.nodes(data=True))
    # print(g.edges(data=True))
    # print(edges_args)
    nx.draw_networkx_edges(g, nodes_pos, **edges_args)

    # draw node labels
    if nodes_labels is not None:

        # rename args for compatibility to `nx.draw_networkx_labels`
        args = nodes_labels_args.copy()
        del args["ax"]
        for original_key, new_key in {
                "font_size": "fontsize",
                "font_color": "color",
                "font_family": "family",
                "font_weight": "weight"}.items():
            if original_key in args:
                args[new_key] = args[original_key]
                del args[original_key]

        # check if we have list args
        list_args = []
        for arg, value in list(args.items()):
            if isinstance(value, list) and len(value) == len(g.nodes):
                list_args.append((arg, value))
                del args[arg]

        for i, node in enumerate(g.nodes):
            ax.annotate(
                nodes_labels[node], 
                nodes_labels_pos[node], 
                **args, 
                **{a: v[i] for a, v in list_args})

    # draw edge labels
    if edges_labels is not None:
        nx.draw_networkx_edge_labels(
            g, pos=edges_labels_pos, edge_labels=edges_labels, **edges_labels_args)

    return g


def calculate_hull(
        points, 
        scale=1.1, 
        padding="extend", 
        n_interpolate=100, 
        interpolation="linear", 
        return_hull_points=False):
    """Inspired by: https://stackoverflow.com/a/17557853/991496"""
    
    if padding == "scale":
        # scaling based padding
        scaler = sklearn.preprocessing.StandardScaler()
        points_scaled = scaler.fit_transform(points) * scale
        hull_scaled = scipy.spatial.ConvexHull(points_scaled, incremental=True)
        hull_points_scaled = points_scaled[hull_scaled.vertices]
        hull_points = scaler.inverse_transform(hull_points_scaled)
        hull_points = np.concatenate([hull_points, hull_points[:1]])
    
    elif padding == "extend" or isinstance(padding, (float, int)):
        # extension based padding
        if padding == "extend":
            add = (scale - 1) * np.max([
                points[:,0].max() - points[:,0].min(), 
                points[:,1].max() - points[:,1].min()])
        else:
            add = padding
        points_added = np.concatenate([
            points + [0,add], 
            points - [0,add], 
            points + [add, 0], 
            points - [add, 0]])
        hull = scipy.spatial.ConvexHull(points_added, incremental=True)
        hull_points = points_added[hull.vertices]
        hull_points = np.concatenate([hull_points, hull_points[:1]])
    else:
        raise ValueError(f"Unknown padding mode: {padding}")
    
    # number of interpolated points
    nt = np.linspace(0, 1, n_interpolate)
    
    x, y = hull_points[:,0], hull_points[:,1]
    
    # ensures the same spacing of points between all hull points
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t /= t[-1]

    # interpolation types
    # TODO: figure out a good one ... I'd love linear with rounded corners -.-
    if interpolation is None or interpolation == "linear":
        x2 = scipy.interpolate.interp1d(t, x, kind="linear")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="linear")(nt)
    elif interpolation == "quadratic":
        x2 = scipy.interpolate.interp1d(t, x, kind="quadratic")(nt)
        y2 = scipy.interpolate.interp1d(t, y, kind="quadratic")(nt)
    elif interpolation == "cubic":
        x2 = scipy.interpolate.CubicSpline(t, x, bc_type="periodic")(nt)
        y2 = scipy.interpolate.CubicSpline(t, y, bc_type="periodic")(nt)
    else:
        x2 = interpolation(t, x, nt)
        y2 = interpolation(t, y, nt)
    
    X_hull = np.concatenate([x2.reshape(-1,1), y2.reshape(-1,1)], axis=1)
    if return_hull_points:
        return X_hull, hull_points
    else:
        return X_hull
    

def grouped_spline_plot(
        x, y, 
        groups=None,
        xrange=None, 
        n_points=100, 
        spline_function=None, 
        mode="median", 
        color="#1f77b4", 
        label=None,
        plot_center_kwargs=None,
        plot_lower_kwargs=None,
        plot_upper_kwargs=None,
        fill_kwargs=None,
        scatter_kwargs=None,
        errorbar_kwargs=None, 
        groups_kwargs=None,
        ax=None):

    if ax is None:
        ax = plt.gca()

    x_groups = np.unique(x)

    x = np.array([np.argwhere(v == x_groups)[0][0] for v in x])

    u_x = np.arange(len(x_groups))

    if xrange is None:
        xrange = [np.min(u_x), np.max(u_x)]

    center_y = []
    upper_y = []
    lower_y = []

    if mode == "median":
        for v in u_x:
            select = x == v
            center_y.append(np.nanmedian(y[select]))
            upper_y.append(np.nanpercentile(y[select], 75, axis=0))
            lower_y.append(np.nanpercentile(y[select], 25, axis=0))
    elif mode == "mean":
        for v in u_x:
            select = x == v
            mean_y = np.nanmean(y[select])
            std_y = np.nanstd(y[select])
            center_y.append(mean_y)
            upper_y.append(mean_y + std_y)
            lower_y.append(mean_y - std_y)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    center_y = np.array(center_y)
    upper_y = np.array(upper_y)
    lower_y = np.array(lower_y)

    if spline_function is None:
        spline_function = scipy.interpolate.Akima1DInterpolator

    center_x_inter = np.linspace(*xrange, n_points)

    f = spline_function(u_x, center_y)
    center_y_inter = f(center_x_inter)

    f = spline_function(u_x, lower_y)
    lower_y_inter = f(center_x_inter)

    f = spline_function(u_x, upper_y)
    upper_y_inter = f(center_x_inter)

    def init_kwargs(kwargs, **kwargs_default):
        if kwargs is None:
            kwargs = {}
        return {**kwargs_default, **kwargs}

    plot_center_kwargs = init_kwargs(
        plot_center_kwargs, linewidth=3, color=color, label=label, zorder=8)
    plot_lower_kwargs = init_kwargs(
        plot_lower_kwargs, linewidth=3, color=color, linestyle=":", zorder=8)
    plot_upper_kwargs = init_kwargs(
        plot_upper_kwargs, linewidth=3, color=color, linestyle=":", zorder=8)
    fill_kwargs = init_kwargs(
        fill_kwargs, color=color, alpha=0.1, zorder=7)

    scatter_kwargs = init_kwargs(
        scatter_kwargs, color=color, zorder=10)
    errorbar_kwargs = init_kwargs(
        errorbar_kwargs, color=color, linewidth=0, elinewidth=1, zorder=9)

    groups_kwargs = init_kwargs(
        groups_kwargs, linewidth=5, alpha=0.2, color="grey", zorder=5)

    # draw individual lines
    if groups is not None:
        for g in np.unique(groups):
            f = spline_function(x[groups == g], y[groups == g])
            y_inter = f(center_x_inter)
            ax.plot(center_x_inter, y_inter, **groups_kwargs)

    ax.plot(center_x_inter, center_y_inter, **plot_center_kwargs)
    ax.plot(center_x_inter, lower_y_inter, **plot_lower_kwargs)
    ax.plot(center_x_inter, upper_y_inter, **plot_upper_kwargs)
    ax.fill_between(center_x_inter, lower_y_inter, upper_y_inter, **fill_kwargs)

    ax.scatter(u_x, center_y, **scatter_kwargs)
    ax.errorbar(
        u_x, center_y,
        yerr=np.array([center_y - lower_y, upper_y - center_y]),
        **errorbar_kwargs)

    ax.set_xticks(u_x)
    ax.set_xticklabels(x_groups)

    return ax
