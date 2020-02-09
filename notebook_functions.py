"""
functions to help with keeping the notebook uncluttered
(and allow testing of the functions)
"""
import re
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool

from rpy2 import robjects as robj
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

from bct.algorithms.modularity import community_louvain
from bct.algorithms.clustering import clustering_coef_wu_sign
from bct.algorithms.centrality import participation_coef_sign


def _read_adj_matrix(file):
    """read the adjacency matrix file
    """
    return pd.read_csv(file, index_col=0, sep='\t')


def _compare_networks(row):
    """Identify whether the correlation in this row
    represents a "within" network correlation or
    a "between" network correlation
    
    Parameters
    ----------
    row : pandas.Series
    
    Returns
    -------
    str
    """
    if row['source_network'] == row['target_network']:
        return 'within'
    else:
        return 'between'

def _fishers_r_to_z(r):
    return np.arctanh(r)


def _fishers_z_to_r(z):
    """
    Parameters
    ----------
    z : float or numpy.array(like)

    Returns
    -------
    float or numpy.array(like)
    """
    return np.tanh(z)


def _adj_to_edge(df):
    """convert adjacency matrix to edge list
    """
    return nx.to_pandas_edgelist(nx.from_pandas_adjacency(df))


def _identify_within_between(edge_df):
    edge_df.dropna(inplace=True) # drop self connections
    # isolate the networks from the longform schaefer names
    edge_df['source_network'] = edge_df['source'].str.split('-').apply(lambda x: x[1])
    edge_df['target_network'] = edge_df['target'].str.split('-').apply(lambda x: x[1])
    # specify whether the network-network correlation is within or between
    edge_df['network_connection'] = edge_df[['source_network', 'target_network']].apply(_compare_networks, axis=1)

    return edge_df


def _condense_within_between(proc_edge_df):
    info_df = proc_edge_df.groupby(['network_connection', 'source_network']).describe().T.loc[('weight', 'mean'), :].to_frame()
    info_df.columns = info_df.columns.droplevel()
    info_df.reset_index(inplace=True)

    return info_df


def summarize_network_correlations(file, participant_id):
    """
    Parameters
    ----------
    file : str
        filename of symmetric adjacency matrix
    participant_id : participant_id
        participant_id identifier

    Returns
    -------
    info_df : pandas.DataFrame
        dataframe with a within and between network measure for each network
    """
    adj_z_df = _read_adj_matrix(file)
    adj_r_df = _fishers_z_to_r(adj_z_df)
    
    edge_df = _adj_to_edge(adj_r_df)

    proc_edge_df = _identify_within_between(edge_df)

    info_df = _condense_within_between(proc_edge_df)

    info_df['participant_id'] = [participant_id] * info_df.shape[0]

    print(f"finished {participant_id}")

    return info_df


def _subtract_matrices(file1, file2):
    """Take the difference between two adjacency matrices
    """
    df1 = _read_adj_matrix(file1)
    df2 = _read_adj_matrix(file2)
    diff_df = df1 - df2
    diff_edge_df = _adj_to_edge(diff_df)
    diff_edge_df.dropna(inplace=True)

    return diff_edge_df


def _proc_diff_df(edge_df, participant_id):
    """slightly redundant with summarize_network
    """
    proc_edge_df = _identify_within_between(edge_df)

    info_df = _condense_within_between(proc_edge_df)

    info_df['participant_id'] = [participant_id] * info_df.shape[0]

    info_df.reset_index(inplace=True)

    print(f"finished {participant_id}")
    
    return info_df


def calc_diff_matrices(file1, file2, participant_id):
    """calculate the average within/between network correlation differences
    between two adjacency matrices
    """  
    edge_df = _subtract_matrices(file1, file2)
    
    out_df = _proc_diff_df(edge_df, participant_id)

    # translate the difference back to a Pearson's R.
    out_df['mean_r'] = _fishers_z_to_r(out_df['mean'])
    
    return out_df


def z_score_cutoff(arr, thresh):
    """give correlations very close to 1 a more reasonable z-score
    0.99 r == 2.647 z
    (this is the max z score I would be interested in,
     anything above does not explain meaningful differences)
    """
    return arr.clip(-thresh, thresh)


def _combine_adj_matrices_wide(dfs):
    names = dfs[0].columns
    upper_idxs = np.triu_indices(len(names), k=1)
    new_colnames = ['__'.join([names[i], names[j]]) for i, j in zip(*upper_idxs)]
    wide_df = pd.DataFrame(np.array([df.values[upper_idxs] for df in dfs]), columns=new_colnames)

    return wide_df


def bind_matrices(objs, label):
    """combine all adjacency matrices to a wide format where
    each column represents a unique roi-roi pair and each row represents
    an observation from a participant_id.
    """

    dfs = [_read_adj_matrix(obj.path) for obj in objs]
    participant_ids = [obj.entities['subject'] for obj in objs]
    wide_df = _combine_adj_matrices_wide(dfs)
    wide_df['participant_id'] = participant_ids
    wide_df['task'] = [label] * len(participant_ids)

    return wide_df


STATS = importr('stats')
BASE = importr('base')

def _run_model(df, col):
    filt_df = df[['participant_id', 'task', col]]
    filt_df = filt_df.rename({col: 'correlation'}, axis=1)
    collector_dict = {'source_target': None, 'p_value': None, 'estimate': None}
    with localconverter(robj.default_converter + pandas2ri.converter):
        r_df = robj.conversion.py2rpy(filt_df)
    model = STATS.lm(formula='correlation ~ task', data=r_df)
    summary = BASE.summary(model)
    res = summary.rx2('coefficients')
    res = np.asarray(res) # sometimes is a FloatMatrix
    if res.ndim == 2:
        p_val = res[1][3]
        estimate = res[1][0]
    else:
        p_val = res[7] # manually check this
        estimate = res[1] # manually check this
    # print(col)
    # print(res)
    collector_dict['source_target'] = col
    collector_dict['p_value'] = p_val
    collector_dict['estimate'] = estimate
    return collector_dict


def model_corr_diff_mt(wide_df, n_threads):
    """setup to run linear regression for every roi-roi pair
    """
    cols = set(wide_df.columns)
    # I do not want to iterate over these columns
    cols = list(cols - set(["task", "participant_id", "nan_rois", "num_nan_rois"]))
    args = [(wide_df[['participant_id', 'task', col]], col) for col in cols]
    # run this in parallel to speed up computation
    with Pool(n_threads) as p:
        sep_dicts = p.starmap(_run_model, args)
    dict_collector = {
            k: [d.get(k) for d in sep_dicts]
            for k in set().union(*sep_dicts)}
    model_df = pd.DataFrame.from_dict(dict_collector)
    return model_df


def _flip_hemisphere_network(w):
    """schaefer roi names look like LH-ContA_SPL_1,
    this changes them to look like ContA-LH_SPL_1
    """
    comps = w.split('-')
    return '-'.join([comps[1], comps[0]]) + '_' + ''.join(comps[2:])


def _edge_to_adj(df, measure):
    df[["source", "target"]] = df["source_target"].str.split('__', expand=True)
    return nx.to_pandas_adjacency(nx.from_pandas_edgelist(df, edge_attr=measure), weight=measure)


def _sort_columns(df):
    df_rename = df.rename(_flip_hemisphere_network, axis=1).rename(_flip_hemisphere_network, axis=0)
    col_list = df_rename.columns.tolist()
    col_list.sort()

    return df_rename.loc[col_list, col_list]


def make_symmetric_df(df, measure):
    tmp_df = _edge_to_adj(df, measure)
    pretty_df = _sort_columns(tmp_df)
    
    return pretty_df


def _make_pretty_schaefer_heatmap(adj_df, **hm_kwargs):
    if adj_df.shape[0] != adj_df.shape[1]:
        raise ValueError("The dataframe is not square")

    # get the network assignments (assumed name is like ContA-LH_SPL_1)
    networks = adj_df.columns.str.split('-', n=1, expand=True).get_level_values(0)
    # at what indices do the networks change (e.g., go from ContA to ContB)
    # https://stackoverflow.com/questions/19125661/find-index-where-elements-change-value-numpy
    network_change_idxs = np.where(np.roll(networks,1)!=networks)[0]
    # find the midpoint index in each network to place a label
    tmp_idx = np.append(network_change_idxs, adj_df.shape[0])
    midpoints = (tmp_idx[1:] + tmp_idx[:-1]) // 2
    
    # create figure axes to plot onto
    fig, ax = plt.subplots(figsize=(24, 20))
    # make the heatmap
    ax = sns.heatmap(adj_df, ax=ax, **hm_kwargs)
    # add horizontal (hlines) and vertical (vlines) to delimit networks
    ax.hlines(network_change_idxs, xmin=0, xmax=adj_df.shape[0])
    ax.vlines(network_change_idxs, ymin=0, ymax=adj_df.shape[1])
    
    # remove ticklines on the axes
    ax.tick_params(length=0)
    
    # add network labels to the y-axis
    ax.set_yticks(midpoints)
    ax.set_yticklabels(networks.unique(), fontdict={'fontsize': 'large', 'fontweight': 'heavy'}, va="center")

    # add network labels to the x-axis
    ax.set_xticks(midpoints)
    ax.set_xticklabels(networks.unique(), fontdict={'fontsize': 'large', 'fontweight': 'heavy'}, ha="center")


def _identify_nan_entries(adj_df):
    adj_arr = adj_df.values
    np.fill_diagonal(adj_arr, 0)

    rois = adj_df.columns
    # assuming nans are constant through a roi
    # also assuming the first roi is not nan
    nan_idxs = np.argwhere(np.isnan(adj_arr[0,:]))
    nan_rois = rois[nan_idxs]

    return nan_idxs, nan_rois, len(nan_rois)


def _run_graph_theory_measure(adj_df, graph_func, **graph_kwargs):
    nan_idxs, nan_rois, num_nan_rois = _identify_nan_entries(adj_df)
    adj_df = adj_df.drop(labels=nan_rois, axis=1).drop(labels=nan_rois, axis=0)
    adj_arr = adj_df.values

    np.fill_diagonal(adj_arr, 0)

    if 'ci' in graph_kwargs.keys() and nan_rois.any():
        graph_kwargs['ci'] = np.delete(graph_kwargs['ci'], nan_idxs)

    graph_output = graph_func(adj_arr, **graph_kwargs)

    return nan_rois, num_nan_rois, graph_output


def calc_modularity(file, participant_id, task):
    # read the file
    adj_df = _read_adj_matrix(file)
    
    # rename and sort the columns
    adj_df = _sort_columns(adj_df)
    
    adj_r_df = _fishers_z_to_r(adj_df)

    # pass the original community classification (based on schaefer)
    orig_ci = adj_r_df.columns.str.split('-', n=1, expand=True).get_level_values(0)

    # run modularity
    nan_rois, num_nan_rois, (ci, modularity) = _run_graph_theory_measure(
        adj_r_df, community_louvain, B='negative_asym', ci=orig_ci
    )

    num_ci = len(np.unique(ci))

    result_dict = {
        'nan_rois': nan_rois,
        'num_nan_rois': num_nan_rois,
        'ci': ci,
        'num_ci': num_ci,
        'modularity': modularity,
        'participant_id': participant_id,
        'task': task,
    }

    return result_dict


def calc_clustering_coef(file, participant_id, task):
    # read the file
    adj_df = _read_adj_matrix(file)
    
    # rename and sort the columns
    adj_df = _sort_columns(adj_df)
    
    adj_r_df = _fishers_z_to_r(adj_df)

    nan_rois, num_nan_rois, cluster_coefs = _run_graph_theory_measure(
        adj_r_df, clustering_coef_wu_sign, coef_type='constantini'
    )

    # combine cluster coefs with their respective rois
    track_idx = 0
    cluster_coef_dict = {}
    for roi in adj_df.columns:
        if roi in list(nan_rois):
            cluster_coef_dict[roi] = np.nan
        else:
            cluster_coef_dict[roi] = cluster_coefs[track_idx]
            track_idx += 1

    cluster_coef_dict['nan_rois'] = nan_rois
    cluster_coef_dict['num_nan_rois'] = num_nan_rois
    cluster_coef_dict['participant_id'] = participant_id
    cluster_coef_dict['task'] = task

    return cluster_coef_dict


def calc_participation_coef(file, participant_id, task):
    # read the file
    adj_df = _read_adj_matrix(file)
    
    # rename and sort the columns
    adj_df = _sort_columns(adj_df)
    
    adj_r_df = _fishers_z_to_r(adj_df)

    ci = adj_r_df.columns.str.split('-', n=1, expand=True).get_level_values(0)

    nan_rois, num_nan_rois, (pos_p_coef, neg_p_coef) = _run_graph_theory_measure(
        adj_r_df, participation_coef_sign, ci=ci
    )
    # combine participation coefs with their respective rois
    track_idx = 0
    participation_coef_dict = {}
    for roi in adj_r_df.columns:
        roi_pos = roi + '_pos'
        roi_neg = roi + '_neg'
        if roi in list(nan_rois):
            participation_coef_dict[roi_pos] = np.nan
            participation_coef_dict[roi_neg] = np.nan
        else:
            participation_coef_dict[roi_pos] = pos_p_coef[track_idx]
            participation_coef_dict[roi_neg] = neg_p_coef[track_idx]
            track_idx += 1
    
    participation_coef_dict['nan_rois'] = nan_rois
    participation_coef_dict['num_nan_rois'] = num_nan_rois
    participation_coef_dict['participant_id'] = participant_id
    participation_coef_dict['task'] = task

    return participation_coef_dict