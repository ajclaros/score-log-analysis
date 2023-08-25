#!/usr/bin/env python3
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import re
import seaborn as sns
from matplotlib import cm
from graphviz import *

def create_dfs(fish_type):
    """
    Creates multiple dataframes for each fish type:
    behavior_time_df: time of behavior, columns: behavior, behavioral category, duration,
                                                 average_duration, stderr_duration, prob,
                                                    average_prob, stderr_prob
    transition_df: transition matrix, columns: behavior, behavior_next, behavioral category,
                                                counts, average_counts, stderr_counts, prob,
                                                average_prob, stderr_prob
    """
    files = os.listdir(f"./data/{fish_type}")
    if not os.path.exists(f"./data/{fish_type}/converted_data"):
        os.makedirs(f"./data/{fish_type}/converted_data")
    csv = [filename for filename in files if filename.endswith(".csv") or filename.endswith(".tsv")]
    df = {data: pd.read_csv(path.join("data", fish_type, data)) for data in csv}
    keys = list(df.keys())


    behavior_time = dict()
    transition_matrix = dict()
    transition_probabilities = dict()
    behaviors = []
    categories = []
    single_probabilities = []
    behavior_counts = []

    for i, (name, frame) in enumerate(df.items()):
        print(df[name].columns)
        transition_matrix[name] = df[name].groupby(['behavior', 'behavior_next', 'behavioral category']).count()
        transition_matrix[name].rename(columns={'duration': 'counts'}, inplace=True)
        transition_matrix[name].drop(['subject'], axis=1, inplace=True)
        behavior_time[name] = df[name].groupby(['behavior','behavioral category']).agg({'duration':sum})
        behaviors.append(frame['behavior'].unique())
        categories.append(frame['behavioral category'].unique())
    # merge all files into a single dataframe for each type of data
    transition_df = pd.concat(transition_matrix.values(), axis=1, sort=False)
    behavior_time_df = pd.concat(behavior_time.values(), axis=1, sort=False)
    transition_df.fillna(0, inplace=True)
    behavior_time_df.fillna(0, inplace=True)
    transition_df.reset_index(level=[0,1,2], inplace=True)
    behavior_time_df.reset_index(level=[0,1], inplace=True)
    # print(transition_df)
    # print(behavior_time_df)
    # return
    # print(transition_df)
    # return transition_df
    transition_df['average_counts'] = transition_df.groupby(['behavior', 'behavior_next', 'behavioral category'])['counts'].transform('mean')
    transition_df['stderr_counts'] = transition_df.groupby(['behavior', 'behavior_next', 'behavioral category'])['counts'].transform('sem')
    transition_df['prob'] = transition_df['counts']/transition_df['counts'].sum()
    transition_df['average_prob'] = transition_df.groupby(['behavior', 'behavior_next', 'behavioral category'])['counts'].transform('mean')
    transition_df['stderr_prob'] = transition_df.groupby(['behavior', 'behavior_next', 'behavioral category'])['counts'].transform('sem')

    behavior_time_df['average_duration'] = behavior_time_df.groupby(['behavior', 'behavioral category'])['duration'].transform('mean')
    behavior_time_df['stderr_duration'] = behavior_time_df.groupby(['behavior', 'behavioral category'])['duration'].transform('sem')
    behavior_time_df['prob'] = behavior_time_df['duration']/behavior_time_df['duration'].sum()
    behavior_time_df['average_prob'] = behavior_time_df.groupby(['behavior', 'behavioral category'])['prob'].transform('mean')
    behavior_time_df['stderr_prob'] = behavior_time_df.groupby(['behavior', 'behavioral category'])['prob'].transform('sem')
    # return behavior_time_df, transition_df

    # ----------------
    # fill in missing values (Maybe don't for single trial data)
    # transition_df.fillna(0, inplace=True)
    # behavior_time_df.fillna(0, inplace=True)
    # ----------------
    behavior_time_df.to_csv(path.join("data", fish_type, "converted_data", "behavior_time.csv"), index=False)
    transition_df.to_csv(path.join("data", fish_type, "converted_data", "transition_df.csv"), index=False)


def get_avg_stderr(df, columnname):
    """
    Get average and standard error for each row
    """
    temp = df.copy()
    temp['avg']=0.0
    temp['stderr']=0.0
    temp.reset_index(inplace=True)
    obs_columns = np.array([column for column in df.columns if re.search(r'\d', column)])
    for j, row in enumerate(df.iterrows()):
        temp.at[j, columnname+'_avg'] = row[1][obs_columns].mean()
        temp.at[j, columnname+'_stderr'] = row[1][obs_columns].sem()
    return temp

def import_data(columns):
    """
    Import data as a dictionary of dataframes
    """
    df = dict()
    data_folder = path.join("data")
    filenames = [filename for filename in os.listdir("data") if filename.endswith(".csv") or filename.endswith(".tsv")]
    for filename in filenames:
        name, ext = filename.split(".")
        if ext == "csv":
            temp_df = pd.read_csv(path.join(data_folder, filename))
            df[name] = temp_df[columns]
        elif ext == "tsv":
            temp_df = pd.read_csv(path.join(data_folder, filename), sep="\t")
            df[name] = temp_df[columns]
    return df

def split_subject(df):
    """
    Split data into different folders based on the subject
    """
    filenames = list(df.keys())
    subjects = [df[filename]['subject'].unique()[0] for filename in filenames]
    # create folders for each subject
    for subject in subjects:
        if not os.path.exists(f"./data/{subject}"):
            os.makedirs(f"./data/{subject}")
    for filename in filenames:
        frame = df[filename]
        if len(frame.groupby('subject'))>1:
            # split data into different subjects
            for name, group in frame.groupby('subject'):
                if name == 'nil':
                    print(f"Missing Subject found in {filename}")
                    continue
                else:
                    pathname = path.join("data", name, filename+".csv")
                    group.to_csv(pathname, index=False)
        else:
            # save data into the same folder
            pathname = path.join("data", frame['subject'].unique()[0], filename+".csv")
            frame.to_csv(pathname, index=False)
def categorize_behaviors(df, behavior_classifications):
    """
    Categorize behaviors into different classes
    """
    for behavior_class in behavior_classifications.keys():
        for behavior in behavior_classifications[behavior_class]:
            df.loc[df['Behavior']==behavior, 'Behavioral category'] = behavior_class
    return df
def behavior_durations(df):
    """
    Calculate the duration of each behavior
    """
    df['behavior_next'] = df['Behavior'].shift(-1)
    df['time_next'] = df['Time'].shift(-1)
    df['duration'] = df['time_next'] - df['Time']
    df = df.dropna()
    return df

def normalize_terms(df, combined_terms):
    """
    Normalize terms in the dataframe
    combined_terms: dictionary of terms with keys as the normalized term and values as the list of terms to be combined

    """
    for key in combined_terms.keys():
        df.loc[df['Behavior'].isin(combined_terms[key]), 'Behavior'] = key
    return df


def create_transition_matrix(fish_type):
    transition_df = pd.read_csv(f"data/{fish_type}/converted_data/transition_df.csv")
    # rows are the current behavior, columns are the next behavior
    matrix = pd.pivot_table(transition_df, values='counts', index=['behavior'], columns=['behavior_next'])
    matrix.to_csv(f"data/{fish_type}/converted_data/transition_matrix.csv")



def create_state_model(fish_type, behavior_classifications, category_colors, prob_threshold=0.001,
                       edge_color='slategray', edge_scale=20,node_scale=2, displayEdgeLabels=True):
    print(f"Creating graph for {fish_type}")
    ## finish next session
    categories = list(behavior_classifications.keys())
    transition_df = pd.read_csv(f"data/{fish_type}/converted_data/transition_df.csv")
    behavior_time_df = pd.read_csv(f"data/{fish_type}/converted_data/behavior_time.csv")
    # print(behavior_time_df.groupby('behavioral category'))
    g = Digraph('G', engine='dot')
    # g.graph_attr['rankdir'] = 'TB'
    # g.graph_attr['overlap'] = 'scale'
    # g.graph_attr['compound'] = 'true'
    g.attr(fixed_size='false', overlap='scale', size=str(100),
           packMode='clust', compound='true', label=f'{fish_type}: Duration >{prob_threshold}',
           fontname='fira-code', labelloc='t', fontsize='15')
# , fontcolor='white', bgcolor='black')


    df = {}
    for category in behavior_classifications.keys():
        df[category] = {}
        try:
            df[category]['behavior_df'] = behavior_time_df.groupby('behavioral category').get_group(category)
            df[category]['transition_df'] = transition_df.groupby('behavioral category').get_group(category)
        except KeyError:
            print(f"{category} not found in {fish_type}")
            continue
    # print(df)
    for i, category in enumerate(df.keys()):
        if 'behavior_df' not in df[category].keys():
            continue
        with g.subgraph(name=f'cluster_{category}') as c:
            c.attr(label=category, fontname='fira-code', labelloc='b', color='black')
            c.attr(cluster="true", fontcolor=category_colors[category])
            for j, row in df[category]['behavior_df'].iterrows():
                prob = np.round(row['prob'], 2)
                c.node(row['behavior'], label=f"{row['behavior']}\n{prob}", color=category_colors[category], fontcolor='white',
                       height=str(row['prob']), shape='circle', style='filled')
    for i, category in enumerate(df.keys()):
        if 'transition_df' not in df[category].keys():
            continue
        for j, row in df[category]['transition_df'].iterrows():
            scaled = edge_scale * row['prob']
            weight = np.round(row['prob'],3)
            if weight < prob_threshold:
                continue
            # g.edge(row['behavior'], row['behavior_next'], color=edge_color,
            #        penwidth=str(scaled), headxlabel=str(weight*1), labelfontcolor='Red')
            if displayEdgeLabels:
                g.edge(row['behavior'], row['behavior_next'], color=edge_color, label= str(weight), labelloc='c', fontsize='15', penwidth=str(scaled), labelfontcolor='red')
            else:
                g.edge(row['behavior'], row['behavior_next'], color=edge_color, penwidth=str(scaled))

    g.render(f"data/{fish_type}/converted_data/graph", view=True, quiet=True, format='svg')

def create_barplots(fish_type, custom_colors=None, use_duration=False):
    transition_df = pd.read_csv(f"data/{fish_type}/converted_data/transition_df.csv")
    behavior_time_df = pd.read_csv(f"data/{fish_type}/converted_data/behavior_time.csv")
    behavior_time_df.sort_index(level=['behavioral category'], inplace=True, ascending=False)
    behavior_time_df.sort_values(by=['behavioral category', 'duration'], inplace=True, ascending=True)
    print(behavior_time_df)
    # plot duration of each behavior, color coded by behavioral category
    fig, ax = plt.subplots(figsize=(10, 5))
    column = 'prob'
    if use_duration:
        column = 'duration'
    if not custom_colors:
        sns.barplot(x='behavior', y=column, hue='behavioral category', data=behavior_time_df, palette={
            'Aversive': 'blue',
            'Reproductive': 'forestgreen',
            'Aggressive': 'red'
        })
    else:
        sns.barplot(x='behavior', y=column, hue='behavioral category', data=behavior_time_df, palette=custom_colors)
    plt.savefig(f"data/{fish_type}/converted_data/barplot_{column}.png")
    plt.show()

def split_logs(behavior_classifications, combined_terms, extension='csv'):
    # dfs = {name: pd.read_csv(path.join('data')) for name in os.listdir('data') if name.endswith('.csv') or name.endswith('.tsv')}
    files = [name for name in os.listdir('data') if name.endswith('.csv') or name.endswith('.tsv')]
    dfs = {}
    if 'tsv' in extension:
        dfs = {name: pd.read_csv(path.join('data', name), sep='\t') for name in files}
    else:
        dfs = {name: pd.read_csv(path.join('data', name)) for name in files}
    for name, df in dfs.items():
        for group, frame in df.groupby('Subject'):
            frame = frame[['Time', 'Behavior', 'Behavioral category']]
            frame = categorize_behaviors(frame, behavior_classifications)
            frame = normalize_terms(frame, combined_terms)
            frame.to_csv(path.join('data', group, f'{group}_log.csv'), index=False)

