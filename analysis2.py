import importlib
import os
import re
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import utils

importlib.reload(sys.modules['utils'])

colors = {
    'Aggressive':'red',
    'Aversive':'blue',
    'Reproductive':'forestgreen'}
# pd.set_option('display.max_columns', 100)
columns = ['Time', 'Subject', 'Behavior', 'Behavioral category']
behavior_classifications = dict()
behavior_classifications['Aggressive'] = ['b peck', 'y peck']
behavior_classifications['Aversive'] = ["y flee", "b flee"]
behavior_classifications['Reproductive'] = ['b circle', 'y circle', 'y follow', 'b follow', 'Enter/Exit']
# normalize terms - combine multiple or mispelled into a single term
combined_terms = dict()
combined_terms['Enter/Exit'] = ['enter', 'exit', 'Pot Entry', 'Pot Exit']

colors = {
    'Aggressive':'red',
    'Aversive':'blue',
    'Reproductive':'forestgreen'}

    # transition_probabilities = pd.read_csv(f"data/{fish_type}/converted_data/transition_probabilites.csv", index_col=0).set_index('behavior')
    # G = nx.MultiDiGraph()
    # node_pos = dict()
    # category_spacing = 10
    # for i, (category, behaviors) in enumerate(behavior_classifications.items()):
    #     for j, behavior in enumerate(behaviors):
    #         node_pos[behavior] = (i*category_spacing, len(behaviors) -j)
    # for category, behaviors in behavior_classifications.items():
    #     for behavior in behaviors:
    #         behavior_time = behavior_time_probabilities.T['behavior']
    #         G.add_node(behavior, pos=node_pos[behavior], size=behavior_time[behavior]*1000, color=colors[category])

    # plt.figure(figsize=(10,10))
    # nx.draw(G, pos=node_pos, node_size=[G.nodes[node]['size'] for node in G.nodes()], node_color=[G.nodes[node]['color'] for node in G.nodes()], with_labels=True)
    # plt.show()









data = utils.import_data(columns)
folders = [folder for folder in os.listdir("data") if not folder.endswith(".csv") and not folder.endswith(".tsv")]
# keys = list(folders.keys())

for fish_type in data.keys():
    data[fish_type] = utils.normalize_terms(data[fish_type], combined_terms)
    data[fish_type] = utils.categorize_behaviors(data[fish_type], behavior_classifications)
    data[fish_type] = utils.behavior_durations(data[fish_type])
    data[fish_type] = data[fish_type][['Behavior', 'behavior_next', 'Behavioral category', 'duration', 'Subject']]
    data[fish_type].columns = [column.lower() for column in data[fish_type].columns]
utils.split_subject(data)
# utils.split_logs(behavior_classifications, combined_terms, extension='tsv')
for fish_type in folders:
    # utils.create_raster(fish_type, ['b flee'], colors, duration_behaviors=['Enter/Exit'])
#     # creates aggregated statistics for each fish type within ./data/{fish_type}/convereted_data/
#     # behavior_time.csv, transtion_df.csv
    print(fish_type)
    utils.create_dfs(fish_type)
    utils.create_transition_matrix(fish_type)
    # utils.create_state_model(fish_type, behavior_classifications, colors, displayEdgeLabels=False)
    # utils.create_barplots(fish_type, custom_colors=colors.values())




    # break
