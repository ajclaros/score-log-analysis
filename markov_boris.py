import numpy as np
import os
import pandas as pd
from graphviz import *
import decimal
def GetKey(val, dictionary):
    for key, value in dictionary.items():
        if val in value:
            return key
    return "key doesn't exist"
color ='yellow'
folder_name = "dyad"
data = {}
for csv in os.listdir('{}/{}/converted_data/'.format(folder_name, color)):
    if '.csv' not in csv:
        continue
    data[csv[:-4]] = pd.read_csv('{}/{}/converted_data/{}'.format(folder_name, color,csv),index_col=0)
colors = {'Aggressive':'red', 'Reproductive':'forestgreen','Aversive':'blue'}
text_color ={'Aggressive':'#913838', 'Aversive':'#1e1e60', 'Reproductive':'#1e34c1b'}
subgraph = {}
filenames = list(data.keys())
categories = data[filenames[0]]['Behavioral category'].unique()
behaviors = data[filenames[0]]['Behavior'].unique()
prob_files = [files for files in filenames if 'prob' in files]
columns = data[prob_files[0]].columns
associations = data[filenames[0]].groupby('Behavioral category')['Behavior'].apply(list)
transition_individual_probabilities = data['transition_probabilites'].groupby(['Behavior']).sum()
behavior_time_prob = data['behavior_time_probabilities']
for column in columns[2:-1]:

    graph = Digraph(name= column, format='png')
    graph.graph_attr['rankdir']='TB'
    graph.attr(fixed_size='false', overlap='scale', size=str(90), packMode='clust', compound='true', label='{}  {} >0.01\%'.format(color, column), fontname='fira-code')
    for i, category in enumerate(categories):
        subgraph['cluster_{}'.format(category)] = Digraph(name='cluster_{}'.format(category))
        subgraph['cluster_{}'.format(category)].attr(label='{}'.format(category), style='filled', color='white', labelloc='b')
    for i, behavior in enumerate(behaviors):
        graph.node(behavior)
    for behavior_class, behaviors in associations.iteritems():
        for item in behaviors:
            subgraph['cluster_{}'.format(behavior_class)].node(item, color=colors[behavior_class], height= str(behavior_time_prob[behavior_time_prob['Behavior']==item]*20), width=str(behavior_time_prob[behavior_time_prob['Behavior']==item]*20))
#            subgraph['cluster_{}'.format(behavior_class)].node_attr.update()
    associations_dict  = associations.to_dict()
    for behavior_class in subgraph.keys():
       graph.subgraph(subgraph[behavior_class])

    for index, row in data['transition_probabilites'].iterrows():
        edge_color= ''
        if GetKey(row[0], associations_dict) == GetKey(row[1], associations_dict):
            edge_color = colors[GetKey(row[0], associations_dict)]
            txtcolor = text_color[GetKey(row[0], associations_dict)]
        else:
            edge_color = 'slategray'
            txtcolor='black'
        temp_var= transition_individual_probabilities.loc[row[0]][column]
#        if temp_var <1e-04:
#            size_label='0'
#            continue
        size_label = "{}".format(round(row[column]/transition_individual_probabilities.loc[row[0]][column]*100, 2))

        if size_label=='0':
            continue
        else:
            graph.edge(row[0], row[1], color=edge_color, headxlabel= str('{}'.format(size_label)), labelloc='c', fontsize='15', fontcolor=txtcolor, penwidth=str(.1* float( size_label)))
    graph.render('{}/{}/converted_data/markov/{}_{}'.format(folder_name, color,color, column), format='png', quiet=True)
