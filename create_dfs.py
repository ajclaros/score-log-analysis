 import decimal
import os
import re
import pandas as pd
import numpy as np
from graphviz import *
def flatten(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

#edit group folder name
fish_type = "blue"
folder = 'dyad'


files = os.listdir('{}/{}'.format(folder, fish_type))
if not os.path.exists('{}/{}/converted_data'.format(folder, fish_type)):
    os.makedirs("{}/{}/converted_data".format(folder, fish_type))
csv = [x for x in files if 'tsv' in x]
df = [pd.read_csv('{}/{}/{}'.format(folder, fish_type, data), header=15, sep='\t') for data in csv]
g = Digraph(name = fish_type, engine='dot')
g.attr(label=fish_type, labelloc='t', )
behavior_time = []
transition_matrix = []
transition_probabilities = []
behaviors = []
categories = []
single_probabilities = []
for i, frame in enumerate(df):

    

    #Edit duplicates that you want to combine like:
    # isin(['Entry/Exit Pot', 'Entry /Exit Pot']), 'Pot', df[i]['Behavior'])
    # isin(['name_0', 'name_1', 'name_2']), 'combined_name', df[i]['Behavior])
    df[i]['Behavior'] = np.where(df[i]['Behavior'].isin(['Entry/Exit Pot', 'Entry /Exit Pot']), 'Pot', df[i]['Behavior'])

    #Edit behaviors example:
    #.isin(['flee']), 'Aversive', df[i]['Behavioral category'])
    #.isin(['behav_0', 'behav_1', 'behav_2']), 'behavior label', df[i]['Behavior])
    df[i]['Behavioral category'] = np.where(df[i]['Behavior'].isin(['Lead', 'Dig Substrate', 'Pot Dig',  'Foraging', 'Pot', 'Quiver', 'Scrape']), 'Reproductive', df[i]['Behavioral category'])
    df[i]['Behavioral category'] = np.where(df[i]['Behavior'].isin(['Chase', 'Border Fight', 'Lateral Side Display']), 'Aggressive', df[i]['Behavioral category'])
    df[i]['Behavioral category'] = np.where(df[i]['Behavior'].isin(['flee']), 'Aversive', df[i]['Behavioral category'])





    df[i]['time2'] = df[i]['Time'].shift(-1)
    #create new column of transition behavior
    df[i]['Behavior next'] = df[i]['Behavior'].shift(-1)
    df[i]['duration'] = df[i]['time2']-df[i]['Time']
    #removing first and last rows because we dont know their length
    df[i]= df[i][1:-1]
    #Drop irrelevant columns
    df[i]= df[i].filter(['Behavior', 'Behavioral category', 'Behavior next', 'duration', 'Subject'])
    #percentage of each behavior time (each size of a node for a behavior in a graph will correspond to its percentage)
    #count up transition behaviors, create probabilities
    transition_matrix.append(df[i].groupby(['Behavior', 'Behavior next', 'Behavioral category']).count())
    behavior_time.append(df[i].groupby(['Behavior','Behavioral category']).agg({'duration':sum}))
    transition_matrix[i].rename(columns={'duration':'Counts'}, inplace=True)
    transition_matrix[i].drop(['Subject'], axis=1, inplace=True)
    behaviors.append(df[i]['Behavior'].unique())
    categories.append(df[i]['Behavioral category'].unique())
    associations =df[i].groupby('Behavioral category')['Behavior'].unique().apply(list)
#merge all transition matrices into transition_df
transition_df = transition_matrix.pop(0)
transition_df.reset_index(level=[0,1,2], inplace=True)
transition_df.rename(columns={'Counts':'Counts_0'}, inplace=True)
behavior_time_df = behavior_time.pop(0)
behavior_time_df.rename(columns={'duration':'duration_0'}, inplace=True)
for i, frame in enumerate(transition_matrix):
    transition_matrix[i].reset_index(level=[0,1,2], inplace=True)
    transition_df = pd.merge(transition_df, transition_matrix[i], on=['Behavior', 'Behavior next', 'Behavioral category'], how='outer', suffixes=[None, '_{}'.format(i+1)])
    behavior_time_df = pd.merge(behavior_time_df, behavior_time[i], on=['Behavior','Behavioral category'], how='outer', suffixes=[None,'_{}'.format(i+1)])
transition_df.rename(columns={'Counts':'Counts_1'}, inplace=True)
behavior_time_df.rename(columns={'duration':'duration_1'}, inplace=True)
transition_df.fillna(0, inplace=True)
behavior_time_df.fillna(0, inplace=True)
all_behaviors = np.unique(np.concatenate(behaviors))
categories = np.unique(np.concatenate(categories))
transition_probabilities = transition_df.copy()
behavior_time_probabilities = behavior_time_df.copy()
transition_probabilities.columns= ['prob_{}'.format(col[-1]) if 'Counts' in col else col for col in transition_df.columns]
for column in [prob_column for prob_column in transition_probabilities.columns if 'prob' in prob_column]:
    transition_probabilities[column]= transition_probabilities[column]/transition_probabilities[column].sum()
behavior_time_probabilities.columns= ['prob_{}'.format(col[-1]) if 'duration' in col else col for col in behavior_time_df.columns]
for column in [prob_column for prob_column in behavior_time_probabilities.columns if 'prob' in prob_column]:
    behavior_time_probabilities[column]= behavior_time_probabilities[column]/behavior_time_probabilities[column].sum()

#find average and standard error for every row, only uses columns that have digits in them (counts_0, prob_0, etc)
dataframes = [behavior_time_df, behavior_time_probabilities, transition_df, transition_probabilities]
filenames = ['behavior_time.csv', 'behavior_time_probabilities.csv', 'transition_df.csv','transition_probabilites.csv']

for i, frame in enumerate(dataframes):
    temp = frame.copy()
    temp['avg']=0.0
    temp['sterr']=0.0
    temp.reset_index(inplace=True)
    obs_columns = np.array([column for column in frame.columns if re.search(r'\d', column)])
    for j, row in enumerate(frame.iterrows()):
        temp.at[j, 'avg'] = row[1][obs_columns].mean()
        temp.at[j, 'stderr'] = row[1][obs_columns].sem()
    dataframes[i] = temp
    if 'index' in dataframes[i].columns:
        dataframes[i].drop(['index'], axis=1, inplace=True)
    dataframes[i].set_index('Behavior')
    dataframes[i].to_csv('{}/{}/converted_data/{}'.format(folder, fish_type, filenames[i]))
