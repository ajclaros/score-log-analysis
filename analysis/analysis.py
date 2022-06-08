import os
import pandas as pd

import sys
from io import StringIO
import decimal
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Rectangle
import numpy as np
import decimal
from graphviz import *
import cv2


# Color mappings
colors = {
    'Aggressive':'red',
    'Aversive':'blue',
    'Reproductive':'forestgreen'}

#change bar colors
bar_colors = {'blue female 2': 'blue', 'yellow female 2':'yellow', 'blue female 1':'green', 'yellow female 1':'red'}

#Replace 'Aggressive', 'Aversive', 'Reproductive' with behavior classes
#replace lists on left side with list of associated behaviors for each class
behavior_classifications = dict()
behavior_classifications['Aggressive'] = ['b peck', 'y peck']
behavior_classifications['Aversive'] = ["y flee", "b flee"]
behavior_classifications['Reproductive'] = ['b circle', 'y circle', 'y follow', 'b follow', 'enter', 'exit']

# Replace duplicate or multiple behaviors to combine into a single term
#
combined_behaviors = dict()
#combined_behaviors['Enter/Exit'] = ['enter', 'exit', 'Pot Entry', 'Pot Exit']

#Behavior names to create rastar plots
behavior_raster = ['b peck', 'y peck','y flee','b flee','b circle','y circle','y follow', 'b follow', 'enter']


#Raster plot sizes
raster_width = 20
raster_height = 5
def create_df(fish_type):

    files = os.listdir('data/{}'.format(fish_type))
    if not os.path.exists('data/{}/converted_data'.format(fish_type)):
        os.makedirs("data/{}/converted_data".format(fish_type))
    csv = [x for x in files if 'tsv' in x or 'csv' in x]

    df = {data:pd.read_csv('data/{}/{}'.format(fish_type, data)) for data in csv}

    #df = [pd.read_csv('data/{}/{}'.format(fish_type, data), index_col=[0]) for data in csv]
    g = Digraph(name = fish_type, engine='dot')
    g.attr(label=fish_type, labelloc='t', )
    behavior_time = {}
    transition_matrix = {}
    transition_probabilities ={}
    behaviors = []
    categories = []
    single_probabilities = []
    for i, (name, frame) in enumerate(df.items()):
        global behavior_classifications
        for key in combined_behaviors.keys():
            df[name]['Behavior'] = np.where(df[name]['Behavior'].isin(combined_behaviors[key]), key, df[name]['Behavior'])

        for key in behavior_classifications.keys():
            df[name]['Behavioral category'] = np.where(df[name]['Behavior'].isin(behavior_classifications[key]), key, df[name]['Behavioral category'])


        df[name]['time2'] = df[name]['Time'].shift(-1)
        #create new column of transition behavior
        df[name]['Behavior next'] = df[name]['Behavior'].shift(-1)
        df[name]['duration'] = df[name]['time2']-df[name]['Time']
        #removing first and last rows because we dont know their length
        df[name]= df[name][1:-1]
        #Drop irrelevant columns
        df[name]= df[name].filter(['Behavior', 'Behavioral category', 'Behavior next', 'duration', 'Subject'])
        #percentage of each behavior time (each size of a node for a behavior in a graph will correspond to its percentage)
        #count up transition behaviors, create probabilities
        transition_matrix[name] = df[name].groupby(['Behavior', 'Behavior next', 'Behavioral category']).count()
        behavior_time[name] = df[name].groupby(['Behavior','Behavioral category']).agg({'duration':sum})
        transition_matrix[name].rename(columns={'duration':name}, inplace=True)
        transition_matrix[name].drop(['Subject'], axis=1, inplace=True)
        behaviors.append(df[name]['Behavior'].unique())
        categories.append(df[name]['Behavioral category'].unique())
        associations =df[name].groupby('Behavioral category')['Behavior'].unique().apply(list)

    #merge all transition matrices into transition_df
    score_names = list(transition_matrix.keys())
    print(score_names)
    transition_df = transition_matrix.pop(score_names[0])
    transition_df.reset_index(level=[0,1,2], inplace=True)
    transition_df.rename(columns={'Counts':score_names[0]}, inplace=True)
    behavior_time_df = behavior_time.pop(score_names[0])
    behavior_time_df.rename(columns={'duration':score_names[0]}, inplace=True)

    for i, name in enumerate(score_names[1:]):
        transition_matrix[name].reset_index(level=[0,1,2], inplace=True)
        transition_df = pd.merge(transition_df, transition_matrix[name], on=['Behavior', 'Behavior next', 'Behavioral category'], how='outer', suffixes=[None, name])
        behavior_time_df = pd.merge(behavior_time_df, behavior_time[name], on=['Behavior','Behavioral category'], how='outer', suffixes=[None,name])
    transition_df.fillna(0, inplace=True)
    behavior_time_df.fillna(0, inplace=True)
    all_behaviors = np.unique(np.concatenate(behaviors))
    categories = np.unique(np.concatenate(categories))
    transition_probabilities = transition_df.copy()
    behavior_time_probabilities = behavior_time_df.copy()
    transition_probabilities.columns= [score_names[-1] if 'Counts' in col else col for col in transition_df.columns]
    for column in [prob_column for prob_column in transition_probabilities.columns if '.tsv' in prob_column]:
        transition_probabilities[column]= transition_probabilities[column]/transition_probabilities[column].sum()
    behavior_time_probabilities.columns= [score_names[-1] if 'duration' in col else col for col in behavior_time_df.columns]
    for column in [prob_column for prob_column in behavior_time_probabilities.columns if 'prob' in prob_column]:
        behavior_time_probabilities[column]= behavior_time_probabilities[column]/behavior_time_probabilities[column].sum()

    #find average and standard error for every row, only uses columns that have digits in them (counts_0, prob_0, etc)
    dataframes = [behavior_time_df, behavior_time_probabilities, transition_df, transition_probabilities]
    filenames = ['behavior_time.csv', 'behavior_time_probabilities.csv', 'transition_df.csv','transition_probabilites.csv']

    for i, frame in enumerate(dataframes):
        temp = frame.copy()
        temp['avg']=0.0
        temp['stderr']=0.0
        temp.reset_index(inplace=True)
        obs_columns = np.array([column for column in frame.columns if re.search(r'\d', column)])
        for j, row in enumerate(frame.iterrows()):
            temp.at[j, 'avg'] = row[1][obs_columns].mean()
            temp.at[j, 'stderr'] = row[1][obs_columns].sem()
        dataframes[i] = temp
        if 'index' in dataframes[i].columns:
            dataframes[i].drop(['index'], axis=1, inplace=True)
        dataframes[i].set_index('Behavior')
        dataframes[i].to_csv('data/{}/converted_data/{}'.format(fish_type, filenames[i]))

def import_dataframes():
    df = dict()
    filenames = [filename for filename in os.listdir('data/') if 'tsv' in filename or 'csv' in filename]
    for tsv in filenames:
       with open('data/{}'.format(tsv), 'r') as f:
           s = f.readlines()
       match = ['Time', 'Media', 'file', 'path', 'Total', 'length', 'FPS', 'Subject', 'Behavior', 'Behavioral', 'category', 'Comment', 'Status']
       for i, line in enumerate(s):
           if line.split() == match:
               s = '\n'.join(s[i::])
               s = StringIO(s)
               df[tsv] = pd.read_csv(s, sep='\t')
    return df

def GetKey(val, dictionary):
    for key, value in dictionary.items():
        if val in value:
            return key
    return "key for {} doesn't exist".format(val)

def split_subject(df):
    filenames = list(df.keys())
    subjects = [df[filename]["Subject"].unique()[0] for filename in filenames]
    for subject in subjects:
        if not os.path.exists('data/{}'.format(subject)):
            os.mkdir('data/{}'.format(subject))
    for filename in filenames:
        frame = df[filename]
        if len(frame.groupby('Subject'))>1:
            for name, group in frame.groupby('Subject'):
                if name =='nil':
                    continue
                idx = filename.index('.tsv')
                new_filename = filename[:idx] + name + filename[idx:]
                group.to_csv('data/{}/{}'.format(name, new_filename), index=False)
        else:
            frame.to_csv('data/{}/{}'.format(frame['Subject'][0], filename), index=False)
    return

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

def create_graph(data, fish_type):
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
        graph.attr(fixed_size='false', overlap='scale', size=str(90), packMode='clust', compound='true', label='{}  {} >0.01\%'.format(fish_type, column), fontname='fira-code')
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
            else:
                edge_color = 'slategray'
            temp_var= transition_individual_probabilities.loc[row[0]][column]
    #        if temp_var <1e-04:
    #            size_label='0'
    #            continue
            size_label = "{}".format(round(row[column]/transition_individual_probabilities.loc[row[0]][column]*100, 2))

            if size_label=='0':
                continue
            else:
                graph.edge(row[0], row[1], color=edge_color, headxlabel= str('{}'.format(size_label)), labelloc='c', fontsize='15', penwidth=str(.1* float( size_label)))
        graph.render('data/{}/converted_data/markov/{}_{}'.format(fish_type,fish_type, column), format='png', quiet=True)
    return

def create_barplots(folders, custom_colors=None):
    behavior_time = [pd.read_csv('data/{}/converted_data/behavior_time_probabilities.csv'.format(group)) for group in folders]
    #behavior_time = [pd.read_csv('data/{}/converted_data/behavior_time_probabilities.csv'.format(group), index_col=[0]) for group in folders]
    for i, behavior in enumerate(behavior_time):
         for beh in behavior_time[i]['Behavior']:
             if beh not in behavior_time[i-1]['Behavior'].values:
                behavior_time[i-1]=behavior_time[i-1].append(pd.Series(),ignore_index=True )
                behavior_time[i-1]['Behavior']= behavior_time[i-1]['Behavior'].replace(np.nan, beh)
         for beh in behavior_time[i-1]['Behavior']:
             if beh not in behavior_time[i-1]['Behavior'].values:
                 behavior_time[i]= behavior_time[i].append(pd.Series(),ignore_index=True )
                 behavior_time[i]['Behavior']= behavior_time[i]['Behavior'].replace(np.nan, beh)

         behavior_time[i].fillna(0, inplace=True)
         behavior_time[i].set_index('Behavior', inplace=True)


         behavior_time[i]['color']= bar_colors[folders[i]]
         behavior_time[i]['group'] = folders[i]
         behavior_time[i]= behavior_time[i][['avg','color','stderr', 'group']]
         behavior_time[i].sort_index(inplace=True)
         behavior_time[i].reset_index(inplace=True)
         for col in behavior_time[i].columns:
             if col == 'group':
                 behavior_time[i][col] = behavior_time[i][col].fillna(behavior_time[i]['group'][0])
             else:
                 behavior_time[i][col] = behavior_time[i][col].fillna(0)
    joined= pd.merge(behavior_time[-1], behavior_time[0], how='outer', on=['Behavior','avg','stderr', 'group','color'])
    joined.fillna(value = {'avg':0.0, 'stderr':0.0, 'color':joined['color'].unique()[-2],'group':joined['group'].unique()[-2]}, inplace= True)
    if len(behavior_time)>2:
        for i, name in enumerate(behavior_time[:-1]):
            joined = pd.merge(joined, behavior_time[i+1], how='outer', on=['Behavior','avg','stderr','group','color'])
            joined.fillna(value = {'avg':0.0, 'stderr':0.0, 'color':joined['color'].unique()[-2],'group':joined['group'].unique()[-2]}, inplace= True)


    fig, ax =plt.subplots()
    pivoted = joined.pivot(index='Behavior', columns='group', values='avg')
    #convert to percent
    pivoted = pivoted.div(pivoted.sum(axis=0), axis=1)*100

    if custom_colors is not None:
        pivoted.plot(kind='bar', width=.9, color=custom_colors, ax=ax, yerr=joined.pivot(index='Behavior',columns='group',values='stderr').fillna(0).T.values, error_kw=dict(ecolor='slategray' ,capthick=2, capsize=5))
    else: pivoted.plot(kind='bar', width=.9, color=[bar_colors[group] for group in pivoted.columns], ax=ax, yerr=joined.pivot(index='Behavior',columns='group',values='stderr').fillna(0).T.values, error_kw=dict(ecolor='slategray' ,capthick=2, capsize=5))


    ax.tick_params(axis='x', colors='black', rotation=90)
    ax.tick_params(axis='y', colors='black')
    ax.xaxis.label.set_text('Behavior')
    ax.xaxis.label.set_color('black')


    ax.yaxis.label.set_text('Average percent')
    ax.yaxis.label.set_color('black')
    for i, tick in enumerate(ax.get_xticklabels()):
        color = colors[GetKey([tick.get_text()][0], behavior_classifications)]
        plt.gca().get_xticklabels()[i].set_color(color)

    plt.gca().set_title("")
    plt.tight_layout()
    if not os.path.exists('./images/'):
        os.mkdir("./images/")
    plt.savefig('./images/behavior_time_dyad_barplot.png', dpi=300)
    plt.clf()
    



def combine_rasters(df):
    if not os.path.lexists('./images/combined/'):
        os.makedirs('./images/combined')
    origin_files = [name.replace('.tsv', '') for name in df.keys()]

    image_folders = [string for string in os.listdir('./images') if '.' not in string]
    image_folders.remove('combined')
    for name in origin_files:
        images=[]
        for i, fold in enumerate(image_folders):
            temp = [image_file for image_file in os.listdir('./images/{}/'.format(fold)) if name in image_file]
            print('--------')
            print(i)
        
            print(fold)
            print(temp)
            image = cv2.imread('./images/{}/{}'.format(fold, temp[0]))

            images.append(image)

        vert_images = cv2.vconcat(images)
        cv2.imwrite('./images/combined/{}.png'.format(name), vert_images)

def create_raster(filename, animal_type, behaviors, duration_behaviors = None):
    data = pd.read_csv('./data/{}/{}'.format(animal_type, filename))
    blue =  (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
#    data = pd.read_csv('./data/{}/{}'.format(animal_type, filename), index_col=0)
    data['time2'] = data['Time'].shift(-1)
    fig, ax = plt.subplots(figsize=(raster_width,raster_height))
    cmap = get_cmap('tab10')
    colors = cmap.colors
    for i, beh in enumerate(behavior_raster):
        if duration_behaviors is not None:
            if beh in duration_behaviors:

                duration = data[data['Behavior']==beh]['time2']-data[data['Behavior']==beh]['Time']
                print(duration)
#                ax.eventplot(data[data['Behavior']==beh]['Time'], color='k', label=beh)
                for ix, (row_index, row) in enumerate(data[data['Behavior']==beh].iterrows()):
                    if ix==0:
                       ax.add_patch(Rectangle((row['Time'],0.9), width=row['time2']-row['Time'], height=0.2, label=beh, color=blue))
                    else:
                       ax.add_patch(Rectangle((row['Time'],0.9), width=row['time2']-row['Time'], height=0.2, color=blue))
                continue
        ax.eventplot(data[data['Behavior']==beh]['Time'], color= colors[i], label=beh)
    fig.suptitle('Filename: {}, Type: {}'.format(filename, animal_type))
    plotname = filename.split('.')[0]
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    if not os.path.lexists('./images/{}/'.format(animal_type)):
        os.makedirs('./images/{}'.format(animal_type))
    plt.savefig('./images/{}/{}_{}.png'.format(animal_type,plotname,animal_type))
    plt.clf()










df = import_dataframes()
split_subject(df)
folders = [folder for folder in os.listdir('data/') if 'tsv' not in folder and 'csv' not in folder]
for fish_type in folders:
    print("Creating folder: {}".format(fish_type))
    create_df(fish_type)
    print('Creating images')
    data = {}
    for csv in os.listdir('data/{}/converted_data/'.format(fish_type)):
        if '.csv' not in csv:
            continue
        data[csv[:-4]] = pd.read_csv('data/{}/converted_data/{}'.format(fish_type,csv),index_col=0)
    create_graph(data, fish_type)
create_barplots(folders[:2] , custom_colors = ['blue', 'yellow'])


animal_types = {dirname:os.listdir('./data/{}'.format(dirname)) for dirname in os.listdir('./data/') if os.path.isdir(os.path.join('./data', dirname))}
for animal_type in animal_types.keys():
    print(animal_type)
    for filename in os.listdir('./data/{}'.format(animal_type)):
        if not os.path.isdir('./data/{}/{}'.format(animal_type, filename)):
            beh_data = create_raster(filename, animal_type, behavior_raster, duration_behaviors=['enter'])
#combine_rasters(df)
print('done')
