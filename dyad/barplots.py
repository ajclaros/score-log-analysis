import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os




#change for group names
group_0= 'blue'
group_1 = 'yellow'
group_0_bt= pd.read_csv('{}/converted_data/behavior_time_probabilities.csv'.format(group_0),index_col=0)
group_1_bt = pd.read_csv('{}/converted_data/behavior_time_probabilities.csv'.format(group_1), index_col=0)

#new_index = ['Chase F','Chase M','Dig','Ent Pot', 'Ext Pot','Lead','QVR', 'Side disp']
#new_index_y= ['Chase F','Flee F','Flee M','QVR','Dig', 'Side disp']
#group_0_bt.rename(index={group_0_bt.index[i]: new_index[i] for i, null in enumerate(group_0_bt.index.values)}, inplace=True)
#group_1_bt.rename(index={group_1_bt.index[i]: new_index_y[i] for i, null in enumerate(group_1_bt.index.values)}, inplace=True)
#group_1_bt.reset_index(inplace=True)
#group_0_bt.reset_index(inplace=True)
for beh in group_0_bt['Behavior']:
    if beh not in group_1_bt['Behavior'].values:

        group_1_bt= group_1_bt.append(pd.Series(),ignore_index=True )
        group_1_bt['Behavior']= group_1_bt['Behavior'].replace(np.nan, beh)
for beh in group_1_bt['Behavior']:
    if beh not in group_0_bt['Behavior'].values:
        group_0_bt= group_0_bt.append(pd.Series(),ignore_index=True )
        group_0_bt['Behavior']= group_0_bt['Behavior'].replace(np.nan, beh)

group_0_bt.fillna(0, inplace=True)
group_1_bt.fillna(0, inplace=True)
group_0_bt.set_index('Behavior', inplace=True)


#change for specified colors
colors = ['blue', 'gold']
group_0_bt['color']= colors[0]

group_1_bt['color']=colors[1]

group_1_bt.set_index('Behavior', inplace=True)

group_0= group_0_bt[['avg','color','stderr']]
group_1= group_1_bt[['avg','color','stderr']]
group_1.sort_index(inplace=True)
group_1.sort_index(inplace=True)
group_1.reset_index(inplace=True)
group_1.reset_index(inplace=True)
joined= pd.merge(group_0,group_1, how='outer', on=['Behavior', 'avg','stderr','color'])
fig, ax =plt.subplots()


joined.pivot(index='Behavior',columns='color', values='avg').plot(kind='bar', width=.9, color=colors, ax=ax, yerr=joined.pivot(index='Behavior',columns='color',values='stderr').fillna(0).T.values, error_kw=dict(ecolor='slategray' ,capthick=2, capsize=5))
ax.tick_params(axis='x', colors='black', rotation=90)
ax.tick_params(axis='y', colors='black')
ax.xaxis.label.set_text('Behavior')
ax.xaxis.label.set_color('black')


ax.yaxis.label.set_text('Average percent')
ax.yaxis.label.set_color('black')
beh_mapping = {}
for group, df in group_0_bt.groupby('Behavior'):
    key=group
    val = df['Behavioral category'].values[0]
    if key not in beh_mapping.keys():
        beh_mapping[key]=val

for group, df in group_1_bt.groupby('Behavior'):
    key=group
    val = df['Behavioral category'].values[0]

    if key not in beh_mapping.keys():
        beh_mapping[key]=val
for key, val in beh_mapping.items():
    if val=='Aggressive':
        beh_mapping[key]='red'
    if val=='Reproductive':
        beh_mapping[key]='green'

    if val=='Aversive':
        beh_mapping[key]='blue'
for i, tick in enumerate(ax.get_xticklabels()):
    color = beh_mapping[tick.get_text()]
    if color==0:
        color='blue'
    plt.gca().get_xticklabels()[i].set_color(color)

plt.gca().set_title("")
plt.tight_layout()

plt.savefig('behavior_time_dyad_barplot.png', dpi=300)
