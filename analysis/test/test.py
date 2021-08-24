#!/usr/bin/env python3
#to be run in the same directory as analysis.py
#finds all unique behaviors to be classified
behs = []
for d in df.values():
    behs.append(d['Behavior'].unique().tolist())
behs = flatten(behs)
def unique(list1):

    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list
print(unique(behs))
