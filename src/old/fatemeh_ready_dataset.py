"""
Created on 05/04/2024

@author: Dan Schumacher

This code will accomplish formatting the dataset the way that Fatemeh wants it.

- I will limit it to just the mini_TQ dataset
- there will not be any preprocessing on the data
- I will switch the 2 instances of messed up random context

- cols will beas follows
- 'i' : # 0 to len(data)
- 'idx' : id but matched to dataset (could be useful when/if we test/train on multiple datasets concated)
- 'question': the question 
- 'answer': the answer
- 'relevant_context: relevant context, in this case made by GPT-3.5', 
- 'source': what datset did data come from (could be useful when/if we test/train on multiple datasets concated)
- 'type': train dev or test (I like to do this in case i need to make global changes, I can stack and unstack easily),
- 'random_context': shifted from other rows' 
- 'wrong_date_context': same as relevant context but remapped with fabricated dates'

"""

#endregion
#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================

import pandas as pd
import os 

os.chdir('/home/dan/DeepLearning/mini_temporal/clean_for_fatemeh/')

os.path.exists('./old_data/train.jsonl')


#endregion
#region # Read in data
# =============================================================================
# Read in data
# =============================================================================
# read in 
train = pd.read_json('./old_data/train.jsonl', lines=True)
# limit to just mini data set
train = train[train['source'] == 'TQ_explicet_explicet']


# repeat for dev and test
dev = pd.read_json('./old_data/dev.jsonl', lines=True)
dev = dev[dev['source'] == 'TQ_explicet_explicet']

test = pd.read_json('./old_data/test.jsonl', lines=True)
test = test[test['source'] == 'TQ_explicet_explicet']


#endregion
#region # CLEANING
# =============================================================================
# CLEANING
# =============================================================================
# SHOW ISSUE TO FATEMEH
def test_for_bad_random_context(df):
    flag = False
    for e, rc in enumerate(df['random_context']):
        if len(rc.split()) > 300:
            flag = True
            print('INDEX:',e, 'LEN:', len(rc.split())) 
    if flag == False:
        print('NO LONG CONTEXT FOUND')
        

# these context are so much longer bc stolen from other datset
test_for_bad_random_context(test)

# only take context from the same dataset.
def correct_shift(df):
    # Shift 'relevant_context' down by one
    df['random_context'] = df['relevant_context'].shift(-1)
    # Wrap the last 'relevant_context' to the first 'random_context'
    df.loc[0, 'random_context'] = df['relevant_context'].iloc[-1]
    # Set the last 'random_context' to the first 'relevant_context'
    df.loc[len(df) - 1, 'random_context'] = df['relevant_context'].iloc[0]

    return df

# apply correct shifting
correct_shift(train)
correct_shift(dev)
correct_shift(test)

# recheck, 
test_for_bad_random_context(train)
test_for_bad_random_context(dev)
test_for_bad_random_context(test)
#endregion
#endregion
#region # SAVING
# =============================================================================
# SAVING
# =============================================================================
train.to_json('./new_data/train.jsonl', lines=True,orient='records')
dev.to_json('./new_data/dev.jsonl', lines=True,orient='records')
test.to_json('./new_data/test.jsonl', lines=True,orient='records')

# mini_temporal/clean_for_fatemeh/new_data
#endregion
