#endregion
#region # IMPORTS AND SET UP
# =============================================================================
# IMPORTS AND SET UP
# =============================================================================

import os
import pandas as pd

# Change directory to the specified path
os.chdir('/home/dan/mini_temporal')
#endregion
#region # FUNCTIONS
# =============================================================================
# FUNCTIONS
# =============================================================================

def move_baseline(s):
    words = s.split()
    if 'baseline' in words:
        words.remove('baseline')
        words.insert(2, 'baseline')
    return ' '.join(words)

def find_evaluated_on(name):
    evaluated_on = name.split()[4]
    if evaluated_on == 'evaluated':
        evaluated_on = name.split()[3]
    return evaluated_on
    
#endregion
#region # LOAD AND CLEAN DATA
# =============================================================================
# LOAD AND CLEAN DATA
# =============================================================================
df = pd.read_json('2b_generations/_IT_c_first/AQA/gemma-1_1-2b-it/_IT_c_first_results.jsonl', lines=True, orient='records')

df['NAME'] = df['NAME'].apply(lambda x: x.split('/')[-1][:-6].replace('_date', '-date').replace('_context', '-context').replace('1_1', '1.1').replace('_', ' '))

# MOVE THE WORD BASELINE SO THAT IT FITS WITH THE SPLITTING LATER
df['NAME'] = df['NAME'].apply(lambda x: move_baseline(x))

# SET UP THE COLUMNS WE NEED
df['DATASET'] = df['NAME'].apply(lambda x: x.split()[0])
df['BASE_MODEL'] = df['NAME'].apply(lambda x: x.split()[1])
df['TRAINED_ON'] = df['NAME'].apply(lambda x: x.split()[2])
df['EVALUATED_ON'] = df['NAME'].apply(lambda x: find_evaluated_on(x))
df['MODEL'] =  df['NAME'].apply(lambda x: f"{x.split()[1]} + {x.split()[2]}")

models = list(df['MODEL'].unique())

final_df = pd.DataFrame({'MODEL': models})

# Define columns for the final DataFrame
columns = [
    'no_context_ACC', 'no_context_F1', 'no_context_BEM',
    'rel_context_ACC', 'rel_context_F1', 'rel_context_BEM',
    'wrong-date_context_ACC', 'wrong-date_context_F1','wrong-date_context_BEM',
    'rand_context_ACC', 'rand_context_F1', 'rand_context_BEM',
    
]

# Initialize columns with NaN
for col in columns:
    final_df[col] = float('nan')

# Fill the final DataFrame
for index, row in df.iterrows():
    model = row['MODEL']
    evaluated_on = row['EVALUATED_ON']
    acc = row['ACC']
    f1 = row['F1']
    bem = row['BEM']

    if evaluated_on == 'no-context':
        final_df.loc[final_df['MODEL'] == model, 'no_context_ACC'] = acc
        final_df.loc[final_df['MODEL'] == model, 'no_context_F1'] = f1
        final_df.loc[final_df['MODEL'] == model, 'no_context_BEM'] = bem
    elif evaluated_on == 'relevant-context':
        final_df.loc[final_df['MODEL'] == model, 'rel_context_ACC'] = acc
        final_df.loc[final_df['MODEL'] == model, 'rel_context_F1'] = f1
        final_df.loc[final_df['MODEL'] == model, 'rel_context_BEM'] = bem
    elif evaluated_on == 'random-context':
        final_df.loc[final_df['MODEL'] == model, 'rand_context_ACC'] = acc
        final_df.loc[final_df['MODEL'] == model, 'rand_context_F1'] = f1
        final_df.loc[final_df['MODEL'] == model, 'rand_context_BEM'] = bem
    elif evaluated_on == 'wrong-date-context':
        final_df.loc[final_df['MODEL'] == model, 'wrong-date_context_ACC'] = acc
        final_df.loc[final_df['MODEL'] == model, 'wrong-date_context_F1'] = f1
        final_df.loc[final_df['MODEL'] == model, 'wrong-date_context_BEM'] = bem
    

# Display the final DataFrame
print(final_df)

final_df.to_csv('_IT_c_first_results.tsv', sep = '\t')