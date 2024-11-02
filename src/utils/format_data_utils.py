import random
import pandas as pd
import re
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

def generate_random_context_Menat(df: pd.DataFrame, seed: int = 42):
    """
    Assign random contexts to each row in the DataFrame. Each random context 
    will be selected from rows with different questions.
    """
    random.seed(seed)  # Set seed for reproducibility
    # Create a new column in the original DataFrame
    df['random_context'] = '' 
    
    for idx, row in df.iterrows():
        # Exclude rows that have the same question
        available_contexts = df[df['id'] != row['id']]['relevant_context'].tolist()
        
        if not available_contexts:
            # In case there are no available contexts, skip this row
            print(f"No available context for question: {row['question']}")
            continue

        # Assign random context from available options

        df.at[idx, 'random_context'] = random.choice(available_contexts)

def generate_context_unshuffled(df: pd.DataFrame):
    """Generate unshuffled context from the relevant context."""
    df['context_unshuffled'] = df['relevant_context'].apply(lambda x: tuple(sorted(x.split('\n'))) if pd.notnull(x) else ())

def generate_random_context_TimeQA(df: pd.DataFrame):
    """Generate random context by shuffling unique contexts."""
    generate_context_unshuffled(df)
    unique_unshuffled_contexts = df['context_unshuffled'].unique()
    random_contexts = []

    for _, row in df.iterrows():
        current_unshuffled_context = row['context_unshuffled']
        choice_list = [c for c in unique_unshuffled_contexts if c != current_unshuffled_context]
        
        if not choice_list:
            random_contexts.append('')  # If no unique context is available
        else:
            chosen_context = random.choice(choice_list)
            random_contexts.append('\n'.join(random.sample(list(chosen_context), len(chosen_context))))

    df['random_context'] = random_contexts
    df.drop(columns=['context_unshuffled'], inplace=True)
    return df
def temp_reason_get_answer(row:pd.Series) -> str:
    """Extract answers from the text_answers field."""
    return row['text_answers']['text']

def temp_reason_basic_processing(df:pd.DataFrame) -> pd.DataFrame:   
    # get the answer from the text_answers field
    df['answers'] = df.apply(temp_reason_get_answer, axis=1)
    # rename the fact_context field to relevant_context
    if 'fact_context' in df.columns:
        df['relevant_context'] = df['fact_context']
        df.drop(columns=['fact_context'], inplace=True)
    # drop the none_context, neg_answers, and context fields
    if 'none_context' in df.columns:
        df.drop(columns=['none_context'], inplace=True)
    if 'neg_answers' in df.columns:
        df.drop(columns=['neg_answers'], inplace=True)
    if 'context' in df.columns:
        df.drop(columns=['context'], inplace=True)
    return df

def assign_no_context(df: pd.DataFrame):
    """Assign no context to each row in the DataFrame."""
    df['no_context'] = ''

def assign_mixed_context(df: pd.DataFrame, random_seed: int = 42):
    """
    Assigns a mixed context based on the given random seed.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing context options.
    random_seed (int): Seed for the random generator. Default is 42.
    """
    random.seed(random_seed)
    options = ['no_context', 'relevant_context', 'random_context', 'wrong_date_context']

    # Create 'mixed_context' column by directly modifying the DataFrame using .loc[]
    df['mixed_context'] = df.apply(lambda row: row[random.choice(options)], axis=1)

    # Drop the no_context column after use
    # df.drop(columns=['no_context'], inplace=True)

# Predefined list of full months and their abbreviations
MONTHS_FULL = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

MONTHS_ABBREV = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Precompile regex patterns
YEAR_PATTERN = re.compile(r'\b\d{4}\b')
MONTH_PATTERN = re.compile(r'\b(?:' + '|'.join(MONTHS_FULL + MONTHS_ABBREV) + r')\b')
DECADE_PATTERN = re.compile(r'\b\d{4}s\b')

def generate_false_year(actual_year: int):
    """Generate a false year excluding the actual year."""
    years = np.setdiff1d(np.arange(1850, 2024), np.array([actual_year]))
    return np.random.choice(years)

def generate_false_month(actual_month: str) -> str:
    """Generate a false month, supporting full and abbreviated months."""
    if actual_month in MONTHS_FULL:
        filtered_months = [m for m in MONTHS_FULL if m != actual_month]
    else:
        filtered_months = [m for m in MONTHS_ABBREV if m != actual_month]
    
    return np.random.choice(filtered_months)

def generate_false_decade(actual_decade: int) -> str:
    """Generate a false decade excluding the actual decade."""
    decade_start = (actual_decade // 10) * 10
    decades = np.setdiff1d(np.arange(1850, 2030, 10), np.array([decade_start]))
    return str(np.random.choice(decades)) + 's'

def falsify_dates(text: str) -> str:
    """Extract and replace dates with falsified ones."""
    # Replace decades
    falsified_text = DECADE_PATTERN.sub(lambda x: generate_false_decade(int(x.group(0)[:4])), text)
    
    # Replace years
    falsified_text = YEAR_PATTERN.sub(lambda x: str(generate_false_year(int(x.group(0)))), falsified_text)

    # Replace months (full and abbreviated)
    falsified_text = MONTH_PATTERN.sub(lambda x: generate_false_month(x.group(0)), falsified_text)
    
    return falsified_text

def generate_wd_context(df: pd.DataFrame, num_workers: int):
    """Run the processing in parallel using multiple workers and update the DataFrame."""
    with Pool(processes=num_workers) as pool:
        # Process the specified column in the DataFrame
        results = list(tqdm(pool.imap(falsify_dates, df['relevant_context'], chunksize=100), total=len(df)))
    
    # Update the DataFrame with the falsified context
    df['wrong_date_context'] = results