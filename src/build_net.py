import pandas as pd
import pickle
from utils import *



# Import Datasets -------------------------------------------------------------

## all abstractss
abstract_df = pd.read_json(abstract_file)
print("Number of Abstracts:" ,len(abstract_df))
## all possible reviewers
reviewers_df = pd.read_json(reviewers_file)
print("Number of Available Reviewers:",  len(reviewers_df))

reviewers_df['topics'] = reviewers_df['topics'].apply(
    lambda x: [item.upper() if isinstance(item, str) else item for item in x] 
    if isinstance(x, list) 
    else x
)

# Data Cleaning ---------------------------------------------------------------

# extract dictionary that maps numbers to affiliations
affiliations_dict = abstract_df['affiliations'].apply(lambda x: {affil['nr']: affil['affiliation'] for affil in x})
abstract_df['authors_aff'] = [parse_affiliations(authors, affil_map) for authors, affil_map in zip(abstract_df['authors'], affiliations_dict)]

abstract_df = expand_authors(abstract_df)


# select bits sessions and uppercase
abstract_df['session'] = abstract_df['session'].str.upper()
bits_topics = set(abstract_df['session'][abstract_df['session'].notnull()])



# Scopus Search ---------------------------------------------------------------

# trova ID scopus per tutti gli autori degli abstract e per i reviewers

if all(os.path.exists(f) for f in [cached_authors, cached_reviewers]):
    authors_id, reviewers_id = pd.read_json('../cache/cached_authors.json', orient='index'), pd.read_json('../cache/cached_reviewers.json', orient='index')
    logging.info("Autori e Revisori già esistenti, carico.")
elif os.path.exists(cached_authors):
    authors_id = pd.read_json('../cache/cached_authors.json', orient='index')
    logging.info("Autori già esistenti, carico.")
    reviewers_id = process_reviewers(reviewers_df, bits_topics)
    os.makedirs('../cache', exist_ok=True) 
    reviewers_id.to_json(cached_reviewers, orient='index', force_ascii=False)
    logging.info("Revisori salvati.")
elif os.path.exists(cached_reviewers):
    reviewers_id = pd.read_json('../cache/cached_reviewers.json', orient='index')
    logging.info("Revisori già esistenti, carico.")
    authors_id = process_authors(abstract_df)
    os.makedirs('../cache', exist_ok=True) 
    authors_id.to_json(cached_authors, orient='index', force_ascii=False)
    logging.info("Autori salvati.")
else:   
    authors_id, reviewers_id = process_all(abstract_df, reviewers_df, bits_topics)


    
    os.makedirs('../cache', exist_ok=True) 
    authors_id.to_json(cached_authors, orient='index', force_ascii=False)
    reviewers_id.to_json(cached_reviewers, orient='index', force_ascii=False)
    logging.info("Autori e Revisori salvati.")


# Network Construction --------------------------------------------------------
logging.info("Creating the Network...")
G = create_reviewer_author_network(authors_id, reviewers_id)
with open(cached_graph, 'wb') as f:
    pickle.dump(G, f)
logging.info("Network built and cached.")