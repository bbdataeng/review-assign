import pandas as pd
from utils import *
import pickle


# trova ID scopus per tutti gli autori degli abstract e per i reviewers
if os.path.exists(cached_authors) and os.path.exists(cached_reviewers):
    authors_id = pd.read_json(cached_authors, orient='index')
    authors_id['scopus_id'] = authors_id['scopus_id'].fillna(0).astype(int)
    
    print("Number of Posters: ", len(authors_id[authors_id['type'] == 'Poster'].groupby('abstract_id').size()))
    print("Numer of Oral Presentations: ", len(authors_id[authors_id['type'] == 'Oral communication'].groupby('abstract_id').size()))
    
    reviewers_id = pd.read_json(cached_reviewers, orient='index')
    
    exploded_topics = reviewers_id.explode('topics')

    # numero revisori per topic
    reviewers_per_topic = exploded_topics['topics'].value_counts().reset_index()
    reviewers_per_topic.columns = ['topic', 'reviewer_count']
    
    # numero abstract per topic
    abstracts_per_session = authors_id.drop_duplicates(subset=['abstract_id', 'session']) \
                         .groupby('session') \
                         .size() \
                         .reset_index(name='abstract_count')
    
    print("Load cached authors and reviewers.")
else:
    print("No data found. You need to run build_net first.")



if os.path.exists(cached_graph):
    with open(cached_graph, 'rb') as f:
        G = pickle.load(f)
        mapping = {node: str(node) for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping) #assicurarsi che siano stringhe
        [nx.set_node_attributes(G, {n: int(d['bits_id']) for n, d in G.nodes(data=True) if 'bits_id' in d and str(d['bits_id']).strip().isdigit()}, 'bits_id')]
    print(G.nodes)
    print("Load cached graph.")
    for node in G.nodes():
        if 'topics' in G.nodes[node]:
        # Convert each topic to uppercase if it's a string
            G.nodes[node]['topics'] = [topic.upper() for topic in G.nodes[node]['topics']]
else:
    print("No data found. You need to run build_net first.")



# Reviewer Assignment ---------------------------------------------------------
logging.info("Assegnazione dei revisori in corso...")
logging.info(f"Number of abstracts to be assigned: {len(authors_id['abstract_id'].unique())}")
logging.info(f"Number of available reviewers: {len(reviewers_id)}")

save_final_assignments(G, authors_id, max_ass = max_assign, jolly_revs=jolly_revs, max_ass_jolly = max_assign_jolly)

