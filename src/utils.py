from pybliometrics.scopus import AuthorSearch, ScopusSearch, AuthorRetrieval
import pybliometrics
import pandas as pd
import os
import time
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import yaml
import logging
from datetime import datetime
pybliometrics.scopus.init()

# per topic match
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from collections import defaultdict

# logger = logging.getLogger()
# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

os.makedirs('../logs', exist_ok=True)

log_filename = datetime.now().strftime("../logs/assignment_%Y%m%d_%H%M%S.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'), 
                                            
    ]
)

logger = logging.getLogger()
logging.info(f"Inizio processo.")

with open('../config/config.yaml', 'r') as file:
    
    config = yaml.safe_load(file)


abstract_file = config["input_paths"]["abstracts"]
reviewers_file = config["input_paths"]["reviewers"]

cached_authors = config["cache_paths"]["authors"]
cached_reviewers = config["cache_paths"]["reviewers"]
cached_graph = config["cache_paths"]["coauth_graph"]


max_assign = config["parameters"]["max_assign"]
max_assign_jolly = config["parameters"]["max_assign_jolly"]
jolly_revs = config["parameters"]["jolly_revs"]
req_reviewers_map = config["parameters"].get("required_reviewers", {})

abstract_reviewers = config["output_paths"]["abstract_reviewers"]
reviewers_assignment = config["output_paths"]["reviewers_assignment"]


def single_map(element_list, mapping):
    mapped = [mapping.get(word.upper(), "OTHER APPLICATIONS") for word in element_list]
    return list(set(mapped)) 


def parse_affiliations(authors, affil_map):
    authors_aff = []
    for author in authors:
        author_copy = author.copy()
        nr_list = author['nr'] if isinstance(author['nr'], list) else [author['nr']]
        author_copy['affiliations'] = [affil_map[nr] for nr in nr_list]
        authors_aff.append(author_copy)
    return authors_aff

def expand_authors(abstract_df):

    exploded_df = abstract_df.explode('authors_aff')

    # extract name, surname and affiliations
    exploded_df['author_name'] = exploded_df['authors_aff'].apply(lambda x: x['name'])
    exploded_df['author_surname'] = exploded_df['authors_aff'].apply(lambda x: x['surname'])
    exploded_df['author_affiliations'] = exploded_df['authors_aff'].apply(lambda x: ', '.join(x['affiliations']))

    # select only relevant columns
    exploded_df = exploded_df[['ID', 'session', 'title', 'type', 'author_name', 'author_surname', 'author_affiliations']]
    exploded_df.columns = ['ID', 'session', 'title', 'type', 'name', 'surname', 'affiliation']
    return exploded_df


def search_scopus_id(author_info, max_retries=3):
    """Search Scopus ID from name, surname and affiliation"""
    
    surname = author_info['surname']
    given_name = author_info['name']
    
    
    # try with name, surname and affiliation
    if 'affiliation' in author_info:
        affiliation = author_info['affiliation']
        query = f"AUTHLAST({surname}) and AUTHFIRST({given_name}) and AFFIL({affiliation})"
        print("QUERY:", query)
        
        for attempt in range(max_retries):
            try:
                search = AuthorSearch(query, verbose=False)
                if search.authors:
                    print("Scopus ID found.")
                    return search.authors[0].eid  # first match
                else:
                    break  # No results with affiliation
            except Exception as e:
                if attempt == max_retries - 1:
                    break #
                time.sleep(2)
    
    # try without affiliation
    query = f"AUTHLAST({surname}) and AUTHFIRST({given_name})"
    print("QUERY:", query)
    
    for attempt in range(max_retries):
        try:
            search = AuthorSearch(query, verbose=False)
            if search.authors:
                print("Scopus ID found without affiliation.")
                return search.authors[0].eid  # first match
            else:
                print("No Scopus ID found with or without affiliation.")
                return None
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    
    return None


def process_authors(df):
    
    """Process authors dataset (abstracts dataset) and returns another
    dataframe with associated Scopus ID"""
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            if search_scopus_id(row):
                scopus_id = search_scopus_id(row).split('-')[-1]
            
            result = {
                'abstract_id': row['ID'],
                'abstract_title': row['title'],
                'session': row['session'],
                'type':row['type'],
                'author_name': row['name'],
                'author_surname': row['surname'],
                'affiliation': row['affiliation'],
                'scopus_id': scopus_id
            }
            results.append(result)
            
            time.sleep(0.5)
                
        except Exception as e:
            print(f"Error for the abstract {row['ID']}: {str(e)}")
    
    return pd.DataFrame(results)



def get_scopus_topics(scopus_id):
    """Retrieve author's subject areas from Scopus"""
    try:
        author = AuthorRetrieval(scopus_id)
        time.sleep(0.5)  
        if author.subject_areas:
            return [area.area for area in author.subject_areas]
        return []
    except Exception as e:
        print(f"Error retrieving topics for {scopus_id}: {str(e)}")
        return []
    
class TopicMapper:
    def __init__(self, sessions):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.sessions = [t.upper() for t in sessions]
        self.topic_embeddings = self.model.encode(self.sessions)
        
    def map_to_conference_topic(self, scopus_topics, threshold=0.6):
        """Mappa i topic Scopus alle sessioni della conferenza in base alla similarità"""
        if not scopus_topics:
            return ["OTHER APPLICATIONS"]
            
        scopus_embeddings = self.model.encode([t.upper() for t in scopus_topics])
        similarity_matrix = cosine_similarity(scopus_embeddings, self.topic_embeddings)
        
        mapped_topics = set()
        for i, scopus_topic in enumerate(scopus_topics):
            max_sim_idx = np.argmax(similarity_matrix[i])
            max_sim = similarity_matrix[i][max_sim_idx]
            
            if max_sim > threshold:
                mapped_topics.add(self.sessions[max_sim_idx])
        
        return list(mapped_topics) if mapped_topics else ["OTHER APPLICATIONS"]
    
def process_reviewers(df_reviewers, sessions):
    topic_mapper = TopicMapper(sessions)

    results = []
    for _, row in tqdm(df_reviewers.iterrows(), total=len(df_reviewers)):
        try:
            reviewer_info = {
                'name': row['name'],
                'surname': row['surname'],
                'topics': row.get('topics', [])
            }
            
            scopus_id = search_scopus_id(reviewer_info)
            scopus_id = scopus_id.split('-')[-1]  
            
            # se la lista dei topics è vuota o se non contiene topics validi (in sessioni) -> cerca su Scopus
            if not reviewer_info['topics'] or not any(topic in sessions for topic in reviewer_info['topics']):
                print(f"Topics vuoti o non validi per {row['name']} {row['surname']}, cerco su Scopus...")
                
                print("ID trovato: ", scopus_id)
                scopus_topics = get_scopus_topics(scopus_id)  # estrae i topic da Scopus
                print(f"I topic da Scopus per {row['name']} sono: {scopus_topics}")
                reviewer_info['topics'] = topic_mapper.map_to_conference_topic(scopus_topics)  # mapping alle sessioni
                print(f"Mappati in: {reviewer_info['topics']}")
            else:
                print(f"Reviewer {row['name']} {row['surname']} ha già i seguenti topics: {reviewer_info['topics']}")

            # dizionario dei risultati
            result = {
                'reviewer_id': row['ID'],
                'reviewer_name': reviewer_info['name'],
                'reviewer_surname': reviewer_info['surname'],
                'topics': reviewer_info['topics'],
                'scopus_id': scopus_id if 'scopus_id' in locals() else None  # Aggiungi scopus_id se trovato
            }
            results.append(result)
        
        except Exception as e:
            print(f"Error processing reviewer {row['name']}: {str(e)}")

    return pd.DataFrame(results)



# per processare entrambi i dataset
def process_all(authors_df, reviewers_df, sessions):
    
    """Process both authors and reviewers all at once"""
    
    logging.info("Searching for authors Scopus ID...")

    authors_results = process_authors(authors_df)
    
    logging.info("Searching for reviewers Scopus ID...")
    reviewers_results = process_reviewers(reviewers_df, sessions)
    
    return authors_results, reviewers_results


def create_reviewer_author_network(abstracts_df, reviewers_df, min_sleep=3, max_retries=3):
    
    """
    Creates a co-authorship network between authors and reviewers
    """
    
    # extract id list
    author_ids = abstracts_df['scopus_id'].dropna().unique().tolist()
    author_ids = list(map(str, author_ids))
    reviewer_ids = reviewers_df['scopus_id'].dropna().unique().tolist()
    reviewer_ids = list(map(str, reviewer_ids))
    all_ids = list(set(author_ids + reviewer_ids))
    all_ids = list(map(str, all_ids))

    G = nx.Graph()    
    
    # add attributes
    for _, row in abstracts_df.iterrows():
        if pd.notna(row['scopus_id']):
            G.add_node(str(row['scopus_id']), 
                      type='author',
                      name=row['author_name'],
                      given_name=row['author_name'] + " " + row['author_surname'], 
                      surname=row['author_surname'],
                      affiliation=row['affiliation'],
                      abstract_id=row['abstract_id'],
                      abstract_title=row['abstract_title'],
                      reviewer=False)
    
    for _, row in reviewers_df.iterrows():
        if pd.notna(row['scopus_id']):
            G.add_node(str(row['scopus_id']), 
                      type='reviewer',
                      name=row['reviewer_name'],
                      given_name=row['reviewer_name'] +  " " + row['reviewer_surname'],
                      surname=row['reviewer_surname'],
                      bits_id = row['reviewer_id'],
                    #   affiliation=row['reviewer_affiliation'],
                      topics=row['topics'],
                      reviewer=True,
                      review_count=0)
    
    # list of collaboration
    collaborations = defaultdict(list)
    
    # considerare solo i reviewers???
    for person_id in tqdm(reviewer_ids, desc="Processing collaborations"):
        # scopus search to find documents
        documents = []
        for attempt in range(max_retries):
            try:
                search = ScopusSearch(f"AU-ID({person_id})")
 
                time.sleep(min_sleep)
                documents = search.results
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error searching documents for ID {person_id}: {str(e)}")
                    continue
                time.sleep(min_sleep * 2)
        
        if not documents:
            continue
            
        # for each document finds the co-authors
        for doc in documents:
            if not doc.author_ids:
                continue
                
            coauthors = doc.author_ids.split(';')
            coauthors = [i.split('-')[-1] for i in coauthors]
            for coauthor in coauthors:
                if coauthor == person_id:
                    continue  # salva se stesso
                
                # aggiungi edge se l'altro autore è nella nostra lista
                if coauthor in all_ids:
                    # crea edge ordinato per evitare duplicati
                    edge = tuple(sorted((person_id, coauthor)))
                    
                    # updates collaboration documents list
                    collaborations[edge].append(doc.eid)
 
    # adds edges with weights (number of documents)
    for edge, doc_ids in collaborations.items():
        G.add_edge(edge[0], edge[1], weight=len(doc_ids))
    
    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G




def reviewer_choice(G, abstract_num, abstract_df, req_reviewers_map, max_ass=6, jolly_revs=None, max_ass_jolly=12):
    
    # Estrazione dati abstract
    abstract_data = abstract_df.loc[abstract_df['abstract_id'] == abstract_num].iloc[0]
    topic = abstract_data['session']
    abstract_type = abstract_data['type']
    abstract_title = abstract_data.get('abstract_title', 'Titolo non disponibile')
    author_ids = list(map(str, abstract_df.loc[abstract_df['abstract_id'] == abstract_num, 'scopus_id'].dropna().tolist()))
    
    required_reviewers = req_reviewers_map.get(abstract_type, 3)
    
    logging.info(f"\nAssegnazione abstract {abstract_num} - Tipo: {abstract_type}")
    logging.info(f"Topic: {topic} | Autori: {author_ids} | Revisori richiesti: {required_reviewers}")

    # Setup pool (Jolly e Standard)
    jolly_reviewers = set(jolly_revs if jolly_revs else [])
    standard_nodes = [n for n in G.nodes() if n not in jolly_reviewers]
    all_potential_reviewers = list(jolly_reviewers) + standard_nodes

    def count_collaborations(reviewer, authors):
        """Conta il numero totale di documenti co-autorati tra il revisore e gli autori"""
        total_collabs = 0
        for author in authors:
            if G.has_edge(reviewer, author):
                total_collabs += G[reviewer][author].get('weight', 1)
        return total_collabs
    
    def is_eligible(reviewer, max_collabs=0):
        # 1. Verifica che sia effettivamente un revisore nel grafo
        if reviewer not in G or not G.nodes[reviewer].get('reviewer', False):
            return False
        
        # 2. Esclude se il revisore è uno degli autori dell'abstract (self-review)
        if reviewer in author_ids:
            return False
            
        # 3. VERIFICA COLLABORAZIONI: non superare il limite consentito (0 o 1)
        if count_collaborations(reviewer, author_ids) > max_collabs:
            return False
            
        # 4. VERIFICA TOPIC 
        if topic not in G.nodes[reviewer].get('topics', []):
            return False
            
        # 5. VERIFICA LIMITI CARICO DI LAVORO
        limit = max_ass_jolly if reviewer in jolly_reviewers else max_ass
        current_count = G.nodes[reviewer].get('review_count', 0)
        
        if current_count >= limit:
            return False
            
        return True

    def get_candidates(pool, max_collabs=0):
        return [rev for rev in pool if is_eligible(rev, max_collabs)]
        
    def rank_reviewers(candidates):
        """Ordina i revisori per equità (chi ha meno revisioni), favorendo i Jolly in caso di pareggio"""
        def score(rev):
            limit = max_ass_jolly if rev in jolly_reviewers else max_ass
            current = G.nodes[rev].get('review_count', 0)
            return current / limit if limit > 0 else 1
            
        # (Score di carico, Priorità Jolly: 0 se jolly 1 se standard, Random per sbloccare i pareggi)
        ranked = [
            (score(rev), 0 if rev in jolly_reviewers else 1, random.random(), rev) 
            for rev in candidates
        ]
        
        ranked.sort(key=lambda x: (x[0], x[1], x[2]))
        return [rev for (_, _, _, rev) in ranked]

    # --- INIZIO ASSEGNAZIONE ---
    selected = []
    
    # Step 1: MATCH PERFETTO (Topic OK + 0 Collaborazioni)
    candidates_step1 = get_candidates(all_potential_reviewers, max_collabs=0)
    
    # Se è un Late Poster, diamo priorità assoluta ai Jolly in fase di ranking (già gestito da rank_reviewers)
    if abstract_type == "Late Poster":
        # Filtriamo solo i jolly che superano i requisiti stringenti
        jolly_candidates_step1 = [c for c in candidates_step1 if c in jolly_reviewers]
        selected.extend(rank_reviewers(jolly_candidates_step1)[:required_reviewers])
    else:
        selected.extend(rank_reviewers(candidates_step1)[:required_reviewers])

    # Step 2: FALLBACK (Topic OK + Max 1 Collaborazione)
    if len(selected) < required_reviewers:
        remaining = required_reviewers - len(selected)
        candidates_step2 = get_candidates(all_potential_reviewers, max_collabs=1)
        # Rimuoviamo chi è già stato selezionato nello Step 1
        candidates_step2 = [c for c in candidates_step2 if c not in selected]
        
        if abstract_type == "Late Poster":
            jolly_candidates_step2 = [c for c in candidates_step2 if c in jolly_reviewers]
            selected.extend(rank_reviewers(jolly_candidates_step2)[:remaining])
        else:
            selected.extend(rank_reviewers(candidates_step2)[:remaining])

    # Step 3: CONTROLLO FALLIMENTO E LOGGING ERRORI
    if len(selected) < required_reviewers:
        missing = required_reviewers - len(selected)
        
        # Stampa l'errore ben visibile a schermo e nei log
        error_msg = (
            f"\n{'='*50}\n"
            f"ERRORE: ASSEGNAZIONE FALLITA\n"
            f"Abstract ID: {abstract_num}\n"
            f"Titolo: '{abstract_title}'\n"
            f"Topic: {topic}\n"
            f"Tipo: {abstract_type}\n"
            f"Trovati {len(selected)} revisori su {required_reviewers} necessari.\n"
            f"Motivo: Nessun revisore disponibile con il topic corretto e un massimo di 1 collaborazione pregressa.\n"
            f"{'='*50}\n"
        )
        logging.critical(error_msg)
    
    # Aggiornamento contatore per i revisori effettivamente assegnati
    for reviewer in selected:
        G.nodes[reviewer]['review_count'] = G.nodes[reviewer].get('review_count', 0) + 1
        
    return selected if len(selected) > 0 else None

def save_final_assignments(G, authors_df, max_ass = 6, jolly_revs=None, max_ass_jolly = 10):
    """
    Salva due file json:
    1) Per ogni abstract: abstract_id, titolo, sessione, tipo e 3 revisori 
    2) Per ogni reviewer: nome, numero di abstract assegnati, e la lista di abstract ids
    """
    

    # File per abstracts -----------------------------------
    abstract_assignments = []
    assignment_records = []  
    not_assigned = 0
    
    for abstract_id in authors_df['abstract_id'].unique():

        # generate list of assigned reviewers
        assigned_reviewers = reviewer_choice(
        G=G, 
        abstract_num=abstract_id, 
        abstract_df=authors_df, 
        req_reviewers_map=req_reviewers_map, 
        max_ass=max_assign,             
        jolly_revs=jolly_revs,          
        max_ass_jolly=max_assign_jolly  
)
        if not assigned_reviewers:
            not_assigned +=1
            logging.info(f"Nessun revisore assegnato per l'abstract {abstract_id}")
            continue
        
        # abstract info
        abstract_info = authors_df[authors_df['abstract_id'] == abstract_id].iloc[0]
        
        assignment = {
            'abstract_id': abstract_id,
            'abstract_title': abstract_info['abstract_title'],
            'session': abstract_info['session'],
            'type': abstract_info['type'],
        }
        required_reviewers = 2 if assignment['type'] == "Poster" else 3
        # print(assigned_reviewers)s
        for i, reviewer_id in enumerate(assigned_reviewers[: required_reviewers], 1):
            assignment[f'reviewer{i}_name'] = G.nodes[reviewer_id]['given_name']
            assignment[f'reviewer{i}_id'] = G.nodes[reviewer_id]['bits_id']
        
            
            assignment_records.append({
                'reviewer_id': reviewer_id,
                'abstract_id': abstract_id,
                'abstract_title': abstract_info['abstract_title'],
                'session': abstract_info['session'],
                'type': abstract_info['type'],
            })
        
        abstract_assignments.append(assignment)

    # save abstract file
    abstract_assignments_df = pd.DataFrame(abstract_assignments)
    abstract_assignments_df[abstract_assignments_df.filter(regex='reviewer.*_id').columns] = \
        abstract_assignments_df.filter(regex='reviewer.*_id')\
        .apply(pd.to_numeric, errors='coerce')\
        .astype('Int64') # per convertire gli id in interi 
 
    os.makedirs(os.path.dirname(abstract_reviewers), exist_ok=True)
    with open(abstract_reviewers, mode='w', newline='\n') as f:
        f.write('[\n')
        
        for idx, row in abstract_assignments_df.iterrows():
            row = row.dropna().to_json(f, force_ascii=False)
            if idx < len(abstract_assignments_df) - 1:
                f.write(',\n')

        f.write(']\n')
    # File per reviewers ----------------------------------
    if assignment_records:
        all_assignments = pd.DataFrame(assignment_records)
        
        # group by reviewer
        reviewer_groups = all_assignments.groupby('reviewer_id')

        reviewer_data = []
        for reviewer_id, group in reviewer_groups:
            reviewer_node = G.nodes[reviewer_id]
            reviewer_data.append({
                'reviewer_name': reviewer_node['given_name'],
                'reviewer_bits_id': reviewer_node['bits_id'],
                'total_assignments': len(group),
                'assigned_abstracts_ids': "; ".join(map(str, group['abstract_id'])),
            })
        
        # save reviewers file
        pd.DataFrame(reviewer_data).to_json(
            reviewers_assignment, orient='records', force_ascii=False
        )
    logging.info(f"Number of assigned abstracts: {len(authors_df['abstract_id'].unique()) - not_assigned}")
    logging.info(f"Number of NON assigned abstracts: {not_assigned}")
    logging.info(f"Assignments Saved.")