from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import openai
import pickle
import tiktoken

tokenizer = tiktoken.encoding_for_model('text-embedding-ada-002')
def count_tokens(text: str) -> int:
    tokens = [tokenizer.decode([x]) for x in tokenizer.encode(text)]
    return len(tokens)

df = pd.read_csv('data.csv', sep = '$')
df['tokens'] = [count_tokens(c) for c in df['contents'].values]

COMPLETIONS_MODEL = 'gpt-3.5-turbo'
EMBEDDING_MODEL = 'text-embedding-ada-002'

MAX_SECTION_LEN = 4000

COMPLETIONS_API_PARAMS = {
    'temperature': 0.0,
    'top_p': 0.0,
    'max_tokens': 0,
    'model': COMPLETIONS_MODEL,
    'stream': True,
    'user': 'aiteam'
}

with open('doc_embeddings.pickle', 'rb') as handle:
    document_embeddings = pickle.load(handle)

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    result = openai.Embedding.create(
      model = model,
      input = text
    )
    return result['data'][0]['embedding']

def vector_similarity(x: List[float], y: List[float]) -> float:
    return np.dot(np.array(x), np.array(y)) / (np.linalg.norm(np.array(x)) * np.linalg.norm(np.array(y)))

def order_document_sections_by_query_similarity(query: str, document_embeddings: Dict[Tuple[str, str], np.ndarray]) -> List[Tuple[float, Tuple[str, str]]]:
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in document_embeddings.items()
    ], reverse = True)
    
    return document_similarities

def get_document(question: str, context_embeddings: dict = document_embeddings, df: pd.DataFrame = df) -> str:
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    q_tokens = count_tokens(question)
    
    for _, section_index in most_relevant_document_sections:
        idx = df.index[(df['products'] == section_index[0]) & (df['indexes'] == section_index[1])][0]   
        document_section = df.iloc[[idx]]
        
        if document_section.tokens.values[0] > MAX_SECTION_LEN:
            break
        
        return str(section_index), document_section.contents.values[0]
    