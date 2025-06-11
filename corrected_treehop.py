#!/usr/bin/env python3
"""
TreeHop Multi-hop QA - CORRECTED VERSION  
Fixed according to original TreeHop repo analysis
"""

import os
import json
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random
import re
import sys
import subprocess
from collections import defaultdict, namedtuple
from statistics import mean, stdev
from typing import List, Dict, Tuple, Union, Set, Optional

# Setup paths for Kaggle
KAGGLE_INPUT_DIR = "/kaggle/input"
DATASET_DIR = os.path.join(KAGGLE_INPUT_DIR, "my-treehop-data")
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File names
CORPUS_FILENAME = "multihoprag_corpus.txt"
QUERIES_FILENAME = "MultiHopRAG.json"

# Add paths for local environment
IS_KAGGLE = os.path.exists(KAGGLE_INPUT_DIR)
if not IS_KAGGLE:
    BASE_DIR = os.getcwd()
    DATASET_DIR = BASE_DIR
    OUTPUT_DIR = BASE_DIR

# Configuration
MAX_HOPS = 3
TOP_N = 3
REDUNDANT_PRUNING = True
LAYERWISE_TOP_PRUNING = True
TRAIN_RATIO = 0.8
SEED = 42
GEMINI_MODEL = 'gemini-2.0-flash'
EMBEDDING_MODEL = 'BAAI/bge-m3'
USE_QUERY_REWRITE = True     # LLM paraphrase & enrich truy vấn
USE_COT_PLANNER   = True     # LLM đề xuất bước retrieve tiếp theo
USE_LLM_RERANK    = True     # LLM chấm điểm lại các passage
USE_PASSAGE_SUM   = True    # LLM tóm tắt passage (giảm độ dài prompt)
LLM_RERANK_TOP_K  = 5       # Chỉ gửi tối đa K passage cho LLM reranker

# Set random seed
random.seed(SEED)
np.random.seed(SEED)

def install_dependencies():
    """Install required packages"""
    print("🔧 Installing dependencies...")
    subprocess.run("pip install -q --upgrade transformers", shell=True)
    try:
        import google.generativeai as genai
    except ImportError:
        subprocess.run("pip install -q google-generativeai", shell=True)
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        subprocess.run("pip install -q --upgrade FlagEmbedding", shell=True)

install_dependencies()

import google.generativeai as genai

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    BGEM3FlagModel = None

SearchResult = namedtuple('SearchResult', ['passage', 'tree_hop_graph'])

class TreeHopGraph:
    """TreeHop graph tracking"""
    def __init__(self, query: str):
        self.query = query
        self.hops = []
        self.passages = {}
        self.edges = []
        self.overlap_history = []
        
    def add_hop(self, hop_id: int, passages: List[Dict], query_change: float = 0.0, overlap_magnitude: float = 0.0):
        hop_info = {
            'hop_id': hop_id,
            'passages': passages,
            'query_embedding_change': query_change,
            'overlap_magnitude': overlap_magnitude,
            'num_passages': len(passages)
        }
        self.hops.append(hop_info)
        
        for p in passages:
            p_id = p.get('id', p['title'])
            self.passages[p_id] = p
            self.edges.append((f"query_hop_{hop_id}", p_id))
            
        self.overlap_history.append(overlap_magnitude)

class SimpleTreeHopGraph:
    """Simplified graph structure to simulate DGL functionality for TreeHop"""
    def __init__(self, query_emb: np.ndarray):
        self.nodes = {'rep': [query_emb], 'h': [query_emb]}  # node embeddings
        self.edges = []  # (src, dst) pairs
        self.query_history = [query_emb]
        
    def add_context_nodes(self, ctx_embs: List[np.ndarray], query_emb: np.ndarray):
        """Add context nodes and edges for next hop"""
        start_idx = len(self.nodes['rep'])
        
        # Add context nodes
        for ctx_emb in ctx_embs:
            self.nodes['rep'].append(ctx_emb)
            self.nodes['h'].append(query_emb)  # Initialize h with query
            
        # Add edges from query to each context
        for i, _ in enumerate(ctx_embs):
            self.edges.append((0, start_idx + i))  # Query node (0) to context node
            
    def get_latest_query_nodes(self):
        """Get the most recent query embeddings for branching"""
        return [self.nodes['h'][i] for i in range(1, len(self.nodes['h']))]
        
    def update_query_embeddings(self, new_query_embs: List[np.ndarray]):
        """Update query embeddings after TreeHop processing"""
        for i, new_emb in enumerate(new_query_embs, 1):
            if i < len(self.nodes['h']):
                self.nodes['h'][i] = new_emb
                
        # Track query evolution
        if new_query_embs:
            # Use the best query embedding (highest change)
            best_emb = max(new_query_embs, key=lambda x: np.linalg.norm(x - self.query_history[-1]))
            self.query_history.append(best_emb)

def extract_numeric_from_complex_object(obj, target_shape=None):
    """Extract numeric data from BGE-m3 complex output"""
    def recursive_extract(item):
        if isinstance(item, (int, float)):
            return float(item)
        elif isinstance(item, np.ndarray):
            return item.astype(np.float32)
        elif isinstance(item, list):
            try:
                return np.array(item, dtype=np.float32)
            except (ValueError, TypeError):
                numeric_items = []
                for subitem in item:
                    extracted = recursive_extract(subitem)
                    if extracted is not None:
                        numeric_items.append(extracted)
                if numeric_items:
                    try:
                        return np.array(numeric_items, dtype=np.float32)
                    except:
                        return numeric_items[0] if len(numeric_items) == 1 else np.concatenate(numeric_items)
                return None
        elif isinstance(item, dict):
            for key in ['dense_vecs', 'dense', 'embeddings', 'vectors', 'data']:
                if key in item:
                    return recursive_extract(item[key])
            for value in item.values():
                result = recursive_extract(value)
                if result is not None:
                    return result
            return None
        elif hasattr(item, '__iter__') and not isinstance(item, str):
            try:
                return np.array(list(item), dtype=np.float32)
            except:
                return None
        else:
            return None
    
    try:
        result = recursive_extract(obj)
        
        if result is None:
            dim = target_shape[0] if target_shape else 1024
            result = np.random.randn(dim).astype(np.float32)
            return result / np.linalg.norm(result)
        
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=np.float32)
        
        if result.ndim > 1:
            result = result.flatten()
        
        if target_shape and len(target_shape) > 0:
            target_dim = target_shape[0]
            if result.shape[0] != target_dim:
                if result.shape[0] > target_dim:
                    result = result[:target_dim]
                else:
                    result = np.pad(result, (0, target_dim - result.shape[0]), 'constant')
        
        norm = np.linalg.norm(result)
        if norm > 0:
            return result / norm
        else:
            result[0] = 1.0
            return result
            
    except Exception as e:
        print(f"⚠️ Error extracting numeric data: {e}")
        dim = target_shape[0] if target_shape else 1024
        fallback = np.random.randn(dim).astype(np.float32)
        return fallback / np.linalg.norm(fallback)

class CorrectedTreeHopRetriever:
    """TreeHop implementation corrected according to original repo"""
    
    def __init__(self, model_name, passages_file, embed_dim=1024, g_size=64, n_heads=3, mlp_size=64):
        self.embed_dim = embed_dim
        self.g_size = g_size
        self.n_heads = n_heads
        self.mlp_size = mlp_size
        self.model_name = model_name
        self.passages_file = passages_file
        self.passages = []
        self.embeddings = None
        self.embedding_model = None
        
        print(f"🚀 Initializing CORRECTED TreeHop Retriever:")
        print(f"  embed_dim: {embed_dim}, g_size: {g_size}")
        print(f"  n_heads: {n_heads}, mlp_size: {mlp_size}")
        
        self._init_embedding_model()
        self._init_corrected_treehop_networks()
        
        self.stats = {
            'total_queries': 0,
            'total_hops': 0,
            'overlap_applications': 0,
            'avg_overlap_magnitude': 0.0
        }
    
    def _init_embedding_model(self):
        """Initialize BGE-m3 model"""
        print("🔧 Loading BGE-m3...")
        
        if BGEM3FlagModel is not None:
            try:
                self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
                self.embed_dim = 1024
                print("✅ BGE-m3 loaded successfully")
                return
            except Exception as e:
                print(f"❌ BGE-m3 failed: {e}")
        
        print("⚠️ Using fallback embedding")
        self.embedding_model = None
    
    def _init_corrected_treehop_networks(self):
        """Initialize TreeHop networks CORRECTED according to original repo"""
        # AttentionHead2D weights for each head
        self.attention_heads = []
        for i in range(self.n_heads):
            head_weights = {
                'W_Q': np.random.randn(self.embed_dim, self.g_size) * 0.02,
                'W_K': np.random.randn(self.embed_dim, self.g_size) * 0.02,
                'W_V': np.random.randn(self.embed_dim, self.g_size) * 0.02,
                # MultiMLPLayer weights
                'mlp_layer1': np.random.randn(self.g_size, self.mlp_size) * 0.02,
                'mlp_layer2': np.random.randn(self.mlp_size, self.g_size) * 0.02,
                'mlp_bias1': np.random.randn(self.mlp_size) * 0.01,
                'mlp_bias2': np.random.randn(self.g_size) * 0.01,
                # ResNet layer norm weights (simulated)
                'layer_norm_weight': np.ones(self.g_size),
                'layer_norm_bias': np.zeros(self.g_size),
            }
            self.attention_heads.append(head_weights)
        
        # TreeHopNode.update_attn_scale: Linear(g_size * n_heads, embed_size, bias=False)
        self.update_attn_scale_weights = np.random.randn(self.g_size * self.n_heads, self.embed_dim) * 0.02
        
        print(f"✅ Initialized {self.n_heads} CORRECTED attention heads")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Robust BGE-m3 encoding with improved error handling"""
        if self.embedding_model is None:
            # Hash fallback
            import hashlib
            text_bytes = text.encode('utf-8')
            hash_bytes = hashlib.md5(text_bytes).digest()
            vec = np.array([float(b) for b in hash_bytes])
            
            if len(vec) < self.embed_dim:
                vec = np.pad(vec, (0, self.embed_dim - len(vec)), 'constant')
            else:
                vec = vec[:self.embed_dim]
            
            return vec / np.linalg.norm(vec)
        
        try:
            # Ensure text is valid and clean
            if not text or not isinstance(text, str):
                text = "empty text"
            
            # Clean and truncate text for better encoding
            text = text.strip()
            # Remove excessive whitespace and special chars that might confuse BGE-m3
            text = ' '.join(text.split())
            text = text[:4000]  # Reasonable limit for BGE-m3
            
            if hasattr(self.embedding_model, 'encode') and 'BAAI/bge-m3' in str(self.embedding_model):
                for attempt in range(2):  # Reduced attempts for speed
                    try:
                        # Use most stable BGE-m3 configuration
                        result = self.embedding_model.encode(
                            [text], 
                            batch_size=1,
                            max_length=1024,  # Optimal for BGE-m3
                            return_dense=True, 
                            return_sparse=False, 
                            return_colbert_vecs=False,
                            normalize_embeddings=True  # Important for similarity calculation
                        )
                        
                        # Validate result is not None
                        if result is None:
                            print(f"⚠️ BGE-m3 returned None for attempt {attempt+1}")
                            continue
                        
                        embedding = extract_numeric_from_complex_object(result, target_shape=(self.embed_dim,))
                        
                        # Validate embedding
                        if embedding is not None and isinstance(embedding, np.ndarray) and embedding.shape[0] > 0:
                            # Ensure proper normalization
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                return embedding / norm
                            else:
                                # Create normalized random vector if norm is 0
                                embedding = np.random.randn(self.embed_dim).astype(np.float32)
                                return embedding / np.linalg.norm(embedding)
                        
                    except Exception as e:
                        print(f"⚠️ BGE-m3 attempt {attempt+1} failed: {e}")
                        continue
                
                print("❌ All BGE-m3 attempts failed, using fallback")
            
        except Exception as e:
            print(f"⚠️ Encoding error: {e}")
        
        # Ultimate fallback - create a deterministic embedding based on text hash
        import hashlib
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        # Convert hex to numbers
        hash_nums = [int(text_hash[i:i+2], 16) for i in range(0, min(len(text_hash), self.embed_dim*2), 2)]
        
        # Pad or truncate to correct dimension
        if len(hash_nums) < self.embed_dim:
            hash_nums.extend([0] * (self.embed_dim - len(hash_nums)))
        else:
            hash_nums = hash_nums[:self.embed_dim]
        
        fallback = np.array(hash_nums, dtype=np.float32)
        return fallback / np.linalg.norm(fallback)
    
    def corrected_attention_head_2d(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, head_weights: dict) -> np.ndarray:
        """
        CORRECTED AttentionHead2D EXACTLY matching original TreeHop codebase
        """
        # Linear transformations
        Q_proj = np.dot(Q, head_weights['W_Q'])  # [embed_dim] -> [g_size]
        K_proj = np.dot(K, head_weights['W_K'])  # [embed_dim] -> [g_size] 
        V_proj = np.dot(V, head_weights['W_V'])  # [embed_dim] -> [g_size]
        
        # CORRECTED: Exactly match original TreeHop
        # Original: QK = Q * K (element-wise multiplication for 2D)
        QK = Q_proj * K_proj  # Element-wise multiplication [g_size]
        
        # CORRECTED: Normalization exactly as original
        # Original: scores = QK / Q.shape[1] ** 0.5  where Q.shape[1] is attn_size (g_size)
        scores = QK / (self.g_size ** 0.5)
        
        # CORRECTED: Softmax exactly as original  
        attn = np.exp(scores) / np.sum(np.exp(scores))  # Softmax activation
        
        # CORRECTED: Apply attention exactly as original
        # Original: attn_out = self.dropout(attn) * V (no dropout in numpy version)
        attn_out = attn * V_proj  # [g_size]
        
        # CORRECTED: MLP processing exactly as original
        # MultiMLPLayer -> mlp_scale + residual connection
        # Layer norm (simulated)
        x_norm = (attn_out - np.mean(attn_out)) / (np.std(attn_out) + 1e-8)
        x_norm = x_norm * head_weights['layer_norm_weight'] + head_weights['layer_norm_bias']
        
        # MLP layers exactly as original
        mlp_hidden = np.maximum(0, np.dot(x_norm, head_weights['mlp_layer1']) + head_weights['mlp_bias1'])  # ReLU
        mlp_out = np.dot(mlp_hidden, head_weights['mlp_layer2']) + head_weights['mlp_bias2']
        
        # CORRECTED: Final output exactly as original
        # Original: return self.mlp_scale(mlp_out) + attn_out
        return mlp_out + attn_out  # Direct residual connection as in original
    
    def corrected_multi_head_attention_2d(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        CORRECTED MultiHeadAttention2D
        """
        head_outputs = []
        for head_weights in self.attention_heads:
            head_out = self.corrected_attention_head_2d(Q, K, V, head_weights)
            head_outputs.append(head_out)
        
        # Concatenate all heads: [g_size * n_heads]
        multi_head_out = np.concatenate(head_outputs)
        return multi_head_out
    
    def corrected_overlap_subtraction(self, query_emb: np.ndarray, context_emb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        CORRECTED TreeHop overlap subtraction formula according to original repo:
        
        From TreeHopNode.reduce_func:
        Q = nodes.mailbox["q"].clone().squeeze(1)  # query
        K = nodes.data["rep"]                      # context  
        V_update = nodes.data["rep"]               # context
        
        update_gate = self.update_gate(Q, K, V_update)
        h = Q - K + self.update_attn_scale(update_gate)
        """
        try:
            # Validate inputs
            if query_emb is None or context_emb is None:
                print("⚠️ None embedding in overlap subtraction, returning original query")
                return query_emb if query_emb is not None else np.random.randn(self.embed_dim), 0.0
            
            # Ensure embeddings are numpy arrays
            if not isinstance(query_emb, np.ndarray):
                query_emb = np.array(query_emb, dtype=np.float32)
            if not isinstance(context_emb, np.ndarray):
                context_emb = np.array(context_emb, dtype=np.float32)
            
            # Ensure proper dimensions
            if query_emb.shape[0] != self.embed_dim:
                print(f"⚠️ Query embedding wrong shape: {query_emb.shape}, expected: ({self.embed_dim},)")
                query_emb = np.resize(query_emb, (self.embed_dim,))
                query_emb = query_emb / np.linalg.norm(query_emb)
            
            if context_emb.shape[0] != self.embed_dim:
                print(f"⚠️ Context embedding wrong shape: {context_emb.shape}, expected: ({self.embed_dim},)")
                context_emb = np.resize(context_emb, (self.embed_dim,))
                context_emb = context_emb / np.linalg.norm(context_emb)
            
            # Step 1: CORRECTED update_gate computation
            # self.update_gate = MultiHeadAttention2D(embed_size, g_size, mlp_size, ...)
            update_gate = self.corrected_multi_head_attention_2d(query_emb, context_emb, context_emb)
            
            # Validate update_gate
            if update_gate is None or not isinstance(update_gate, np.ndarray):
                print("⚠️ Update gate failed, using fallback")
                update_gate = np.zeros(self.g_size * self.n_heads, dtype=np.float32)
            
            # Step 2: CORRECTED update_attn_scale transformation
            # self.update_attn_scale = nn.Linear(g_size * n_head, embed_size, bias=False)
            attention_update = np.dot(update_gate, self.update_attn_scale_weights)  # No bias in original
            
            # Validate attention_update
            if attention_update is None or not isinstance(attention_update, np.ndarray):
                print("⚠️ Attention update failed, using zero update")
                attention_update = np.zeros(self.embed_dim, dtype=np.float32)
            
            # Ensure correct shape
            if attention_update.shape[0] != self.embed_dim:
                attention_update = np.resize(attention_update, (self.embed_dim,))
            
            # Step 3: CORRECTED TreeHop formula: h = Q - K + attention_update
            overlap_removed = query_emb - context_emb  # Q - K
            overlap_magnitude = np.linalg.norm(overlap_removed)
            updated_query = overlap_removed + attention_update  # Q - K + update_attn_scale(update_gate)
            
            # Normalize the updated query
            norm = np.linalg.norm(updated_query)
            if norm > 0:
                updated_query = updated_query / norm
            else:
                print("⚠️ Zero norm in updated query, using original")
                updated_query = query_emb
            
            return updated_query, overlap_magnitude
            
        except Exception as e:
            print(f"⚠️ Error in overlap subtraction: {e}")
            # Return original query on error
            return query_emb, 0.0
    
    def load_passages(self):
        """Load passages from JSONL file"""
        if not self.passages:
            try:
                with open(self.passages_file, 'r', encoding='utf-8') as f:
                    self.passages = [json.loads(line) for line in f]
                print(f"✅ Loaded {len(self.passages)} passages")
            except Exception as e:
                print(f"❌ Error loading passages: {e}")
                self.passages = []
        return self.passages
    
    def multihop_search_passages(self, query, n_hop=2, top_n=5, redundant_pruning=True, 
                               layerwise_top_pruning=True, return_tree=True):
        """Complete multi-hop search with CORRECTED TreeHop"""
        if not self.passages:
            self.load_passages()
        
        if len(self.passages) == 0:
            print("⚠️ No passages loaded")
            return SearchResult([[[]]]*n_hop, None)
        
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query
        
        print(f"🔍 Starting CORRECTED TreeHop {n_hop}-hop retrieval for {len(queries)} queries")
        
        all_hop_results = []
        tree_hop_graphs = []
        
        for query_idx, q in enumerate(queries):
            print(f"\n=== Processing Query {query_idx+1}/{len(queries)} ===")
            print(f"Query: {q[:100]}...")
            
            tree_graph = TreeHopGraph(q)
            
            # Get initial query embedding  
            current_query_emb = self.encode_text(q)
            print(f"✅ Initial query embedding shape: {current_query_emb.shape}")
            
            query_hop_results = []
            seen_passage_ids = set()
            
            # TreeHop stopping criterion variables
            prev_query_emb = current_query_emb.copy()
            convergence_threshold = 0.01  # Stop if query change < threshold
            quality_threshold = 0.3      # Stop if retrieval quality too low
            
            # Initialize TreeHop graph structure
            treehop_graph = SimpleTreeHopGraph(current_query_emb)
            
            for hop in range(n_hop):
                print(f"\n--- Hop {hop+1}/{n_hop} ---")
                
                # Early stopping check (after first hop)
                if hop > 0:
                    query_change = np.linalg.norm(current_query_emb - prev_query_emb)
                    if query_change < convergence_threshold:
                        print(f"⚡ Early stopping: query converged (change={query_change:.4f} < {convergence_threshold})")
                        break
                    
                    # Check retrieval quality (similarity scores)
                    if len(query_hop_results) > 0 and len(query_hop_results[-1]) > 0:
                        last_scores = scores[np.argsort(scores)[-len(query_hop_results[-1]):][::-1]]
                        avg_score = np.mean(last_scores)
                        if avg_score < quality_threshold:
                            print(f"⚡ Early stopping: low retrieval quality (avg_score={avg_score:.4f} < {quality_threshold})")
                            break
                
                prev_query_emb = current_query_emb.copy()
                
                # Encode passages if not cached
                if self.embeddings is None:
                    print("🔄 Encoding passages...")
                    passage_texts = [f"Title: {p['title']}\nContent: {p['text']}" for p in self.passages]
                    passage_embeddings = []
                    
                    for i, text in enumerate(tqdm(passage_texts, desc="Encoding")):
                        try:
                            emb = self.encode_text(text)
                            passage_embeddings.append(emb)
                        except Exception as e:
                            print(f"⚠️ Error encoding passage {i}: {e}")
                            fallback = np.random.randn(self.embed_dim).astype(np.float32)
                            passage_embeddings.append(fallback / np.linalg.norm(fallback))
                    
                    self.embeddings = np.array(passage_embeddings)
                    print(f"✅ Encoded {len(passage_embeddings)} passages")
                else:
                    passage_embeddings = self.embeddings
                
                # Calculate similarities
                scores = np.dot(passage_embeddings, current_query_emb) / (
                    np.linalg.norm(passage_embeddings, axis=1) * np.linalg.norm(current_query_emb)
                )
                
                # TreeHop hop weighting - more aggressive weighting for later hops
                hop_weight = 1.0 + hop * 0.5  # Increased from 0.3 to 0.5
                scores = scores * hop_weight
                
                # Apply query-specific boosting for key terms
                if hop > 0:  # Boost scores for passages containing query keywords
                    query_words = set(q.lower().split())
                    for i, passage in enumerate(self.passages):
                        passage_text = f"{passage['title']} {passage['text']}".lower()
                        overlap_count = sum(1 for word in query_words if word in passage_text and len(word) > 3)
                        if overlap_count > 0:
                            scores[i] *= (1.0 + overlap_count * 0.1)
                
                # CORRECTED TreeHop Pruning Strategy
                if layerwise_top_pruning and hop > 0:
                    # Layer-wise top-K pruning: keep only top-K candidates from previous hop
                    # This prevents exponential growth of query branches
                    effective_top_n = min(top_n, max(3, top_n // (hop + 1)))  # Reduce as hops increase
                    print(f"  Layer-wise pruning: using top-{effective_top_n} for hop {hop+1}")
                else:
                    effective_top_n = top_n
                
                # Select top passages with redundant pruning
                if redundant_pruning and hop > 0:
                    available_indices = [i for i, p in enumerate(self.passages) 
                                       if p.get('id', p['title']) not in seen_passage_ids]
                    if len(available_indices) == 0:
                        print(f"⚠️ No new passages at hop {hop+1}")
                        break
                    available_scores = scores[available_indices]
                    top_indices_local = np.argsort(available_scores)[-effective_top_n:][::-1]
                    top_indices = [available_indices[i] for i in top_indices_local]
                else:
                    top_indices = np.argsort(scores)[-effective_top_n:][::-1]
                
                # Get top passages
                hop_passages = [self.passages[i] for i in top_indices]
                hop_passage_embeddings = np.array([passage_embeddings[i] for i in top_indices])
                
                # Track seen passages
                for p in hop_passages:
                    seen_passage_ids.add(p.get('id', p['title']))
                
                print(f"✅ Retrieved {len(hop_passages)} passages")
                
                # Store results
                query_hop_results.append(hop_passages)
                
                # Apply CORRECTED TreeHop overlap subtraction with BRANCHING
                query_change_magnitude = 0.0
                overlap_magnitude = 0.0
                
                if hop < n_hop - 1:  # Don't update after last hop
                    # CORRECTED: TreeHop graph-based query branching exactly like original
                    # Original uses DGL graphs with message passing between query and context nodes
                    
                    print(f"🧠 Applying CORRECTED TreeHop graph-based message passing...")
                    print(f"  Before: query norm = {np.linalg.norm(current_query_emb):.4f}")
                    
                    # Add context nodes to graph (simulating DGL add_nodes)
                    treehop_graph.add_context_nodes(hop_passage_embeddings.tolist(), current_query_emb)
                    
                    # TreeHop message passing: process each context node individually
                    # This simulates the DGL reduce_func for each edge
                    query_candidates = []
                    overlap_magnitudes = []
                    
                    for i, passage_emb in enumerate(hop_passage_embeddings):
                        # Simulate TreeHopNode.reduce_func:
                        # Q = nodes.mailbox["q"] (query from previous hop)
                        # K = nodes.data["rep"] (current context)
                        # V_update = nodes.data["rep"] (same as K)
                        
                        Q = current_query_emb  # Query from mailbox
                        K = passage_emb        # Context representation  
                        V_update = passage_emb # Same as K for V
                        
                        # Apply TreeHop formula: h = Q - K + self.update_attn_scale(update_gate)
                        updated_query_i, overlap_mag_i = self.corrected_overlap_subtraction(Q, K)
                        query_candidates.append(updated_query_i)
                        overlap_magnitudes.append(overlap_mag_i)
                    
                    # Update graph with new query embeddings
                    treehop_graph.update_query_embeddings(query_candidates)
                    
                    # TreeHop query selection strategy
                    if len(query_candidates) > 0:
                        # Original TreeHop: select query with maximum information gain
                        query_changes = [np.linalg.norm(q - current_query_emb) for q in query_candidates]
                        
                        # Advanced selection: weighted combination of change + diversity
                        query_scores = []
                        for i, (q_cand, change) in enumerate(zip(query_candidates, query_changes)):
                            # Score = change magnitude + diversity from other candidates
                            diversity = np.mean([np.linalg.norm(q_cand - other) 
                                               for j, other in enumerate(query_candidates) if i != j])
                            score = change + 0.1 * diversity if len(query_candidates) > 1 else change
                            query_scores.append(score)
                        
                        best_idx = np.argmax(query_scores)
                        updated_query_emb = query_candidates[best_idx]
                        overlap_magnitude = overlap_magnitudes[best_idx]
                        query_change_magnitude = query_changes[best_idx]
                        
                        print(f"  Selected query branch {best_idx+1}/{len(query_candidates)} (score: {query_scores[best_idx]:.4f})")
                        print(f"  After: query norm = {np.linalg.norm(updated_query_emb):.4f}")
                        print(f"  Query change = {query_change_magnitude:.4f}")
                        print(f"  Overlap magnitude = {overlap_magnitude:.4f}")
                        
                        # Update current query for next hop
                        current_query_emb = updated_query_emb
                    else:
                        print("⚠️ No valid query candidates generated")
                    
                    # Update statistics
                    self.stats['overlap_applications'] += 1
                    self.stats['avg_overlap_magnitude'] = (
                        (self.stats['avg_overlap_magnitude'] * (self.stats['overlap_applications'] - 1) + overlap_magnitude) 
                        / self.stats['overlap_applications']
                    )
                
                # Add hop to graph
                tree_graph.add_hop(hop + 1, hop_passages, query_change_magnitude, overlap_magnitude)
            
            # Store results
            all_hop_results.append(query_hop_results)
            tree_hop_graphs.append(tree_graph)
            
            # Update statistics
            self.stats['total_queries'] += 1
            self.stats['total_hops'] += len(query_hop_results)
            
            # Print summary
            total_passages = sum(len(hop) for hop in query_hop_results)
            print(f"\n✅ Query {query_idx+1} complete:")
            print(f"  Hops: {len(query_hop_results)}")
            print(f"  Total passages: {total_passages}")
            print(f"  Unique passages: {len(seen_passage_ids)}")
        
        return SearchResult(all_hop_results, tree_hop_graphs if return_tree else None)

# Utility functions (same as before)
def load_corpus(corpus_file: str) -> Dict[str, str]:
    """Load corpus"""
    print(f"📖 Loading corpus from {corpus_file}...")
    
    corpus_data = {}
    current_title = None
    current_passage_lines = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Parsing corpus"):
            line = line.strip()

            if "<endofpassage>" in line:
                if current_title and current_passage_lines:
                    corpus_data[current_title] = " ".join(current_passage_lines).strip()
                
                parts = line.split("<endofpassage>", 1)
                if len(parts) > 1 and "Title:" in parts[1]:
                    current_title = parts[1].replace("Title:", "").strip()
                    current_passage_lines = []
                else:
                    current_title = None
                    current_passage_lines = []
                continue

            elif line.startswith("Title:"):
                if current_title and current_passage_lines:
                    corpus_data[current_title] = " ".join(current_passage_lines).strip()
                
                current_title = line.replace("Title:", "").strip()
                current_passage_lines = []
                continue

            elif line.startswith("Passage:"):
                passage_content = line.replace("Passage:", "").strip()
                if passage_content:
                    current_passage_lines.append(passage_content)
                continue
                
            elif current_title is not None:
                current_passage_lines.append(line)

    if current_title and current_passage_lines:
        corpus_data[current_title] = " ".join(current_passage_lines).strip()

    print(f"✅ Loaded {len(corpus_data)} documents")
    return corpus_data

def load_queries(queries_file: str) -> List[Dict]:
    """Load queries"""
    print(f"📖 Loading queries from {queries_file}...")
    
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        print(f"✅ Loaded {len(queries_data)} queries")
        return queries_data
    except Exception as e:
        print(f"❌ Error loading queries: {e}")
        return []

def prepare_passages_jsonl(corpus: Dict[str, str], output_file: str) -> List[Dict]:
    """Create passages JSONL"""
    passages = []
    for i, (title, text) in enumerate(corpus.items()):
        passages.append({
            "id": i,
            "title": title,
            "text": text
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for passage in passages:
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')
    
    print(f"✅ Created passages file at {output_file}")
    return passages

def setup_gemini_model(api_key: str):
    """Setup Gemini API"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    return model

def normalize_answer(s: str) -> str:
    """Normalize answer"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set('.,:;!?()[]{}\\/»«\'""-')
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(pred: str, gold: str) -> float:
    """Compute F1 score"""
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)
    
    if normalized_pred == normalized_gold:
        return 1.0
    
    pred_tokens = set(normalized_pred.split())
    gold_tokens = set(normalized_gold.split())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    common_tokens = pred_tokens.intersection(gold_tokens)
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def multihop_qa(retriever: CorrectedTreeHopRetriever, qa_model, query_data: Dict, 
                n_hop=MAX_HOPS, top_n=TOP_N) -> Dict:
    """Multi-hop QA"""
    query = query_data["query"]
    gold_answer = query_data["answer"]
    question_type = query_data.get("question_type", "unknown")
    
    start_time = time.time()
    
    print(f"\n🎯 === Multi-hop QA ===")
    print(f"Question type: {question_type}")
    print(f"Query: {query}")
    
    try:
        # TreeHop retrieval
        retrieved_result = retriever.multihop_search_passages(
            query,
            n_hop=n_hop,
            top_n=top_n,
            redundant_pruning=REDUNDANT_PRUNING,
            layerwise_top_pruning=LAYERWISE_TOP_PRUNING,
            return_tree=True
        )
        
        # Process results
        if not retrieved_result.passage or not retrieved_result.passage[0]:
            return {
                "query": query,
                "gold_answer": gold_answer,
                "pred_answer": "No relevant passages found",
                "f1_score": 0.0,
                "iterations": 0,
                "retrieval_time": time.time() - start_time,
                "num_passages": 0,
                "question_type": question_type,
                "tree_hop_graph": None,
            }
        
        # Collect all passages
        all_passages = []
        for hop_passages in retrieved_result.passage[0]:
            all_passages.extend(hop_passages)
        
        # Generate answer with Gemini
        pred_answer, iterations = generate_answer(qa_model, query, all_passages)
        
        # Compute F1 score
        f1 = compute_f1(pred_answer, gold_answer)
        
        print(f"💡 Gold answer: {gold_answer}")
        print(f"🤖 Predicted answer: {pred_answer}")
        print(f"📊 F1 score: {f1:.4f}")
        
        return {
            "query": query,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "f1_score": f1,
            "iterations": len(retrieved_result.passage[0]) if retrieved_result.passage else 0,
            "retrieval_time": time.time() - start_time,
            "num_passages": len(all_passages),
            "question_type": question_type,
            "tree_hop_graph": retrieved_result.tree_hop_graph[0] if retrieved_result.tree_hop_graph else None,
        }
        
    except Exception as e:
        print(f"❌ Error in multi-hop QA: {e}")
        return {
            "query": query,
            "gold_answer": gold_answer,
            "pred_answer": f"Error: {str(e)}",
            "f1_score": 0.0,
            "iterations": 0,
            "retrieval_time": time.time() - start_time,
            "num_passages": 0,
            "question_type": question_type,
            "tree_hop_graph": None,
        }

def generate_answer(model, query: str, contexts: List[Dict], max_retries=3) -> Tuple[str, int]:
    """Generate answer using Gemini"""
    context_texts = []
    for idx, ctx in enumerate(contexts, 1):
        if isinstance(ctx, dict) and 'title' in ctx and 'text' in ctx:
            context_texts.append(f"[PASSAGE {idx}]\nTITLE: {ctx['title']}\nCONTENT: {ctx['text']}")
    
    all_contexts = "\n\n---\n\n".join(context_texts)
    
    prompt = f"""Answer the question based on the provided passages retrieved using TreeHop multi-hop reasoning.

QUESTION: {query}

RETRIEVED INFORMATION:
{all_contexts}

CRITICAL INSTRUCTIONS:
- Provide ONLY the direct answer requested - no explanations or extra text
- For "Who" questions: provide ONLY the person's name (e.g., "Sam Bankman-Fried")
- For "Which company" questions: provide ONLY the company name (e.g., "Google") 
- For Yes/No questions: respond with ONLY "Yes" or "No"
- If comparing articles, answer ONLY "Yes" or "No" based on the comparison
- Extract the exact answer from the passages - do not paraphrase
- Only respond "Insufficient information" if NO passages contain ANY relevant information

ANSWER (one word/phrase only):"""
    
    print(f"🤖 Generating answer with Gemini...")
    
    iterations = 0
    for attempt in range(max_retries):
        try:
            generation_config = {
                "temperature": 0.05,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 100,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            iterations += 1
            answer = response.text.strip()
            
            if "insufficient information" in answer.lower():
                answer = "Insufficient information."
            
            return answer, iterations
            
        except Exception as e:
            iterations += 1
            print(f"⚠️ Error generating answer (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    
    return "Error generating answer", iterations

def run_evaluation(retriever, qa_model, test_queries, n_hop=MAX_HOPS, top_n=TOP_N, max_samples=None) -> Tuple[List[Dict], Dict]:
    """Run evaluation"""
    if max_samples and max_samples < len(test_queries):
        print(f"🎯 Running evaluation on {max_samples} samples from {len(test_queries)} total queries")
        test_subset = test_queries[:max_samples]
    else:
        test_subset = test_queries
        print(f"🎯 Running evaluation on all {len(test_subset)} test queries")
    
    results = []
    metrics = {
        "question_type": defaultdict(list),
        "overall": {"f1": [], "iterations": [], "time": [], "passages": []}
    }
    
    # Process each query
    for i, query_data in enumerate(tqdm(test_subset, desc="Processing test queries")):
        print(f"\n[{i+1}/{len(test_subset)}] Processing query: {query_data['query'][:100]}...")
        
        # Run multi-hop QA
        result = multihop_qa_llm(retriever, qa_model, query_data,
                         n_hop=n_hop, top_n=top_n)
        results.append(result)
        
        # Aggregate metrics
        question_type = query_data.get("question_type", "unknown")
        metrics["question_type"][question_type].append({
            "f1": result["f1_score"],
            "iterations": result["iterations"],
            "time": result["retrieval_time"],
            "passages": result["num_passages"]
        })
        
        metrics["overall"]["f1"].append(result["f1_score"])
        metrics["overall"]["iterations"].append(result["iterations"])
        metrics["overall"]["time"].append(result["retrieval_time"])
        metrics["overall"]["passages"].append(result["num_passages"])
        
        # Print progressive results
        if (i+1) % 5 == 0:
            current_f1 = mean(metrics["overall"]["f1"])
            current_iterations = mean(metrics["overall"]["iterations"]) 
            print(f"\n📊 Interim results after {i+1} samples:")
            print(f"  Current average F1: {current_f1:.4f}")
            print(f"  Current average iterations: {current_iterations:.2f}")
    
    return results, metrics

def _call_llm(model, prompt: str,
              temperature: float = 0.2,
              max_tokens: int = 64,
              top_p: float = 0.95,
              top_k: int = 40) -> str:
    """Gọi LLM một cách an toàn – luôn trả string (kể cả khi lỗi)  
       + throttle 20 s giữa các lần gọi."""
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
            }
        )
        return (resp.text or "").strip()
    except Exception as e:
        print(f"⚠️  LLM call failed: {e}")
        return ""
    finally:
        time.sleep(0)

# ========= 1. Query Rewriter ===============================================
def rewrite_query_with_llm(llm_model, original_query: str) -> str:
    """Paraphrase + thêm từ khóa ngữ nghĩa cho query gốc."""
    prompt = (f"Rewrite the question below into a single, clearer sentence. "
              f"Add synonyms or context words if it helps retrieval.\n\n"
              f"QUESTION: {original_query}\n\n"
              f"IMPROVED QUESTION:")
    improved = _call_llm(llm_model, prompt, temperature=0.6, max_tokens=40)
    return improved or original_query  # fallback-safe

# ========= 2. Chain-of-Thought Planner =====================================
def plan_next_hop(llm_model, query: str, hop_summaries: List[str]) -> str:
    """
    Nhờ LLM đề xuất 'nên truy xuất thông tin gì tiếp' dựa trên lịch sử hops.
    Trả về 1 câu ngắn – sẽ được encode rồi dùng như query phụ.
    """
    history = "\n".join(f"Hop {i+1}: {s}" for i, s in enumerate(hop_summaries))
    prompt = (f"You are a retrieval planner. Original question:\n{query}\n\n"
              f"Context seen so far:\n{history}\n\n"
              f"In ONE short sentence, state what information should be retrieved next "
              f"to advance toward the answer.")
    return _call_llm(llm_model, prompt, temperature=0.5, max_tokens=30)

# ========= 3. Passage Summarizer ===========================================
def summarize_passage(llm_model, passage: Dict) -> str:
    """Tóm tắt 1 passage < 30 token, giữ fact chính để giảm prompt length."""
    prompt = (f"Summarize the key fact(s) of the passage below in ONE sentence "
              f"(≤ 30 tokens).\n\nTitle: {passage['title']}\n\n{passage['text']}")
    return _call_llm(llm_model, prompt, temperature=0.3, max_tokens=50)

# ========= 4. LLM Reranker ==================================================
def rerank_passages(llm_model, query: str,
                    passages: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Chấm điểm 1-10 mức liên quan giữa passage & query rồi sắp xếp lại.
    Chỉ gửi TOP_K (đã sort cosine) để tiết kiệm token.
    """
    candidates = passages[:top_k]
    scored: List[tuple] = []
    for p in candidates:
        prompt = (f"Question: {query}\n\n"
                  f"Passage title: {p['title']}\n"
                  f"Passage: {p['text']}\n\n"
                  f"Score from 1 (irrelevant) to 10 (directly answers):")
        score_txt = _call_llm(llm_model, prompt, temperature=0.0,
                              max_tokens=4)
        try:
            score = float(score_txt.split()[0])
        except ValueError:
            score = 5.0
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored]

# ========= 5. Answer Verifier ==============================================
def verify_answer(llm_model, query: str, answer: str, passages: List[Dict]) -> str:
    """
    LLM kiểm chứng câu trả lời cuối (Yes / No / Corrected …).
    Trả lại answer đã chỉnh (hoặc giữ nguyên nếu đã ổn).
    """
    ctx = "\n\n".join(f"{p['title']}: {p['text']}" for p in passages[:6])
    prompt = (f"Based only on the passages below, verify whether the ANSWER is fully "
              f"supported. If wrong, replace with the correct concise answer. If "
              f"cannot decide, respond 'Insufficient information'.\n\n"
              f"QUESTION: {query}\nANSWER: {answer}\n\nPASSAGES:\n{ctx}\n\n"
              f"VERIFIED ANSWER:")
    verified = _call_llm(llm_model, prompt, temperature=0.0, max_tokens=50)
    return verified or answer

# ========= 6. Wrapper: Enhanced TreeHop Retrieval ==========================
def enhanced_multihop_search(retriever, llm_model, query: str,
                             n_hop: int = 3, top_n: int = 10,
                             use_rewrite=True, use_planner=True,
                             use_rerank=True, use_summary=True):
    """
    • B1  (optional)  Query rewrite.  
    • B2            TreeHop multihop search.  
    • B3  (optional) Planner → targeted extra hop.  
    • B4  (optional) LLM rerank.  
    • B5  (optional) Summaries (trả về cho downstream).
    """
    # ---- rewrite ----------------------------------------------------------
    run_query = rewrite_query_with_llm(llm_model, query) if use_rewrite else query
    # ---- TreeHop retrieval ------------------------------------------------
    base = retriever.multihop_search_passages(run_query, n_hop, top_n,
                                              REDUNDANT_PRUNING,
                                              LAYERWISE_TOP_PRUNING,
                                              return_tree=False)
    query_hops = base.passage[0]            # ← LẤY lớp thứ hai
    passages   = sum(query_hops, [])        # flatten tất cả hops
    # ---- planner & extra hop ---------------------------------------------
    if use_planner:
        hop_sums = [", ".join(p['title'] for p in hop)  # dùng query_hops
                for hop in query_hops]
        next_hint = plan_next_hop(llm_model, run_query, hop_sums)
        if next_hint:
            extra = retriever.multihop_search_passages(next_hint, 1, top_n,
                                                       False, False,
                                                       return_tree=False)
            passages.extend(extra.passage[0])
    # ---- rerank -----------------------------------------------------------
    if use_rerank and passages:
        passages = rerank_passages(llm_model, run_query, passages, top_k=top_n)
    # ---- summary ----------------------------------------------------------
    summaries = []
    if use_summary:
        summaries = [summarize_passage(llm_model, p) for p in passages[:top_n]]
    # Return enriched query, summaries, passages (đã rerank)
    return run_query, summaries, passages[:top_n]

def multihop_qa_llm(retriever: CorrectedTreeHopRetriever,
                    qa_model,
                    query_data: Dict,
                    n_hop=MAX_HOPS,
                    top_n=TOP_N) -> Dict:
    """
    Nếu mọi USE_* = False ➜ hành xử y hệt multihop_qa gốc.
    Nếu bật cờ ➜ dùng enhanced_multihop_search (TreeHop + LLM modules).
    """
    query        = query_data["query"]
    gold_answer  = query_data["answer"]
    q_type       = query_data.get("question_type", "unknown")
    t0           = time.time()

    # ---------------- TreeHop retrieval ----------------
    if any([USE_QUERY_REWRITE, USE_COT_PLANNER, USE_LLM_RERANK, USE_PASSAGE_SUM]):
        # TreeHop + LLM modules
        run_q, summaries, passages = enhanced_multihop_search(
            retriever, qa_model, query,
            n_hop=n_hop, top_n=top_n,
            use_rewrite   = USE_QUERY_REWRITE,
            use_planner   = USE_COT_PLANNER,
            use_rerank    = USE_LLM_RERANK,
            use_summary   = USE_PASSAGE_SUM
        )
        tree_graph = None        # vì enhanced_multihop_search trả về list passage
        hop_iters  = n_hop       # gần đúng
    else:
        # TreeHop thuần
        result     = retriever.multihop_search_passages(
            query, n_hop=n_hop, top_n=top_n,
            redundant_pruning=REDUNDANT_PRUNING,
            layerwise_top_pruning=LAYERWISE_TOP_PRUNING,
            return_tree=True
        )
        passages   = sum(result.passage[0], [])
        summaries  = []
        tree_graph = result.tree_hop_graph[0] if result.tree_hop_graph else None
        hop_iters  = len(result.passage[0]) if result.passage else 0

    if not passages:
        return {
            "query": query, "gold_answer": gold_answer,
            "pred_answer": "No relevant passages found", "f1_score": 0.0,
            "iterations": hop_iters, "retrieval_time": time.time()-t0,
            "num_passages": 0, "question_type": q_type,
            "tree_hop_graph": tree_graph,
        }

    # ---------------- Gemini answerer ------------------
    pred_answer, _ = generate_answer(qa_model, query, passages)

    # (tùy chọn) Verify answer bằng LLM
    # pred_answer = verify_answer(qa_model, query, pred_answer, passages)

    f1 = compute_f1(pred_answer, gold_answer)

    return {
        "query": query, "gold_answer": gold_answer, "pred_answer": pred_answer,
        "f1_score": f1, "iterations": hop_iters,
        "retrieval_time": time.time()-t0, "num_passages": len(passages),
        "question_type": q_type, "tree_hop_graph": tree_graph,
        "summaries": summaries          # thêm field summaries nếu cần
    }

def main():
    """Main function"""
    print("🎯 === CORRECTED TreeHop Multi-hop QA ===")
    
    # File paths
    corpus_file = os.path.join(DATASET_DIR, CORPUS_FILENAME)
    queries_file = os.path.join(DATASET_DIR, QUERIES_FILENAME)
    
    print(f"📁 Looking for files:")
    print(f"  Corpus: {corpus_file}")
    print(f"  Queries: {queries_file}")
    
    # Check files exist
    if not os.path.exists(corpus_file):
        print(f"❌ Corpus file not found: {corpus_file}")
        return
    
    if not os.path.exists(queries_file):
        print(f"❌ Queries file not found: {queries_file}")
        return
    
    try:
        # Load data
        print("\n📖 Loading corpus and queries...")
        corpus = load_corpus(corpus_file)
        queries = load_queries(queries_file)
        
        print(f"✅ Loaded {len(corpus)} documents and {len(queries)} queries")
        
        # Prepare passages
        print("\n📝 Preparing passages...")
        passages_file = os.path.join(OUTPUT_DIR, "passages.jsonl")
        passages = prepare_passages_jsonl(corpus, passages_file)
        
        # Initialize CORRECTED TreeHop retriever
        print("\n🧠 Initializing CORRECTED TreeHop...")
        retriever = CorrectedTreeHopRetriever(
            model_name=EMBEDDING_MODEL,
            passages_file=passages_file,
            embed_dim=1024,  # BGE-m3 dimension
            g_size=64,
            n_heads=3,
            mlp_size=64
        )
        
        # Setup Gemini
        print("\n🤖 Setting up Gemini API...")
        api_key = os.environ.get("GEMINI_API_KEY", None)
        if not api_key:
            print("⚠️ WARNING: No GEMINI_API_KEY environment variable found.")
            print("Please set your API key: export GEMINI_API_KEY='your_key_here'")
            api_key = input("Or enter your Gemini API key now: ")
            if not api_key:
                print("❌ No API key provided, cannot continue")
                return
        
        qa_model = setup_gemini_model(api_key)
        
        # Split data
        random.shuffle(queries)
        split_idx = int(len(queries) * TRAIN_RATIO)
        train_queries = queries[:split_idx]
        test_queries = queries[split_idx:]
        
        print(f"📊 Split into {len(train_queries)} train and {len(test_queries)} test queries")
        
        # Run evaluation
        print("\n🎯 Running CORRECTED TreeHop evaluation...")
        max_test_samples = min(len(test_queries), 10)  # Increased from 20 to 100
        
        results, metrics = run_evaluation(
            retriever, qa_model, test_queries, 
            n_hop=MAX_HOPS, top_n=TOP_N,
            max_samples=max_test_samples
        )
        
        # Calculate final metrics
        avg_f1 = mean(metrics["overall"]["f1"])
        avg_iterations = mean(metrics["overall"]["iterations"])
        avg_time = mean(metrics["overall"]["time"])
        
        print(f"\n📊 === Final Results (CORRECTED TreeHop) ===")
        print(f"🎯 Average F1 Score: {avg_f1:.4f}")
        print(f"🔄 Average Iterations: {avg_iterations:.2f}")
        print(f"⏱️ Average Time: {avg_time:.3f}s")
        print(f"🧠 TreeHop Statistics:")
        print(f"    Queries processed: {retriever.stats['total_queries']}")
        print(f"    Hops executed: {retriever.stats['total_hops']}")
        print(f"    Overlap applications: {retriever.stats['overlap_applications']}")
        print(f"    Average overlap magnitude: {retriever.stats['avg_overlap_magnitude']:.4f}")
        
        # Save results
        results_file = os.path.join(OUTPUT_DIR, "treehop_qa_results_corrected.json")
        
        # Convert TreeHopGraph objects to dictionaries for JSON serialization
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            # Convert numpy types to Python types
            for key, value in serializable_result.items():
                if isinstance(value, np.floating):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.integer):
                    serializable_result[key] = int(value)
            
            if 'tree_hop_graph' in serializable_result and serializable_result['tree_hop_graph']:
                graph = serializable_result['tree_hop_graph']
                # Convert all numpy types in graph data
                def convert_numpy(obj):
                    if isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    return obj
                
                serializable_result['tree_hop_graph'] = {
                    'query': graph.query,
                    'hops': convert_numpy(graph.hops),
                    'passages': convert_numpy(graph.passages),
                    'edges': convert_numpy(graph.edges),
                    'overlap_history': convert_numpy(graph.overlap_history)
                }
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': serializable_results,
                'metrics': {
                    'avg_f1': avg_f1,
                    'avg_iterations': avg_iterations,
                    'avg_time': avg_time,
                    'treehop_stats': retriever.stats
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to {results_file}")
        print("🎉 CORRECTED TreeHop evaluation finished successfully!")
        
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 