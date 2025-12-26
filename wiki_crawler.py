#!/usr/bin/env python3
"""
wiki_crawler.py

Wikipedia link-path finder with optional Gemini-guided neighbor scoring.

Features:
 - BFS / best-first (heuristic) / bidirectional search
 - Extract anchor text + one-sentence snippet per hop
 - LLM-guided neighbor re-ranking (Gemini 2.5 Flash) as a hybrid: cheap heuristic -> LLM re-rank
 - On-disk cache for LLM scores (.llm_score_cache.json)
 - Mindmap / pruned flowchart drawing with NetworkX + Matplotlib
 - Interactive mode (core options) when run with no CLI args; utility flags remain CLI-only
 - Explanation always printed; interactive runs ask if you want to save it

Configure Gemini:
  export GEMINI_API_KEY="your_key_here"
"""

import argparse
import os
import sys
import time
import json
import math
import logging
import requests
import difflib
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from heapq import heappush, heappop
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------------------
# Configuration & Constants
# ------------------------
API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "WikiCrawlerBot/1.0 (example@example.com) Python/requests"
GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_KEY_ENV = "GEMINI_API_KEY"
LLM_CACHE_PATH = ".llm_score_cache.json"

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ------------------------
# Utilities
# ------------------------
def load_llm_cache(path: str = LLM_CACHE_PATH) -> Dict[str, float]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load LLM cache: {e}")
    return {}

def save_llm_cache(cache: Dict[str, float], path: str = LLM_CACHE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save LLM cache: {e}")

def ask(prompt: str, default: Optional[str] = None, validator=None) -> str:
    p = f"{prompt} [{default}]: " if default is not None else f"{prompt}: "
    while True:
        try:
            val = input(p).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Interactive prompt aborted by user.")
        if val == "" and default is not None:
            val = default
        if validator:
            try:
                return validator(val)
            except Exception as e:
                print("Invalid value:", e)
                continue
        else:
            return val

def ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        try:
            val = input(f"{prompt} [{d}]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit("Interactive prompt aborted by user.")
        if val == "":
            return default
        if val in ("y", "yes"):
            return True
        if val in ("n", "no"):
            return False
        print("Please answer y or n.")

def _shorten_label(label: str, max_len: int = 28) -> str:
    if len(label) <= max_len:
        return label
    parts = label.split()
    if len(parts) == 1:
        return label[:max_len-3] + "..."
    first = " ".join(parts[:3])
    last = parts[-1]
    short = first + " … " + last
    if len(short) <= max_len:
        return short
    return label[:max_len-3] + "..."

class WikipediaAPIError(Exception):
    pass

# ------------------------
# Helper Classes
# ------------------------

class GeminiHelper:
    """Handles interactions with the Gemini API."""
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"} if api_key else {}

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _call_gemini(self, prompt: str, temperature: float, max_tokens: int) -> Optional[str]:
        if not self.is_configured():
            return None
        
        # --- MOCK MODE FOR TESTING ---
        if self.api_key == "TEST_KEY":
            logger.info("Using MOCK Gemini response (TEST_KEY detected).")
            # Heuristic: return a high score for "Target" or similar keywords in prompt
            if "numeric scorer" in prompt:
                # Mock scores
                return '{"scores": [{"title": "Mock Page", "score": 0.95}]}'
            else:
                # Mock explanation
                return (
                    "**[SIMULATED AI EXPLANATION]**\n\n"
                    "This is a mock response generated because `GEMINI_API_KEY` is set to `TEST_KEY`.\n\n"
                    "1. **Relevance**: The source page links to the target because they share a category.\n"
                    "2. **Heuristic**: The semantic similarity was calculated as high.\n"
                    "3. **Assessment**: A shorter path might exist via the 'United Kingdom' hub node."
                )
        # -----------------------------

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        try:
            resp = requests.post(GEMINI_MODEL_URL, headers=self.headers, json=body, timeout=120)
            resp.raise_for_status()
            j = resp.json()
            return self._extract_text_from_response(j) or resp.text
        except Exception as e:
            # Silent failure: log only if verbose, otherwise suppress to allow fallback
            logger.debug(f"Gemini call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.debug(f"Response content: {e.response.text}")
            return None

    def _extract_text_from_response(self, response_json: Dict[str, Any]) -> Optional[str]:
        """Robustly extracts text from Gemini response."""
        try:
            if "candidates" in response_json:
                return response_json["candidates"][0]["content"]["parts"][0]["text"]
            if "response" in response_json: # Handle potential nested structure
                 return response_json["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def score_neighbors(self, source_title: str, target_title: str, neighbors: List[str], 
                        snippets_map: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, float]:
        
        def build_prompt(batch: List[str]) -> str:
            lines = [
                "You are a strict numeric scorer. For each CANDIDATE neighbor, return a numeric score between 0.0 and 1.0 indicating how likely that neighbor will lead to the TARGET page within a small number of Wikipedia link hops. Return JSON ONLY in this exact format:",
                '{"scores":[{"title":"Neighbor Title","score":0.0}, ...]}',
                "",
                f"TARGET: {target_title}",
                f"SOURCE: {source_title}",
                "",
                "CANDIDATES:"
            ]
            for n in batch:
                lines.append(f"- title: {n}")
                if snippets_map and n in snippets_map:
                    anchor, snip = snippets_map[n]
                    if anchor: lines.append(f"  anchor: {json.dumps(anchor)}")
                    if snip: lines.append(f"  snippet: {json.dumps(snip[:240])}")
            lines.append("\nReturn only valid JSON.")
            return "\n".join(lines)

        prompt = build_prompt(neighbors)
        text = self._call_gemini(prompt, temperature=0.0, max_tokens=len(neighbors)*100 + 200)
        if not text: return {}
        return self._parse_json_scores(text)

    def _parse_json_scores(self, text: str) -> Dict[str, float]:
        scores = {}
        txt = text.strip()
        first = txt.find('{')
        last = txt.rfind('}')
        if first != -1 and last != -1 and last > first:
            json_str = txt[first:last+1]
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "scores" in parsed:
                    for e in parsed["scores"]:
                        t = e.get("title")
                        sc = e.get("score")
                        if isinstance(t, str) and isinstance(sc, (int, float)):
                            scores[t] = float(sc)
            except json.JSONDecodeError:
                pass
        
        # Fallback: regex parsing
        if not scores:
            import re
            for line in txt.splitlines():
                m = re.search(r"\"?([^\"]+)\"?\s*:\s*([01](?:\.\d+)?)", line)
                if m:
                     try:
                        scores[m.group(1)] = float(m.group(2))
                     except ValueError:
                         pass
        return scores

    def get_explanation(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        full_prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()
        return self._call_gemini(full_prompt, temperature=0.12, max_tokens=900)


class FlowchartVisualizer:
    """Handles the complex logic of drawing the crawl graph."""
    
    @staticmethod
    def draw(graph: nx.DiGraph, output_path: str, highlight_path: Optional[List[str]] = None, 
             max_nodes: int = 500, mode: str = "pruned", hide_nonpath_labels: bool = True):
        
        if not graph or len(graph.nodes) == 0:
            raise RuntimeError("No crawl graph recorded to draw.")
        
        G = FlowchartVisualizer._prune_graph(graph, highlight_path, max_nodes, mode)
        pos = FlowchartVisualizer._compute_layout(G, highlight_path, mode)
        FlowchartVisualizer._render_plot(G, pos, highlight_path, output_path, hide_nonpath_labels, mode)

    @staticmethod
    def _prune_graph(full_graph: nx.DiGraph, path: Optional[List[str]], max_nodes: int, mode: str) -> nx.DiGraph:
        G_full = full_graph.copy()
        path = path or []
        path_set = set(path)

        if mode == "path-only":
            if not path: raise RuntimeError("path-only requires a highlight_path")
            return G_full.subgraph(path_set).copy()
        
        nodes_keep = set(path_set)
        if mode in ("path-neighbors", "pruned", "mindmap"):
            if not path and mode == "path-neighbors": raise RuntimeError("path-neighbors requires a highlight_path")
            for n in path:
                nodes_keep.update(G_full.successors(n))
                nodes_keep.update(G_full.predecessors(n))
            
            # If still room, add high degree nodes for context
            if mode != "path-neighbors" and len(nodes_keep) < max_nodes:
                deg_sorted = sorted(G_full.nodes, key=lambda n: G_full.degree(n), reverse=True)
                for n in deg_sorted:
                    if len(nodes_keep) >= max_nodes: break
                    nodes_keep.add(n)
            
            return G_full.subgraph(nodes_keep).copy()
        
        return G_full # mode == "full"

    @staticmethod
    def _compute_layout(G: nx.DiGraph, path: Optional[List[str]], mode: str) -> Dict[Any, Tuple[float, float]]:
        if mode == "mindmap" and path:
            return FlowchartVisualizer._radial_layout(G, path[0])
        elif path and len(path) >= 2:
            return FlowchartVisualizer._linear_spine_layout(G, path)
        else:
            return nx.spring_layout(G, k=0.5, iterations=80)

    @staticmethod
    def _radial_layout(G: nx.DiGraph, start_node: str) -> Dict[Any, Tuple[float, float]]:
        try:
            lengths = nx.single_source_shortest_path_length(G.to_undirected(), start_node)
        except Exception:
            lengths = {n: 0 for n in G.nodes()}
        
        layers: Dict[int, List[str]] = {}
        max_layer = 0
        for n, d in lengths.items():
            layers.setdefault(d, []).append(n)
            max_layer = max(max_layer, d)
            
        pos = {start_node: (0.0, 0.0)}
        for layer in range(1, max_layer + 1):
            nodes = layers.get(layer, [])
            radius = 1.5 * layer
            for i, node in enumerate(nodes):
                angle = (i / max(1, len(nodes))) * 2 * math.pi
                pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
                
        # Handle disconnected components
        missing = [n for n in G.nodes if n not in pos]
        if missing:
            subpos = nx.spring_layout(G.subgraph(missing), k=0.6)
            offset = (max_layer + 2) * 1.5
            for n, p in subpos.items():
                pos[n] = (p[0] * 0.8 + offset, p[1] * 0.8)
        return pos

    @staticmethod
    def _linear_spine_layout(G: nx.DiGraph, path: List[str]) -> Dict[Any, Tuple[float, float]]:
        pos = {}
        for i, node in enumerate(path):
            pos[node] = (i * 1.5, 0.0)
            
        others = [n for n in G.nodes if n not in pos]
        radius = 1.5
        for i, node in enumerate(others):
            angle = (i / max(1, len(others))) * 2 * math.pi
            layer = 1 + ((i // 12) * 0.8)
            # spiraling around the center of mass roughly, or just around 0
            # Centering around the middle of the path is usually better
            center_x = (len(path) - 1) * 1.5 / 2
            pos[node] = (center_x + (len(path)*0.5) * math.cos(angle) * layer, radius * math.sin(angle) * layer)
        return pos

    @staticmethod
    def _render_plot(G: nx.DiGraph, pos: Dict, path: Optional[List[str]], output_path: str, hide_labels: bool, mode: str):
        path_set = set(path) if path else set()
        
        node_sizes = [700 if n in path_set else 200 for n in G.nodes]
        node_colors = []
        for n in G.nodes:
            if path and n == path[0]: node_colors.append("#2ca02c") # Start: Green
            elif path and n == path[-1]: node_colors.append("#1f77b4") # Target: Blue
            elif n in path_set: node_colors.append("#d62728") # Path: Red
            else: node_colors.append("#999999") # Other: Grey

        node_labels = {n: _shorten_label(n) for n in G.nodes if not hide_labels or n in path_set}

        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.92)
        nx.draw_networkx_edges(G, pos, arrowsize=12, arrowstyle='->', width=1, alpha=0.7)
        
        if path and len(path) >= 2:
            path_edges = list(zip(path[:-1], path[1:]))
            valid_edges = [e for e in path_edges if G.has_edge(*e)]
            if valid_edges:
                nx.draw_networkx_edges(G, pos, edgelist=valid_edges, arrowsize=14, arrowstyle='->', width=3, alpha=0.95)

        if node_labels:
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

        # Legend
        legend_items = []
        if path:
            legend_items.append(mpatches.Patch(color="#2ca02c", label="Start"))
            legend_items.append(mpatches.Patch(color="#1f77b4", label="Target"))
            legend_items.append(mpatches.Patch(color="#d62728", label="Path nodes"))
        legend_items.append(mpatches.Patch(color="#999999", label="Other nodes"))
        
        plt.legend(handles=legend_items, loc='upper right', fontsize=9)
        plt.title(f"Wikipedia crawl graph — mode: {mode}")
        plt.axis('off')
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        logger.info(f"Flowchart saved to: {output_path}")

# ------------------------
# Main Crawler Class
# ------------------------

class WikiCrawler:
    def __init__(self, user_agent: Optional[str] = None, sleep_between_requests: float = 0.12, verbose: bool = False):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent or DEFAULT_USER_AGENT})
        self.sleep = float(sleep_between_requests)
        
        # Logging setup
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Caches
        self.link_cache: Dict[str, List[str]] = {}
        self.linkshere_cache: Dict[str, List[str]] = {}
        self.title_cache: Dict[str, str] = {}
        self.llm_cache = load_llm_cache()

        # State
        self.crawl_graph = nx.DiGraph()
        self.decision_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Helpers
        self.gemini = GeminiHelper(os.getenv(GEMINI_KEY_ENV))

    def _api_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("format", "json")
        r = self.session.get(API_ENDPOINT, params=params, timeout=30)
        if r.status_code != 200:
            raise WikipediaAPIError(f"Bad status {r.status_code}: {r.text[:200]}")
        return r.json()

    def resolve_title(self, title: str) -> Optional[str]:
        if not title: return None
        if title in self.title_cache: return self.title_cache[title]
        
        params = {"action": "query", "titles": title, "redirects": 1, "formatversion": 2, "prop": "info"}
        j = self._api_get(params)
        pages = j.get("query", {}).get("pages", [])
        
        if not pages or "missing" in pages[0]:
            return None
            
        normalized = pages[0].get("title")
        self.title_cache[title] = normalized
        self.title_cache[normalized] = normalized
        return normalized

    def random_page_title(self, namespace: int = 0) -> str:
        params = {"action": "query", "list": "random", "rnnamespace": namespace, "rnlimit": 1, "formatversion": 2}
        j = self._api_get(params)
        entries = j.get("query", {}).get("random", [])
        if not entries: raise WikipediaAPIError("Could not fetch random page")
        return entries[0]["title"]

    def get_links(self, title: str) -> List[str]:
        normalized = self.resolve_title(title)
        if not normalized: return []
        if normalized in self.link_cache: return self.link_cache[normalized]
        
        links = []
        params = {"action": "query", "titles": normalized, "prop": "links", "pllimit": "max", "formatversion": 2}
        
        while True:
            j = self._api_get(params)
            time.sleep(self.sleep)
            pages = j.get("query", {}).get("pages", [])
            if pages:
                links.extend([l.get("title") for l in pages[0].get("links", []) if l.get("ns") == 0])
            
            cont = j.get("continue")
            if not cont or "plcontinue" not in cont:
                break
            params["plcontinue"] = cont["plcontinue"]
            
        self.link_cache[normalized] = links
        logger.debug(f"Fetched {len(links)} outgoing links for '{normalized}'")
        return links

    def get_linkshere(self, title: str) -> List[str]:
        normalized = self.resolve_title(title)
        if not normalized: return []
        if normalized in self.linkshere_cache: return self.linkshere_cache[normalized]
        
        incoming = []
        params = {"action": "query", "titles": normalized, "prop": "linkshere", "lhlimit": "max", "lhnamespace": 0, "formatversion": 2}
        
        while True:
            j = self._api_get(params)
            time.sleep(self.sleep)
            pages = j.get("query", {}).get("pages", [])
            if pages:
                incoming.extend([l.get("title") for l in pages[0].get("linkshere", [])])
                
            cont = j.get("continue")
            if not cont or "lhcontinue" not in cont:
                break
            params["lhcontinue"] = cont["lhcontinue"]
            
        self.linkshere_cache[normalized] = incoming
        logger.debug(f"Fetched {len(incoming)} incoming links for '{normalized}'")
        return incoming

    def search_title(self, query: str, limit: int = 1) -> Optional[str]:
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "formatversion": 2}
        j = self._api_get(params)
        hits = j.get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None

    # ------------------------
    # Search Strategies
    # ------------------------

    def find_path_bfs(self, start_title: str, target_title: str, max_depth: int = 6, max_visited: int = 100000) -> Optional[List[str]]:
        start, target = self.resolve_title(start_title), self.resolve_title(target_title)
        if not start or not target: raise ValueError("Start or target missing/resolution failed")
        
        self._reset_graph(start, target)
        if start == target: return [start]

        q = deque([(start, [start], 0)])
        visited = {start}
        visited_count = 0

        while q:
            current, path, depth = q.popleft()
            visited_count += 1
            if visited_count > max_visited: raise RuntimeError("Visited cap exceeded")
            if depth >= max_depth: continue

            try:
                neighbors = sorted(self.get_links(current))
            except Exception as e:
                logger.error(f"get_links failed for {current}: {e}")
                neighbors = []

            self._update_graph(current, neighbors)

            if target in neighbors:
                self.decision_info[(current, target)] = {'method': 'bfs', 'depth': depth + 1, 'note': 'direct neighbor'}
                return path + [target]

            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    q.append((n, path + [n], depth + 1))
                    self.decision_info[(current, n)] = {'method': 'bfs', 'depth': depth + 1, 'note': f'enqueued from {current}'}
        return None

    def find_path_bidi(self, start_title: str, target_title: str, max_depth: int = 6, max_visited: int = 100000) -> Optional[List[str]]:
        start, target = self.resolve_title(start_title), self.resolve_title(target_title)
        if not start or not target: raise ValueError("Start or target missing/resolution failed")
        
        self._reset_graph(start, target)
        if start == target: return [start]

        parent_fwd = {start: None}
        parent_bwd = {target: None}
        q_fwd = deque([(start, 0)])
        q_bwd = deque([(target, 0)])
        visited_fwd = {start}
        visited_bwd = {target}
        visited_count = 0

        while q_fwd and q_bwd:
            # Forward Expansion
            if len(q_fwd) <= len(q_bwd):
                current, depth = q_fwd.popleft()
                visited_count += 1
                if visited_count > max_visited: raise RuntimeError("Visited cap exceeded")
                if depth >= max_depth: continue

                try:
                    neighbors = sorted(self.get_links(current))
                except Exception as e:
                    logger.error(f"get_links failed for {current}: {e}")
                    neighbors = []

                self._update_graph(current, neighbors)

                inter = set(neighbors) & visited_bwd
                if inter:
                    meet = sorted(list(inter))[0]
                    parent_fwd[meet] = current
                    self.decision_info[(current, meet)] = {'method': 'bidi_fwd', 'depth': depth + 1, 'note': 'meeting edge (fwd)'}
                    return self._reconstruct_bidi_path(parent_fwd, parent_bwd, meet, start, target)

                for n in neighbors:
                    if n not in visited_fwd:
                        visited_fwd.add(n)
                        parent_fwd[n] = current
                        q_fwd.append((n, depth + 1))
                        self.decision_info[(current, n)] = {'method': 'bidi_fwd', 'depth': depth + 1, 'note': 'forward enqueued'}
            
            # Backward Expansion
            else:
                current, depth = q_bwd.popleft()
                visited_count += 1
                if visited_count > max_visited: raise RuntimeError("Visited cap exceeded")
                if depth >= max_depth: continue

                try:
                    incoming = sorted(self.get_linkshere(current))
                except Exception as e:
                    logger.error(f"get_linkshere failed for {current}: {e}")
                    incoming = []

                # Add edges REVERSE direction for graph: incoming -> current
                for n in incoming:
                    if not self.crawl_graph.has_node(n): self.crawl_graph.add_node(n)
                    if not self.crawl_graph.has_edge(n, current): self.crawl_graph.add_edge(n, current)

                inter = set(incoming) & visited_fwd
                if inter:
                    meet = sorted(list(inter))[0]
                    parent_bwd[meet] = current
                    self.decision_info[(meet, current)] = {'method': 'bidi_bwd', 'depth': depth + 1, 'note': 'meeting edge (bwd)'}
                    return self._reconstruct_bidi_path(parent_fwd, parent_bwd, meet, start, target)

                for n in incoming:
                    if n not in visited_bwd:
                        visited_bwd.add(n)
                        parent_bwd[n] = current
                        q_bwd.append((n, depth + 1))
                        self.decision_info[(n, current)] = {'method': 'bidi_bwd', 'depth': depth + 1, 'note': 'backward enqueued'}
        return None

    def find_path_best_first(self, start_title: str, target_title: str, max_depth: int = 6,
                             max_visited: int = 50000, max_branch: int = 50,
                             use_llm: bool = True, candidate_pool_size: int = 30,
                             combine_alpha: float = 0.6, enqueue_k: int = 20) -> Optional[List[str]]:
        start, target = self.resolve_title(start_title), self.resolve_title(target_title)
        if not start or not target: raise ValueError("Start or target missing/resolution failed")

        self._reset_graph(start, target)
        if start == target: return [start]

        uid = 0
        # Heap items: (neg_score, depth, unique_id, current_node, path)
        heap: List[Tuple[float, int, int, str, List[str]]] = []
        start_score = self._title_score(start, target)
        heappush(heap, (-start_score, 0, uid, start, [start]))
        visited = {start}
        visited_count = 0

        while heap:
            _, depth, _, current, path = heappop(heap)
            visited_count += 1
            if visited_count > max_visited: raise RuntimeError("Visited cap exceeded")
            if depth >= max_depth: continue

            try:
                neighbors = self.get_links(current)
            except Exception as e:
                logger.error(f"get_links failed for {current}: {e}")
                neighbors = []

            self._update_graph(current, neighbors)

            if target in neighbors:
                self.decision_info[(current, target)] = {'method': 'best+llm' if use_llm else 'best', 'depth': depth + 1, 'note': 'direct neighbor - target found'}
                return path + [target]

            cand_neighbors = [n for n in neighbors if n not in visited]
            if not cand_neighbors: continue

            # 1. Cheap Heuristic Scoring
            cheap_scores = [(self._title_score(n, target), n) for n in cand_neighbors]
            cheap_scores.sort(reverse=True)
            top_m = [n for _, n in cheap_scores[:candidate_pool_size]]

            # 2. LLM Scoring (Optional)
            llm_scores = {}
            if use_llm and self.gemini.is_configured():
                snippets_map = self._collect_snippets(current, top_m)
                llm_scores = self.gemini.score_neighbors(current, target, top_m, snippets_map)

            # 3. Combine & Enqueue
            combined_list = []
            for n in top_m:
                cheap = self._title_score(n, target)
                lscore = llm_scores.get(n, 0.5)
                combined = combine_alpha * cheap + (1.0 - combine_alpha) * (lscore if lscore is not None else 0.5)
                combined_list.append((combined, n, cheap, lscore))

            combined_list.sort(reverse=True, key=lambda x: x[0])
            for combined, n, cheap, lscore in combined_list[:min(enqueue_k, len(combined_list))]:
                if n not in visited:
                    visited.add(n)
                    uid += 1
                    heappush(heap, (-combined, depth + 1, uid, n, path + [n]))
                    self.decision_info[(current, n)] = {
                        'method': 'best+llm' if use_llm else 'best',
                        'depth': depth + 1,
                        'score': combined,
                        'cheap_score': cheap,
                        'llm_score': lscore if lscore is not None else 0.5
                    }
        return None

    # ------------------------
    # Internal Helpers
    # ------------------------

    def _reset_graph(self, start: str, target: Optional[str] = None):
        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)
        if target:
            self.crawl_graph.add_node(target)
        self.decision_info = {}

    def _update_graph(self, source: str, neighbors: List[str]):
        for n in neighbors:
            if not self.crawl_graph.has_node(n): self.crawl_graph.add_node(n)
            if not self.crawl_graph.has_edge(source, n): self.crawl_graph.add_edge(source, n)

    def _reconstruct_bidi_path(self, parent_fwd, parent_bwd, meet, start, target):
        # Forward part
        left = []
        node = meet
        while node:
            left.append(node)
            node = parent_fwd.get(node)
        left = list(reversed(left))
        
        # Backward part
        right = []
        cur = meet
        while cur and cur != target:
            nxt = parent_bwd.get(cur)
            if not nxt: break
            right.append(nxt)
            cur = nxt
            
        full = left + right
        if full[0] != start: full.insert(0, start)
        if full[-1] != target: full.append(target)
        return full

    def _title_score(self, candidate: str, target: str) -> float:
        ratio = difflib.SequenceMatcher(None, candidate.lower(), target.lower()).ratio()
        target_tokens = [t for t in target.lower().split() if len(t) > 2]
        bonus = 0.25 * sum(1 for t in target_tokens if t in candidate.lower())
        return ratio + bonus

    def _collect_snippets(self, current: str, neighbors: List[str]) -> Dict[str, Tuple[str, str]]:
        snippets = {}
        for n in neighbors:
            try:
                sn = self.extract_anchor_snippet(current, n)
                snippets[n] = (sn.get("anchor_text") or "", sn.get("source_snippet") or "")
            except Exception:
                snippets[n] = ("", "")
        return snippets

    def extract_anchor_snippet(self, source_title: str, target_title: str, max_chars: int = 250) -> Dict[str, Optional[str]]:
        src, tgt = self.resolve_title(source_title), self.resolve_title(target_title)
        if not src or not tgt: return {"anchor_text": None, "source_snippet": None}
        
        try:
            params = {"action": "parse", "page": src, "prop": "text", "formatversion": 2}
            j = self._api_get(params)
            time.sleep(self.sleep)
            html = j.get("parse", {}).get("text", "")
        except Exception:
            return {"anchor_text": None, "source_snippet": None}
            
        if not html: return {"anchor_text": None, "source_snippet": None}
        
        soup = BeautifulSoup(html, "html.parser")
        tgt_frag = "/wiki/" + tgt.replace(" ", "_")
        
        # Find anchor tag
        a_tag = (soup.find("a", href=lambda h: h and h.split("#")[0].split("?")[0] == tgt_frag) or
                 soup.find("a", href=lambda h: h and tgt_frag in h) or
                 soup.find("a", title=tgt))
                 
        if not a_tag: return {"anchor_text": None, "source_snippet": None}
        
        anchor_text = a_tag.get_text(strip=True) or None
        
        # Find parent container for snippet
        parent = a_tag.parent
        for _ in range(4):
            if not parent or parent.name in ("p", "li", "div", "td", "section"): break
            parent = parent.parent
            
        snippet = None
        if parent:
            text = parent.get_text(" ", strip=True)
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                if (anchor_text and anchor_text in sent) or (tgt in sent):
                    snippet = sent.strip()
                    break
            if not snippet:
                snippet = (text.strip()[:max_chars] + "...") if len(text) > max_chars else text.strip()
                
        return {"anchor_text": anchor_text, "source_snippet": snippet}

    # ------------------------
    # Explanation
    # ------------------------

    def produce_rich_explanation(self, path: List[str], strategy: str, visited_count: int, elapsed_seconds: float) -> str:
        summary = {
            "strategy": strategy, "start": path[0] if path else None,
            "target": path[-1] if path else None, "hops": len(path)-1 if path else 0,
            "visited_count": visited_count, "elapsed_seconds": elapsed_seconds
        }
        
        steps = []
        for i in range(len(path)-1):
            A, B = path[i], path[i+1]
            meta = self.decision_info.get((A, B)) or self.decision_info.get((B, A)) or {}
            sn = self.extract_anchor_snippet(A, B)
            step = {"i": i+1, "A": A, "B": B, **sn, **{k: meta.get(k) for k in ("method", "depth", "cheap_score", "llm_score")}} # Removed 'score' as it's redundant with combined
            steps.append(step)

        # Attempt to get AI explanation
        if self.gemini.is_configured():
            prompt = self._build_explanation_prompt(summary, steps)
            explanation = self.gemini.get_explanation(prompt["system"], prompt["user"])
            if explanation: return explanation

        # Fallback: Integrated explanation (always used if API key is missing or call fails)
        lines = ["=== Path Explanation ==="]
        lines.append(f"The crawler found a path of {len(path)-1} hops using the '{strategy}' strategy.")
        lines.append(f"It analyzed {visited_count} pages in {elapsed_seconds:.2f} seconds.\n")

        # Suppress "Tip" if this function was called due to an error, or keep it subtle
        if not self.gemini.is_configured():
             pass # No tip needed if we just want it to work without API key
        
        for s in steps:
            lines.append(f"Step {s['i']}: {s['A']} -> {s['B']}")
            
            # Algorithmic context
            method = s.get('method')
            note = s.get('note', '')
            algo_desc = "Link found via crawler."
            if 'bidi_fwd' in str(method):
                algo_desc = "Found while searching FORWARD from the start."
            elif 'bidi_bwd' in str(method):
                algo_desc = "Found while searching BACKWARD from the target."
            elif 'bfs' in str(method):
                algo_desc = "Found during standard breadth-first search."
            elif 'best' in str(method):
                llm_score = s.get('llm_score')
                cheap_score = s.get('cheap_score')
                score = llm_score if llm_score is not None else cheap_score
                if score is not None:
                    algo_desc = f"Selected as a high-potential link (score: {score:.2f})."
                else:
                    algo_desc = "Selected as a high-potential link."
            
            if "meeting edge" in note:
                algo_desc += " ** This was the meeting point of the two searches. **"

            lines.append(f"  [Logic]   {algo_desc}")
            
            # Semantic context
            snippet = s.get('source_snippet')
            if snippet:
                lines.append(f"  [Context] \"...{snippet}...\"")
            else:
                lines.append(f"  [Context] (Link found, but no surrounding text snippet extracted)")
            lines.append("")
            
        return "\n".join(lines)

    def explain_path(self, path: Optional[List[str]]) -> str:
        if not path or len(path) < 2: return "No path or trivial path."
        
        lines = ["Explanation of the path found (step-by-step):", f"Total hops: {len(path)-1}", ""]
        for i in range(len(path)-1):
            a, b = path[i], path[i+1]
            info = self.decision_info.get((a, b)) or self.decision_info.get((b, a))
            step_no = i + 1
            if info:
                parts = [f"{step_no}. {a} -> {b}"]
                for k, v in info.items():
                    if v is not None: parts.append(f"   • {k}: {v}")
                lines.append("\n".join(parts))
            else:
                lines.append(f"{step_no}. {a} -> {b}\n   • reason: link observed (no metadata).")
        return "\n".join(lines)

    def _build_explanation_prompt(self, summary: Dict, steps: List[Dict]) -> Dict[str, str]:
        system = ("You are a technical explainer. Given a crawler trace, produce a clear step-by-step explanation. "
                  "Explain why each link was chosen, referencing snippets. Comment on heuristics/LLM role.")
        lines = ["CRAWL SUMMARY", *[f"- {k}: {v}" for k, v in summary.items()], "", "PATH & DECISIONS"]
        for s in steps:
            lines.append(f"STEP {s['i']}: {s['A']} -> {s['B']}")
            anchor = (s.get('anchor_text') or '')[:120]
            snippet = (s.get('source_snippet') or '')[:220]
            lines.append(f' - anchor_text: "{anchor}"')
            lines.append(f' - source_snippet: "{snippet}"')
            for k in ["method", "cheap_score", "llm_score"]:
                if s.get(k) is not None: lines.append(f" - {k}: {s.get(k)}")
            lines.append("")
        return {"system": system, "user": "\n".join(lines)}

    # ------------------------
    # Flowchart
    # ------------------------
    def draw_flowchart(self, *args, **kwargs):
        """Proxy to FlowchartVisualizer for backward compatibility."""
        FlowchartVisualizer.draw(self.crawl_graph, *args, **kwargs)

# ------------------------
# Main Entry
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Find link path between two Wikipedia pages.")
    parser.add_argument("--start", help="Start page title.")
    parser.add_argument("--target", help="Target page title.")
    parser.add_argument("--random-start", action="store_true", help="Pick random start.")
    parser.add_argument("--strategy", choices=["bfs", "best", "bidi"], default="bidi", help="Search strategy.")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-visited", type=int, default=50000)
    parser.add_argument("--max-branch", type=int, default=50)
    parser.add_argument("--user-agent")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--flowchart", help="Path to save flowchart PNG.")
    parser.add_argument("--flowchart-mode", choices=["path-only","path-neighbors","pruned","mindmap","full"], default="mindmap")
    parser.add_argument("--hide-nonpath-labels", action="store_true")
    parser.add_argument("--explain-file", help="Path to save explanation.")
    parser.add_argument("--use-llm-hopping", action="store_true")
    parser.add_argument("--llm-candidates", type=int, default=30)
    parser.add_argument("--llm-alpha", type=float, default=0.6)
    parser.add_argument("--llm-enqueue-k", type=int, default=20)

    # Interactive setup if no args
    if len(sys.argv) == 1:
        print("=== WikiCrawler interactive setup ===")
        start_title = ask("Start article (or 'random')", default="random")
        random_start = start_title.lower() in ("random", "r")
        if random_start: start_title = None
        target_title = ask("Target article", validator=lambda s: s if s else ValueError("Required"))
        strategy = ask("Strategy (bfs|best|bidi)", default="bidi", validator=lambda s: s if s in ["bfs","best","bidi"] else ValueError("Invalid"))
        
        want_flowchart = ask_yes_no("Save flowchart?", default=False)
        flowchart = ask("Flowchart path", default="./flowchart.png") if want_flowchart else None
        flowchart_mode = ask("Mode", default="mindmap") if want_flowchart else "mindmap"
        hide_nonpath_labels = ask_yes_no("Hide clutter labels?", default=True) if want_flowchart else True
        
        explain_file = None
        if ask_yes_no("Save explanation?", default=True):
            explain_file = ask("Explanation path", default="./explanation.txt")
        
        # Defaults for others
        args = argparse.Namespace(
            start=start_title, random_start=random_start, target=target_title, strategy=strategy,
            flowchart=flowchart, flowchart_mode=flowchart_mode, hide_nonpath_labels=hide_nonpath_labels,
            explain_file=explain_file, max_depth=6, max_visited=50000, max_branch=50,
            user_agent=None, sleep=0.12, verbose=False, use_llm_hopping=False, 
            llm_candidates=30, llm_alpha=0.6, llm_enqueue_k=20
        )
    else:
        args = parser.parse_args()

    crawler = WikiCrawler(user_agent=args.user_agent, sleep_between_requests=args.sleep, verbose=args.verbose)

    if args.random_start:
        args.start = crawler.random_page_title()
        print(f"Random start: {args.start}")
    elif not args.start:
        print("Error: Start page required.")
        return

    if not args.target:
        print("Error: Target page required.")
        return

    try:
        print(f" resolving {args.start} -> {args.target}...")
        path = None
        t0 = time.time()
        
        if args.strategy == "bfs":
            path = crawler.find_path_bfs(args.start, args.target, args.max_depth, args.max_visited)
        elif args.strategy == "best":
            path = crawler.find_path_best_first(args.start, args.target, args.max_depth, args.max_visited,
                                                args.max_branch, args.use_llm_hopping, args.llm_candidates,
                                                args.llm_alpha, args.llm_enqueue_k)
        else:
            path = crawler.find_path_bidi(args.start, args.target, args.max_depth, args.max_visited)
            
        elapsed = time.time() - t0
        
        if path:
            print(f"\n=== PATH FOUND ({len(path)-1} hops) ===")
            for i, t in enumerate(path): print(f"{i}. {t}")
        else:
            print(f"\nNo path found (depth {args.max_depth})")

        visited_count = len(crawler.crawl_graph.nodes)
        rich_expl = crawler.produce_rich_explanation(path or [], args.strategy, visited_count, elapsed)
        
        print("\n--- Explanation ---\n")
        print(rich_expl)

        if args.explain_file:
            with open(args.explain_file, "w", encoding="utf-8") as f:
                f.write(rich_expl)
            print(f"Saved explanation to {args.explain_file}")

        if args.flowchart and path:
            crawler.draw_flowchart(args.flowchart, highlight_path=path, mode=args.flowchart_mode, 
                                   hide_nonpath_labels=args.hide_nonpath_labels)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
