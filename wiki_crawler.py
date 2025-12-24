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
import requests
from collections import deque
from heapq import heappush, heappop
import difflib
import networkx as nx
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple, Any

API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
DEFAULT_USER_AGENT = "WikiCrawlerBot/1.0 (example@example.com) Python/requests"

# Gemini endpoint for generateContent (REST)
GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_KEY_ENV = "AIzaSyDXaq56qXm7bZw704xMeg2pa4KKmQ-nnKM"

# On-disk LLM score cache file
LLM_CACHE_PATH = ".llm_score_cache.json"

# ------------------------
# Utilities
# ------------------------
def load_llm_cache(path: str = LLM_CACHE_PATH) -> Dict[str, float]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_llm_cache(cache: Dict[str, float], path: str = LLM_CACHE_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def ask(prompt: str, default: Optional[str] = None, validator=None) -> str:
    if default is None:
        p = f"{prompt}: "
    else:
        p = f"{prompt} [{default}]: "
    while True:
        try:
            val = input(p).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit("Interactive prompt aborted by user.")
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
            raise SystemExit("Interactive prompt aborted by user.")
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

def parse_int(s: str, name: str) -> int:
    try:
        return int(s)
    except:
        raise ValueError(f"{name} must be an integer")

# ------------------------
# WikiCrawler
# ------------------------
class WikipediaAPIError(Exception):
    pass

class WikiCrawler:
    def __init__(self, user_agent: Optional[str] = None, sleep_between_requests: float = 0.12, verbose: bool = False):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent or DEFAULT_USER_AGENT})
        self.sleep = float(sleep_between_requests)
        self.verbose = verbose

        # caches in-memory
        self.link_cache: Dict[str, List[str]] = {}
        self.linkshere_cache: Dict[str, List[str]] = {}
        self.title_cache: Dict[str, str] = {}

        # graph / decisions
        self.crawl_graph = nx.DiGraph()
        self.decision_info: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # LLM score cache (disk-backed)
        self.llm_cache = load_llm_cache()

    def log(self, *args):
        if self.verbose:
            print("[verbose]", *args)

    def _api_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("format", "json")
        r = self.session.get(API_ENDPOINT, params=params, timeout=30)
        if r.status_code != 200:
            raise WikipediaAPIError(f"Bad status {r.status_code}: {r.text[:200]}")
        return r.json()

    def resolve_title(self, title: str) -> Optional[str]:
        if not title:
            return None
        if title in self.title_cache:
            return self.title_cache[title]
        params = {"action": "query", "titles": title, "redirects": 1, "formatversion": 2, "prop": "info"}
        j = self._api_get(params)
        pages = j.get("query", {}).get("pages", [])
        if not pages:
            return None
        page = pages[0]
        if "missing" in page:
            return None
        normalized = page.get("title")
        self.title_cache[title] = normalized
        self.title_cache[normalized] = normalized
        return normalized

    def random_page_title(self, namespace: int = 0) -> str:
        params = {"action": "query", "list": "random", "rnnamespace": namespace, "rnlimit": 1, "formatversion": 2}
        j = self._api_get(params)
        entries = j.get("query", {}).get("random", [])
        if not entries:
            raise WikipediaAPIError("Could not fetch random page")
        return entries[0]["title"]

    def get_links(self, title: str) -> List[str]:
        normalized = self.resolve_title(title)
        if not normalized:
            return []
        if normalized in self.link_cache:
            return self.link_cache[normalized]
        links = []
        params = {"action": "query", "titles": normalized, "prop": "links", "pllimit": "max", "formatversion": 2}
        while True:
            j = self._api_get(params)
            time.sleep(self.sleep)
            pages = j.get("query", {}).get("pages", [])
            if pages:
                page = pages[0]
                for l in page.get("links", []):
                    if l.get("ns") == 0:
                        links.append(l.get("title"))
            cont = j.get("continue")
            if cont and cont.get("plcontinue"):
                params["plcontinue"] = cont["plcontinue"]
                continue
            break
        self.link_cache[normalized] = links
        self.log(f"Fetched {len(links)} outgoing links for '{normalized}'")
        return links

    def get_linkshere(self, title: str) -> List[str]:
        normalized = self.resolve_title(title)
        if not normalized:
            return []
        if normalized in self.linkshere_cache:
            return self.linkshere_cache[normalized]
        incoming = []
        params = {"action": "query", "titles": normalized, "prop": "linkshere", "lhlimit": "max", "lhnamespace": 0, "formatversion": 2}
        while True:
            j = self._api_get(params)
            time.sleep(self.sleep)
            pages = j.get("query", {}).get("pages", [])
            if pages:
                page = pages[0]
                for l in page.get("linkshere", []):
                    incoming.append(l.get("title"))
            cont = j.get("continue")
            if cont and cont.get("lhcontinue"):
                params["lhcontinue"] = cont["lhcontinue"]
                continue
            break
        self.linkshere_cache[normalized] = incoming
        self.log(f"Fetched {len(incoming)} incoming links for '{normalized}'")
        return incoming

    def search_title(self, query: str, limit: int = 1) -> Optional[str]:
        params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "formatversion": 2}
        j = self._api_get(params)
        hits = j.get("query", {}).get("search", [])
        if not hits:
            return None
        return hits[0]["title"]

    # ------------------------
    # Search algorithms
    # ------------------------
    def find_path_bfs(self, start_title: str, target_title: str, max_depth: int = 6, max_visited: int = 100000):
        start = self.resolve_title(start_title)
        target = self.resolve_title(target_title)
        if not start or not target:
            raise ValueError("Start or target missing/resolution failed")

        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)
        self.decision_info = {}

        if start == target:
            return [start]

        q = deque()
        q.append((start, [start], 0))
        visited = {start}
        visited_count = 0

        while q:
            current, path, depth = q.popleft()
            visited_count += 1
            if visited_count > max_visited:
                raise RuntimeError("Visited cap exceeded")
            if depth >= max_depth:
                continue
            try:
                neighbors = set(self.get_links(current))
            except Exception as e:
                self.log(f"get_links failed for {current}: {e}")
                neighbors = set()

            for n in neighbors:
                if not self.crawl_graph.has_node(n):
                    self.crawl_graph.add_node(n)
                if not self.crawl_graph.has_edge(current, n):
                    self.crawl_graph.add_edge(current, n)

            if target in neighbors:
                self.decision_info[(current, target)] = {'method': 'bfs', 'depth': depth + 1, 'note': 'direct neighbor'}
                return path + [target]

            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    q.append((n, path + [n], depth + 1))
                    self.decision_info[(current, n)] = {'method': 'bfs', 'depth': depth + 1, 'note': f'enqueued from {current}'}
        return None

    def _title_score(self, candidate_title: str, target_title: str) -> float:
        ratio = difflib.SequenceMatcher(None, candidate_title.lower(), target_title.lower()).ratio()
        target_tokens = [t for t in target_title.lower().split() if len(t) > 2]
        bonus = 0.0
        cand = candidate_title.lower()
        for t in target_tokens:
            if t in cand:
                bonus += 0.25
        return ratio + bonus

    # ------------------------
    # LLM-assisted best-first search
    # ------------------------
    def llm_score_neighbors_batch(self,
                                  source_title: str,
                                  target_title: str,
                                  neighbors: List[str],
                                  snippets_map: Optional[Dict[str, Tuple[str, str]]] = None,
                                  batch_size: int = 8,
                                  temperature: float = 0.0,
                                  max_output_tokens: int = 200) -> Dict[str, float]:
        """
        Score neighbors using Gemini. Returns mapping neighbor -> score in [0.0, 1.0].
        Uses on-disk cache to avoid repeated calls.
        """
        # prepare env key
        api_key = os.getenv(GEMINI_KEY_ENV)
        if not api_key:
            # LLM not configured; return neutral (0.5) for each
            return {n: 0.5 for n in neighbors}

        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

        # prepare function to build prompt
        def build_prompt(batch: List[str]) -> str:
            lines = []
            lines.append("You are a strict numeric scorer. For each CANDIDATE neighbor, return a numeric score between 0.0 and 1.0 indicating how likely that neighbor will lead to the TARGET page within a small number of Wikipedia link hops. Do NOT fabricate facts. Use only the provided title and snippet. Return JSON ONLY in this exact format:")
            lines.append('{"scores":[{"title":"Neighbor Title","score":0.0}, ...]}')
            lines.append("")
            lines.append(f"TARGET: {target_title}")
            lines.append(f"SOURCE: {source_title}")
            lines.append("")
            lines.append("CANDIDATES:")
            for n in batch:
                lines.append(f"- title: {n}")
                if snippets_map and n in snippets_map:
                    anchor, snip = snippets_map[n]
                    if anchor:
                        lines.append(f"  anchor: \"{anchor}\"")
                    if snip:
                        snip_short = snip if len(snip) <= 240 else snip[:240] + "..."
                        lines.append(f"  snippet: \"{snip_short}\"")
            lines.append("")
            lines.append("Return only valid JSON as described above and nothing else.")
            return "\n".join(lines)

        scores: Dict[str, float] = {}
        to_query = [n for n in neighbors if self._llm_cache_key(source_title, n) not in self.llm_cache]
        # fill from cache where possible
        for n in neighbors:
            k = self._llm_cache_key(source_title, n)
            if k in self.llm_cache:
                scores[n] = float(self.llm_cache[k])

        # batches
        for i in range(0, len(to_query), batch_size):
            batch = to_query[i:i+batch_size]
            prompt = build_prompt(batch)
            body = {
                "contents": [{"parts": [{"text": prompt}]}],
                "temperature": float(temperature),
                "maxOutputTokens": int(max_output_tokens)
            }
            try:
                resp = requests.post(GEMINI_MODEL_URL, headers=headers, json=body, timeout=120)
                resp.raise_for_status()
                j = resp.json()
                candidate_text = None
                # robust extraction
                if isinstance(j, dict):
                    if "candidates" in j and isinstance(j["candidates"], list) and len(j["candidates"]) > 0:
                        cand = j["candidates"][0]
                        cont = cand.get("content", {})
                        if isinstance(cont, dict) and isinstance(cont.get("parts"), list) and len(cont["parts"]) > 0:
                            candidate_text = cont["parts"][0].get("text")
                    elif "response" in j and isinstance(j["response"], dict):
                        resp_cands = j["response"].get("candidates")
                        if resp_cands and isinstance(resp_cands, list):
                            c = resp_cands[0]
                            cont = c.get("content", {})
                            if isinstance(cont, dict) and isinstance(cont.get("parts"), list) and cont["parts"]:
                                candidate_text = cont["parts"][0].get("text")
                if candidate_text is None:
                    candidate_text = resp.text

                # find first JSON object in text
                txt = candidate_text.strip()
                first = txt.find('{')
                last = txt.rfind('}')
                parsed = None
                if first != -1 and last != -1 and last > first:
                    json_str = txt[first:last+1]
                    try:
                        parsed = json.loads(json_str)
                    except Exception:
                        parsed = None

                if parsed and isinstance(parsed, dict) and "scores" in parsed:
                    for e in parsed["scores"]:
                        t = e.get("title")
                        sc = e.get("score")
                        if isinstance(t, str) and isinstance(sc, (int, float)):
                            scores[t] = float(sc)
                            self.llm_cache[self._llm_cache_key(source_title, t)] = float(sc)
                else:
                    # fallback: attempt to extract numbers line by line
                    for line in txt.splitlines():
                        for n in batch:
                            if n in line:
                                # try to find number in line
                                import re
                                m = re.search(r"([01](?:\.\d+)?)", line)
                                if m:
                                    try:
                                        sc = float(m.group(1))
                                        scores[n] = sc
                                        self.llm_cache[self._llm_cache_key(source_title, n)] = sc
                                    except:
                                        pass
                # small throttle
                time.sleep(0.05)
            except Exception as exc:
                # on error, assign neutral scores and continue
                for n in batch:
                    scores.setdefault(n, 0.5)
                    self.llm_cache.setdefault(self._llm_cache_key(source_title, n), 0.5)
                self.log("LLM scoring batch error:", exc)
                time.sleep(0.1)

        # ensure all neighbors have a score
        for n in neighbors:
            if n not in scores:
                k = self._llm_cache_key(source_title, n)
                if k in self.llm_cache:
                    scores[n] = float(self.llm_cache[k])
                else:
                    scores[n] = 0.5
                    self.llm_cache[k] = 0.5

        # persist cache
        save_llm_cache(self.llm_cache)
        return scores

    def _llm_cache_key(self, source: str, neighbor: str) -> str:
        # key format: source|neighbor
        return f"{source}||{neighbor}"

    def find_path_best_first(self, start_title: str, target_title: str, max_depth: int = 6,
                             max_visited: int = 50000, max_branch: int = 50,
                             use_llm: bool = True, candidate_pool_size: int = 30,
                             combine_alpha: float = 0.6, enqueue_k: int = 20):
        """
        Best-first search augmented with optional LLM re-ranking:
         - compute cheap title_score for neighbors
         - take top candidate_pool_size neighbors
         - get LLM scores (batched) for those candidates (if use_llm=True and GEMINI configured)
         - combine scores: combined = alpha * cheap + (1-alpha) * llm_score
         - enqueue top enqueue_k by combined score
        """
        start = self.resolve_title(start_title)
        target = self.resolve_title(target_title)
        if not start or not target:
            raise ValueError("Start or target missing/resolution failed")

        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)
        self.decision_info = {}

        if start == target:
            return [start]

        uid = 0
        heap: List[Tuple[float, int, int, str, List[str]]] = []
        start_score = self._title_score(start, target)
        heappush(heap, (-start_score, 0, uid, start, [start]))
        visited = {start}
        visited_count = 0

        while heap:
            neg_score, depth, _, current, path = heappop(heap)
            visited_count += 1
            if visited_count > max_visited:
                raise RuntimeError("Visited cap exceeded")
            if depth >= max_depth:
                continue

            try:
                neighbors_all = self.get_links(current)
            except Exception as e:
                self.log("get_links failed for", current, e)
                neighbors_all = []

            # add to graph
            for n in neighbors_all:
                if not self.crawl_graph.has_node(n):
                    self.crawl_graph.add_node(n)
                if not self.crawl_graph.has_edge(current, n):
                    self.crawl_graph.add_edge(current, n)

            # quick check
            if target in neighbors_all:
                self.decision_info[(current, target)] = {'method': 'best+llm' if use_llm else 'best', 'depth': depth + 1, 'note': 'direct neighbor - target found'}
                return path + [target]

            # filter neighbors not visited
            cand_neighbors = [n for n in neighbors_all if n not in visited]
            if not cand_neighbors:
                continue

            # compute cheap title scores
            cheap_scores = [(self._title_score(n, target), n) for n in cand_neighbors]
            cheap_scores.sort(reverse=True)
            top_m = [n for _, n in cheap_scores[:candidate_pool_size]]

            # optionally extract snippets for LLM context (only for top candidates)
            snippets_map: Dict[str, Tuple[str, str]] = {}
            for n in top_m:
                try:
                    sn = self.extract_anchor_snippet(current, n)
                    snippets_map[n] = (sn.get("anchor_text") or "", sn.get("source_snippet") or "")
                except Exception:
                    snippets_map[n] = ("", "")

            # get llm scores (or neutral 0.5 if not configured)
            try:
                llm_scores = self.llm_score_neighbors_batch(current, target, top_m, snippets_map=snippets_map)
            except Exception:
                llm_scores = {n: 0.5 for n in top_m}

            # combine and enqueue top-K
            combined_list = []
            for n in top_m:
                cheap = self._title_score(n, target)
                lscore = llm_scores.get(n, 0.5)
                combined = combine_alpha * cheap + (1.0 - combine_alpha) * lscore
                combined_list.append((combined, n, cheap, lscore))

            combined_list.sort(reverse=True, key=lambda x: x[0])
            top_to_enqueue = combined_list[:min(enqueue_k, len(combined_list))]
            for combined, n, cheap, lscore in top_to_enqueue:
                if n not in visited:
                    visited.add(n)
                    uid += 1
                    heappush(heap, (-combined, depth + 1, uid, n, path + [n]))
                    self.decision_info[(current, n)] = {'method': 'best+llm' if use_llm else 'best', 'depth': depth + 1, 'score': combined, 'cheap_score': cheap, 'llm_score': lscore, 'note': 'enqueued by combined heuristic+LLM'}
        return None

    # ------------------------
    # Bidirectional search
    # ------------------------
    def find_path_bidi(self, start_title: str, target_title: str, max_depth: int = 6, max_visited: int = 100000):
        start = self.resolve_title(start_title)
        target = self.resolve_title(target_title)
        if not start or not target:
            raise ValueError("Start or target missing/resolution failed")
        if start == target:
            return [start]

        self.crawl_graph = nx.DiGraph()
        self.crawl_graph.add_node(start)
        self.crawl_graph.add_node(target)
        self.decision_info = {}

        parent_fwd = {start: None}
        parent_bwd = {target: None}
        q_fwd = deque([(start, 0)])
        q_bwd = deque([(target, 0)])
        visited_fwd = {start}
        visited_bwd = {target}
        visited_count = 0

        while q_fwd and q_bwd:
            if len(q_fwd) <= len(q_bwd):
                current, depth = q_fwd.popleft()
                visited_count += 1
                if visited_count > max_visited:
                    raise RuntimeError("Visited cap exceeded")
                if depth >= max_depth:
                    continue
                try:
                    neighbors = set(self.get_links(current))
                except Exception as e:
                    self.log("get_links failed", current, e)
                    neighbors = set()

                for n in neighbors:
                    if not self.crawl_graph.has_node(n):
                        self.crawl_graph.add_node(n)
                    if not self.crawl_graph.has_edge(current, n):
                        self.crawl_graph.add_edge(current, n)

                inter = neighbors & visited_bwd
                if inter:
                    meet = next(iter(inter))
                    self.decision_info[(current, meet)] = {'method': 'bidi_fwd', 'depth': depth + 1, 'note': 'meeting edge (fwd)'}
                    return self._reconstruct_bidi_path(parent_fwd, parent_bwd, meet, start, target)
                for n in neighbors:
                    if n not in visited_fwd:
                        visited_fwd.add(n)
                        parent_fwd[n] = current
                        q_fwd.append((n, depth + 1))
                        self.decision_info[(current, n)] = {'method': 'bidi_fwd', 'depth': depth + 1, 'note': 'forward enqueued'}
            else:
                current, depth = q_bwd.popleft()
                visited_count += 1
                if visited_count > max_visited:
                    raise RuntimeError("Visited cap exceeded")
                if depth >= max_depth:
                    continue
                try:
                    incoming = set(self.get_linkshere(current))
                except Exception as e:
                    self.log("get_linkshere failed", current, e)
                    incoming = set()

                for n in incoming:
                    if not self.crawl_graph.has_node(n):
                        self.crawl_graph.add_node(n)
                    if not self.crawl_graph.has_edge(n, current):
                        self.crawl_graph.add_edge(n, current)

                inter = incoming & visited_fwd
                if inter:
                    meet = next(iter(inter))
                    self.decision_info[(meet, current)] = {'method': 'bidi_bwd', 'depth': depth + 1, 'note': 'meeting edge (bwd)'}
                    return self._reconstruct_bidi_path(parent_fwd, parent_bwd, meet, start, target)
                for n in incoming:
                    if n not in visited_bwd:
                        visited_bwd.add(n)
                        parent_bwd[n] = current
                        q_bwd.append((n, depth + 1))
                        self.decision_info[(n, current)] = {'method': 'bidi_bwd', 'depth': depth + 1, 'note': 'backward enqueued'}
        return None

    def _reconstruct_bidi_path(self, parent_fwd: Dict[str, Optional[str]], parent_bwd: Dict[str, Optional[str]], meeting_node: str, start: str, target: str) -> List[str]:
        left = []
        node = meeting_node
        while node is not None:
            left.append(node)
            node = parent_fwd.get(node)
        left = list(reversed(left))
        right = []
        cur = meeting_node
        while cur is not None and cur != target:
            nxt = parent_bwd.get(cur)
            if nxt is None:
                break
            right.append(nxt)
            cur = nxt
        full = left + right
        if full[0] != start:
            full = [start] + full
        if full[-1] != target:
            full = full + [target]
        return full

    # ------------------------
    # Explanation helpers
    # ------------------------
    def explain_path(self, path: Optional[List[str]]) -> str:
        if not path or len(path) < 2:
            return "No path or trivial path (start == target). Nothing to explain."
        lines = []
        lines.append("Explanation of the path found (step-by-step):")
        lines.append(f"Total hops: {len(path)-1}")
        lines.append("")
        for i in range(len(path)-1):
            a = path[i]
            b = path[i+1]
            info = self.decision_info.get((a, b)) or self.decision_info.get((b, a))
            step_no = i + 1
            if info:
                method = info.get('method', 'unknown')
                depth = info.get('depth')
                score = info.get('score')
                note = info.get('note', '')
                parts = [f"{step_no}. {a} -> {b}"]
                parts.append(f"   • method: {method}")
                if depth is not None:
                    parts.append(f"   • depth (click count from origin on that side): {depth}")
                if score is not None:
                    parts.append(f"   • combined score: {score:.4f}")
                if 'cheap_score' in info:
                    parts.append(f"   • cheap_score: {info.get('cheap_score'):.4f}")
                if 'llm_score' in info:
                    parts.append(f"   • llm_score: {info.get('llm_score'):.4f}")
                if note:
                    parts.append(f"   • note: {note}")
                lines.append("\n".join(parts))
            else:
                lines.append(f"{step_no}. {a} -> {b}\n   • reason: link observed during crawling (no recorded enqueue metadata).")
        return "\n".join(lines)

    # ------------------------
    # Anchor/snippet extractor
    # ------------------------
    def extract_anchor_snippet(self, source_title: str, target_title: str, max_chars: int = 250) -> Dict[str, Optional[str]]:
        """
        Return {"anchor_text": str|None, "source_snippet": str|None}
        """
        src = self.resolve_title(source_title)
        tgt = self.resolve_title(target_title)
        if not src or not tgt:
            return {"anchor_text": None, "source_snippet": None}
        params = {"action": "parse", "page": src, "prop": "text", "formatversion": 2}
        try:
            j = self._api_get(params)
            time.sleep(self.sleep)
        except Exception:
            return {"anchor_text": None, "source_snippet": None}
        html = j.get("parse", {}).get("text", "")
        if not html:
            return {"anchor_text": None, "source_snippet": None}
        soup = BeautifulSoup(html, "html.parser")
        tgt_frag = "/wiki/" + tgt.replace(" ", "_")
        a_tag = None
        # try exact href match
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.split("#")[0].split("?")[0] == tgt_frag:
                a_tag = a
                break
        if not a_tag:
            for a in soup.find_all("a", href=True):
                if tgt_frag in a["href"]:
                    a_tag = a
                    break
        if not a_tag:
            for a in soup.find_all("a", title=True):
                if a.get("title") == tgt:
                    a_tag = a
                    break
        if not a_tag:
            return {"anchor_text": None, "source_snippet": None}
        anchor_text = a_tag.get_text(strip=True) or None
        parent = a_tag
        for _ in range(4):
            if parent is None:
                break
            if parent.name in ("p", "li", "div", "td", "section"):
                break
            parent = parent.parent
        snippet = None
        if parent:
            text = parent.get_text(" ", strip=True)
            # sentence split
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            found = None
            for sent in sentences:
                if anchor_text and anchor_text in sent:
                    found = sent
                    break
                if tgt in sent:
                    found = sent
                    break
            if found:
                snippet = found.strip()
            else:
                snippet = text.strip()[:max_chars] + ("..." if len(text) > max_chars else "")
        return {"anchor_text": anchor_text, "source_snippet": snippet}

    # ------------------------
    # Flowchart / mindmap drawing
    # ------------------------
    def draw_flowchart(self, output_path: str, highlight_path: Optional[List[str]] = None, max_nodes: int = 500, mode: str = "pruned", hide_nonpath_labels: bool = True):
        if self.crawl_graph is None or len(self.crawl_graph.nodes) == 0:
            raise RuntimeError("No crawl graph recorded to draw.")
        G_full = self.crawl_graph.copy()
        path = highlight_path or []

        # choose nodes to keep
        if mode == "path-only":
            if not path:
                raise RuntimeError("path-only requires a highlight_path")
            nodes_keep = set(path)
            G = G_full.subgraph(nodes_keep).copy()
        elif mode == "path-neighbors":
            if not path:
                raise RuntimeError("path-neighbors requires a highlight_path")
            nodes_keep = set(path)
            for n in path:
                nodes_keep.update(G_full.successors(n))
                nodes_keep.update(G_full.predecessors(n))
            G = G_full.subgraph(nodes_keep).copy()
        elif mode == "full":
            G = G_full
        else:  # pruned or mindmap
            nodes_keep = set(path)
            for n in path:
                nodes_keep.update(G_full.successors(n))
                nodes_keep.update(G_full.predecessors(n))
            if len(nodes_keep) < max_nodes:
                deg_sorted = sorted(G_full.nodes, key=lambda n: G_full.degree(n), reverse=True)
                idx = 0
                while len(nodes_keep) < max_nodes and idx < len(deg_sorted):
                    nodes_keep.add(deg_sorted[idx])
                    idx += 1
            G = G_full.subgraph(nodes_keep).copy()

        # final prune safeguard
        if len(G.nodes) > max_nodes:
            keep = set(path)
            deg_sorted = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)
            idx = 0
            while len(keep) < max_nodes and idx < len(deg_sorted):
                keep.add(deg_sorted[idx])
                idx += 1
            G = G.subgraph(keep).copy()

        # layout
        pos = {}
        if mode == "mindmap":
            if not path:
                pos = nx.spring_layout(G, k=0.5, iterations=80)
            else:
                start = path[0]
                try:
                    lengths = nx.single_source_shortest_path_length(G.to_undirected(), start)
                except Exception:
                    lengths = {n: 0 for n in G.nodes()}
                layers = {}
                max_layer = 0
                for n, d in lengths.items():
                    layers.setdefault(d, []).append(n)
                    max_layer = max(max_layer, d)
                pos[start] = (0.0, 0.0)
                for layer in range(1, max_layer + 1):
                    nodes_in_layer = layers.get(layer, [])
                    if not nodes_in_layer:
                        continue
                    radius = 1.5 * layer
                    count = len(nodes_in_layer)
                    for i, node in enumerate(nodes_in_layer):
                        angle = (i / max(1, count)) * 2 * math.pi
                        x = radius * math.cos(angle)
                        y = radius * math.sin(angle)
                        pos[node] = (x, y)
                missing = [n for n in G.nodes if n not in pos]
                if missing:
                    subpos = nx.spring_layout(G.subgraph(missing), k=0.6, iterations=80)
                    offset = ( (max_layer + 2) * 1.5 )
                    for n, pnt in subpos.items():
                        pos[n] = (pnt[0] * 0.8 + offset, pnt[1] * 0.8)
        else:
            try:
                if len(path) >= 2:
                    for i, node in enumerate(path):
                        pos[node] = (i * 1.5, 0.0)
                    others = [n for n in G.nodes if n not in pos]
                    radius = 1.5
                    for i, node in enumerate(others):
                        angle = (i / max(1, len(others))) * 2 * math.pi
                        layer = 1 + ((i // 12) * 0.8)
                        pos[node] = ((len(path)*0.75) * math.cos(angle) * layer, radius * math.sin(angle) * layer)
                else:
                    pos = nx.spring_layout(G, k=0.5, iterations=80)
            except Exception:
                pos = nx.spring_layout(G, k=0.5, iterations=80)

        # styling
        path_set = set(path)
        node_sizes = []
        node_colors = []
        node_labels = {}
        for n in G.nodes:
            node_sizes.append(700 if n in path_set else 200)
            if path and n == path[0]:
                node_colors.append("#2ca02c")
            elif path and n == path[-1]:
                node_colors.append("#1f77b4")
            elif n in path_set:
                node_colors.append("#d62728")
            else:
                node_colors.append("#999999")
            node_labels[n] = _shorten_label(n, max_len=28) if (not hide_nonpath_labels or n in path_set) else ""

        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.92)
        nx.draw_networkx_edges(G, pos, arrowsize=12, arrowstyle='->', width=1, alpha=0.7)
        if path and len(path) >= 2:
            path_edges = list(zip(path[:-1], path[1:]))
            existing_path_edges = [e for e in path_edges if G.has_edge(*e)]
            if existing_path_edges:
                nx.draw_networkx_edges(G, pos, edgelist=existing_path_edges, arrowsize=14, arrowstyle='->', width=3, alpha=0.95)
        labeled_nodes = {n: lbl for n, lbl in node_labels.items() if lbl}
        if labeled_nodes:
            nx.draw_networkx_labels(G, pos, labels=labeled_nodes, font_size=9)
        import matplotlib.patches as mpatches
        legend_items = []
        if path:
            legend_items.append(mpatches.Patch(color="#2ca02c", label="Start"))
            legend_items.append(mpatches.Patch(color="#1f77b4", label="Target"))
            legend_items.append(mpatches.Patch(color="#d62728", label="Path nodes"))
        legend_items.append(mpatches.Patch(color="#999999", label="Other nodes"))
        plt.legend(handles=legend_items, loc='upper right', fontsize=9)
        plt.title("Wikipedia crawl graph — mode: {}".format(mode))
        plt.axis('off')
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=220)
        plt.close()
        self.log("Flowchart saved to:", output_path)

    # ------------------------
    # Rich explanation using LLM (optional) or fallback
    # ------------------------
    def build_llm_prompt_for_explanation(self, crawl_summary: Dict[str, Any], steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        system = ("You are a technical explainer. Given a crawler trace with short snippets, produce "
                  "a clear step-by-step explanation: for each hop explain why the link was reasonable referencing the snippet, comment on heuristics/LLM role, and finish with an assessment and suggestions.")
        lines = []
        lines.append("CRAWL SUMMARY")
        lines.append(f"- Strategy: {crawl_summary.get('strategy')}")
        lines.append(f"- Start: {crawl_summary.get('start')}")
        lines.append(f"- Target: {crawl_summary.get('target')}")
        lines.append(f"- Total hops: {crawl_summary.get('hops')}")
        lines.append(f"- Pages visited: {crawl_summary.get('visited_count')}")
        lines.append(f"- Time taken (s): {crawl_summary.get('elapsed_seconds'):.2f}")
        lines.append("")
        lines.append("PATH & DECISIONS")
        for s in steps:
            lines.append(f"STEP {s['i']}: {s['A']} -> {s['B']}")
            lines.append(f" - anchor_text: \"{(s.get('anchor_text') or '')[:120]}\"")
            lines.append(f" - source_snippet: \"{(s.get('source_snippet') or '')[:220]}\"")
            lines.append(f" - method: {s.get('method')}")
            if 'cheap_score' in s:
                lines.append(f" - cheap_score: {s.get('cheap_score')}")
            if 'llm_score' in s:
                lines.append(f" - llm_score: {s.get('llm_score')}")
            lines.append("")
        lines.append("INSTRUCTIONS:")
        lines.append("1) For each step, explain concisely (2-6 sentences) why this hop was reasonable, referencing anchor_text/snippet.")
        lines.append("2) If heuristics or LLM were used, explain their role.")
        lines.append("3) Finish with a short assessment: overall confidence (low/medium/high), whether a shorter path likely exists, and 2 brief suggestions to improve search.")
        user_prompt = "\n".join(lines)
        return {"system_prompt": system, "user_prompt": user_prompt, "max_tokens": 900, "temperature": 0.12}

    def call_gemini_for_explanation(self, payload: Dict[str, Any]) -> Optional[str]:
        api_key = os.getenv(GEMINI_KEY_ENV)
        if not api_key:
            return None
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        system = payload.get("system_prompt", "")
        user = payload.get("user_prompt", "")
        prompt_text = system.strip() + "\n\n" + user.strip()
        body = {"contents": [{"parts": [{"text": prompt_text}]}], "temperature": float(payload.get("temperature", 0.12)), "maxOutputTokens": int(payload.get("max_tokens", 900))}
        try:
            resp = requests.post(GEMINI_MODEL_URL, headers=headers, json=body, timeout=120)
            resp.raise_for_status()
            j = resp.json()
            # extract candidate text robustly
            if isinstance(j, dict):
                if "candidates" in j and isinstance(j["candidates"], list) and len(j["candidates"]) > 0:
                    cand = j["candidates"][0]
                    cont = cand.get("content", {})
                    if isinstance(cont, dict) and isinstance(cont.get("parts"), list) and len(cont["parts"]) > 0:
                        return cont["parts"][0].get("text")
                if "response" in j and isinstance(j["response"], dict):
                    resp_cands = j["response"].get("candidates")
                    if resp_cands and isinstance(resp_cands, list) and len(resp_cands) > 0:
                        c = resp_cands[0]
                        cont = c.get("content", {})
                        if isinstance(cont, dict) and isinstance(cont.get("parts"), list) and len(cont["parts"]) > 0:
                            return cont["parts"][0].get("text")
            return resp.text
        except Exception as e:
            self.log("Gemini call failed:", e)
            return None

    def produce_rich_explanation(self, path: List[str], strategy: str, visited_count: int, elapsed_seconds: float) -> str:
        # build summary + steps
        crawl_summary = {"strategy": strategy, "start": path[0] if path else None, "target": path[-1] if path else None, "hops": len(path)-1 if path else 0, "visited_count": visited_count, "elapsed_seconds": elapsed_seconds}
        steps = []
        for i in range(len(path)-1):
            A = path[i]; B = path[i+1]
            meta = self.decision_info.get((A, B)) or self.decision_info.get((B, A)) or {}
            sn = self.extract_anchor_snippet(A, B)
            step = {"i": i+1, "A": A, "B": B, "anchor_text": sn.get("anchor_text"), "source_snippet": sn.get("source_snippet")}
            step.update({k: meta.get(k) for k in ("method", "depth")})
            if 'cheap_score' in meta:
                step['cheap_score'] = meta.get('cheap_score')
            if 'llm_score' in meta:
                step['llm_score'] = meta.get('llm_score')
            steps.append(step)
        payload = self.build_llm_prompt_for_explanation(crawl_summary, steps)
        llm_text = self.call_gemini_for_explanation(payload)
        if llm_text:
            return llm_text
        # fallback: built-in explanation + snippets
        built = self.explain_path(path)
        extra = ["\n--- Anchor snippets (best-effort) ---"]
        for s in steps:
            extra.append(f"{s['i']}. {s['A']} -> {s['B']}")
            extra.append(f"   anchor_text: {s.get('anchor_text')}")
            extra.append(f"   snippet: {s.get('source_snippet')}\n")
        return built + "\n" + "\n".join(extra)

# ------------------------
# Interactive collector (core options only)
# ------------------------
def interactive_collect_core():
    print("=== WikiCrawler interactive setup (core options) ===")
    print("Leave blank to accept defaults shown in [brackets]. Type 'random' for random start.")
    start_or_random = ask("Start article (type 'random' to pick a random page)", default="random")
    if start_or_random.strip().lower() in ("random", "r"):
        random_start = True; start_title = None
    else:
        random_start = False; start_title = start_or_random
    def target_validator(s):
        if not s:
            raise ValueError("Target cannot be empty")
        return s
    target_title = ask("Target article (title or keywords)", default="", validator=target_validator)
    def strat_validator(s):
        s = s.lower()
        if s not in ("bfs","best","bidi"):
            raise ValueError("Choose one of: bfs, best, bidi")
        return s
    strategy = ask("Search strategy (bfs | best | bidi)", default="bidi", validator=strat_validator)
    want_flowchart = ask_yes_no("Save flowchart PNG after run?", default=False)
    flowchart = ""
    flowchart_mode = "mindmap"
    hide_nonpath_labels = True
    if want_flowchart:
        flowchart = ask("Flowchart output filepath", default="./flowchart.png")
        def fc_validator(s):
            s = s.lower()
            if s not in ("path-only","path-neighbors","pruned","mindmap","full"):
                raise ValueError("Choose one of: path-only, path-neighbors, pruned, mindmap, full")
            return s
        flowchart_mode = ask("Flowchart mode (path-only | path-neighbors | pruned | mindmap | full)", default="mindmap", validator=fc_validator)
        hide_nonpath_labels = ask_yes_no("Hide labels for non-path nodes to reduce clutter?", default=True)
    explain = True
    explain_file = ""
    return {"start_title": start_title, "random_start": random_start, "target_title": target_title, "strategy": strategy, "flowchart": flowchart, "flowchart_mode": flowchart_mode, "hide_nonpath_labels": hide_nonpath_labels, "explain": explain, "explain_file": explain_file}

# ------------------------
# Main entry
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Find link path between two Wikipedia pages (bfs, best, bidi).")
    parser.add_argument("--start", help="Start page title (quote if contains spaces).")
    parser.add_argument("--target", help="Target page title (quote if contains spaces).")
    parser.add_argument("--random-start", action="store_true", help="Pick a random start page instead of --start.")
    parser.add_argument("--strategy", choices=["bfs", "best", "bidi"], default="bidi", help="Search strategy: bfs|best|bidi.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum clicks (depth) to attempt.")
    parser.add_argument("--max-visited", type=int, default=50000, help="Safety cap on pages to visit.")
    parser.add_argument("--max-branch", type=int, default=50, help="For best-first: neighbors to consider per expansion.")
    parser.add_argument("--user-agent", help="Custom User-Agent header.")
    parser.add_argument("--sleep", type=float, default=0.12, help="Seconds to sleep between API requests.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed crawl progress.")
    parser.add_argument("--flowchart", help="If set, save PNG flowchart to this filepath.")
    parser.add_argument("--flowchart-mode", choices=["path-only","path-neighbors","pruned","mindmap","full"], default="mindmap", help="Flowchart detail mode.")
    parser.add_argument("--hide-nonpath-labels", action="store_true", help="Hide labels for non-path nodes in flowchart.")
    parser.add_argument("--explain-file", help="If set, save textual explanation to this file path.")
    parser.add_argument("--use-llm-hopping", action="store_true", help="Enable LLM-guided neighbor re-ranking during best-first (requires GEMINI_API_KEY).")
    parser.add_argument("--llm-candidates", type=int, default=30, help="Candidate pool size for LLM re-rank.")
    parser.add_argument("--llm-alpha", type=float, default=0.6, help="Alpha weight for cheap heuristic in combined score (0..1).")
    parser.add_argument("--llm-enqueue-k", type=int, default=20, help="How many top combined neighbors to enqueue per expansion.")
    # decide interactive vs CLI: interactive only when no CLI args given
    interactive_mode = (len(sys.argv) == 1)
    if interactive_mode:
        try:
            conf = interactive_collect_core()
        except SystemExit:
            return
        start_title = conf["start_title"]
        random_start = conf["random_start"]
        target_title = conf["target_title"]
        strategy = conf["strategy"]
        flowchart = conf["flowchart"]
        flowchart_mode = conf["flowchart_mode"]
        hide_nonpath_labels = conf["hide_nonpath_labels"]
        explain = conf["explain"]
        explain_file = conf["explain_file"]
        max_depth = parser.get_default("max_depth")
        max_visited = parser.get_default("max_visited")
        max_branch = parser.get_default("max_branch")
        user_agent = parser.get_default("user_agent")
        sleep = parser.get_default("sleep")
        verbose = parser.get_default("verbose")
        use_llm_hopping = False
        llm_candidates = parser.get_default("llm_candidates")
        llm_alpha = parser.get_default("llm_alpha")
        llm_enqueue_k = parser.get_default("llm_enqueue_k")
    else:
        args = parser.parse_args()
        start_title = args.start
        random_start = args.random_start
        target_title = args.target
        strategy = args.strategy
        max_depth = args.max_depth
        max_visited = args.max_visited
        max_branch = args.max_branch
        user_agent = args.user_agent
        sleep = args.sleep
        verbose = args.verbose
        flowchart = args.flowchart
        flowchart_mode = args.flowchart_mode
        hide_nonpath_labels = args.hide_nonpath_labels
        explain_file = args.explain_file
        explain = True
        use_llm_hopping = args.use_llm_hopping
        llm_candidates = args.llm_candidates
        llm_alpha = args.llm_alpha
        llm_enqueue_k = args.llm_enqueue_k

    crawler = WikiCrawler(user_agent=user_agent, sleep_between_requests=sleep, verbose=verbose)

    if random_start:
        start_title = crawler.random_page_title()
        print(f"Picked random start page: {start_title}")
    else:
        if not start_title and not interactive_mode:
            parser.error("Either --start or --random-start must be provided.")

    if not target_title:
        print("No target provided; exiting.")
        return

    try:
        resolved_target = crawler.resolve_title(target_title)
        if not resolved_target:
            print(f"Target '{target_title}' not found exactly; searching for best match...")
            resolved_target = crawler.search_title(target_title)
            if not resolved_target:
                raise SystemExit(f"Could not find a target page matching '{target_title}'.")
            print(f"Using target page: {resolved_target}")
        else:
            print(f"Target resolved to: {resolved_target}")

        resolved_start = crawler.resolve_title(start_title) if start_title else None
        if resolved_start is None and start_title:
            print(f"Start '{start_title}' not found exactly; searching for best match...")
            resolved_start = crawler.search_title(start_title)
            if not resolved_start:
                raise SystemExit(f"Could not find a start page matching '{start_title}'.")
            print(f"Using start page: {resolved_start}")
        elif resolved_start:
            print(f"Start resolved to: {resolved_start}")

        if not resolved_start:
            raise SystemExit("No start page specified.")

        t0 = time.time()
        path = None
        if strategy == "bfs":
            path = crawler.find_path_bfs(resolved_start, resolved_target, max_depth=max_depth, max_visited=max_visited)
        elif strategy == "best":
            path = crawler.find_path_best_first(resolved_start, resolved_target, max_depth=max_depth, max_visited=max_visited, max_branch=max_branch,
                                                use_llm=use_llm_hopping, candidate_pool_size=llm_candidates, combine_alpha=llm_alpha, enqueue_k=llm_enqueue_k)
        else:
            path = crawler.find_path_bidi(resolved_start, resolved_target, max_depth=max_depth, max_visited=max_visited)
        elapsed = time.time() - t0

        visited_count = len(crawler.crawl_graph.nodes)
        if path:
            print("\n=== PATH FOUND ===")
            for i, t in enumerate(path):
                print(f"{i:2d}. {t}")
            print(f"Total clicks: {len(path)-1}")
        else:
            print("\nNo path found within depth", max_depth)

        # produce rich explanation (LLM if available) or fallback
        rich = crawler.produce_rich_explanation(path or [], strategy, visited_count, elapsed)
        print("\n--- Explanation ---\n")
        print(rich)

        # save explanation if requested
        if explain_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(explain_file)), exist_ok=True)
            except Exception:
                pass
            with open(explain_file, "w", encoding="utf-8") as f:
                f.write(rich)
            print(f"\nExplanation saved to: {explain_file}")
        elif interactive_mode:
            if ask_yes_no("Do you want to save the explanation to a file?", default=True):
                ef = ask("Explanation filepath", default="./explanation.txt")
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(ef)), exist_ok=True)
                except Exception:
                    pass
                with open(ef, "w", encoding="utf-8") as f:
                    f.write(rich)
                print(f"Explanation saved to: {ef}")

        # draw flowchart if requested
        if flowchart:
            try:
                crawler.draw_flowchart(flowchart, highlight_path=path, max_nodes=800, mode=flowchart_mode, hide_nonpath_labels=hide_nonpath_labels)
                print(f"Flowchart saved to: {flowchart}")
            except Exception as e:
                print("Failed to draw flowchart:", e)

    except ValueError as ve:
        print("Error:", ve)

if __name__ == "__main__":
    main()


    print("Jai Ho!!!!!")
