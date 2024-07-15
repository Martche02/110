import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from sympy.combinatorics import Permutation, PermutationGroup
import csv
import threading
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit
import numpy as np

# Função para aplicar uma regra geral usando Numba
@njit
def apply_rule(left, center, right, rule_binary):
    index = (left << 2) | (center << 1) | right
    return int(rule_binary[7 - index])

# Função para gerar a representação canônica de uma configuração circular
def canonical_form(state):
    n = len(state)
    min_state = state
    for i in range(n):
        rotated_state = state[i:] + state[:i]
        if rotated_state < min_state:
            min_state = rotated_state
    return min_state

# Função para gerar o grafo do autômato celular circular
def generate_cellular_automaton(n, rule):
    G = nx.DiGraph()

    rule_binary = np.array([int(bit) for bit in format(rule, '08b')], dtype=np.int32)  # Converter a regra para binário de 8 bits
    seen_states = set()

    # Adicionar nós (estados) para cada combinação de n células
    def add_edges(state):
        canon_state = canonical_form(state)
        if canon_state not in seen_states:
            seen_states.add(canon_state)
            G.add_node(canon_state)
            current_state = [int(bit) for bit in state]
            next_state = np.zeros(n, dtype=np.int32)
            for i in range(n):
                left = current_state[(i - 1) % n]
                center = current_state[i]
                right = current_state[(i + 1) % n]
                next_state[i] = apply_rule(left, center, right, rule_binary)
            next_state_str = ''.join(map(str, next_state))
            next_canon_state = canonical_form(next_state_str)
            return (canon_state, next_canon_state)
        return None

    states = [format(i, f'0{n}b') for i in range(2 ** n)]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(add_edges, state) for state in states]
        for future in as_completed(futures):
            edge = future.result()
            if edge is not None:
                G.add_edge(*edge)

    return G

# Função para calcular o grupo de automorfismo
def calculate_automorphism_group(G):
    GM = GraphMatcher(G, G)
    automorphisms = list(GM.isomorphisms_iter())
    return automorphisms

# Função para identificar o grupo de automorfismos
def identify_automorphism_group(automorphisms, node_labels):
    label_to_index = {label: idx for idx, label in enumerate(node_labels)}
    permutations = []
    for aut in automorphisms:
        perm = [label_to_index[aut[label]] for label in node_labels]
        permutations.append(Permutation(perm))
    
    group = PermutationGroup(permutations)
    return group

# Função para rodar com timeout
def automorphism_group_order_worker(rule, n, queue):
    try:
        G = generate_cellular_automaton(n, rule)
        automorphisms = calculate_automorphism_group(G)
        node_labels = list(G.nodes())
        automorphism_group = identify_automorphism_group(automorphisms, node_labels)
        queue.put(automorphism_group.order())
    except Exception as e:
        queue.put(e)

def automorphism_group_order(rule, n):
    queue = Queue()
    thread = threading.Thread(target=automorphism_group_order_worker, args=(rule, n, queue))
    thread.start()
    thread.join(timeout=30)
    
    try:
        result = queue.get_nowait()
        if isinstance(result, Exception):
            return "-"
        return result
    except Empty:
        return "-"

# Função para processar cada combinação de rule e n
def process_rule_n(rule, n):
    order = automorphism_group_order(rule, n)
    return (rule, n, order)

# Função para gerar a tabela e salvar em CSV
def generate_table_and_save():
    with open('automorphism_groups.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Rule", "n", "Order"])

        n = 3
        while True:
            with ThreadPoolExecutor(max_workers=8) as executor:  # Use um número apropriado de threads
                future_to_rule_n = {executor.submit(process_rule_n, rule, n): rule for rule in range(256)}
                
                all_completed = True
                for future in as_completed(future_to_rule_n):
                    rule = future_to_rule_n[future]
                    try:
                        result = future.result()
                        writer.writerow(result)
                        file.flush()
                        if result[2] == "-":
                            all_completed = False
                    except Exception as e:
                        print(f"Erro ao processar rule {rule} e n {n}: {e}")
                
                if not all_completed:
                    break
            n += 1

# Executar a função para gerar a tabela
generate_table_and_save()
