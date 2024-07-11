import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher
from sympy.combinatorics import Permutation, PermutationGroup
from joblib import Parallel, delayed

# Função para aplicar uma regra geral
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
def generate_cellular_automaton(n, num_steps, rule):
    G = nx.DiGraph()

    rule_binary = format(rule, '08b')  # Converter a regra para binário de 8 bits
    seen_states = set()

    # Adicionar nós (estados) para cada combinação de n células
    for i in range(2 ** n):
        state = format(i, f'0{n}b')  # Representação binária do estado
        canon_state = canonical_form(state)
        if canon_state not in seen_states:
            seen_states.add(canon_state)
            G.add_node(canon_state)

    # Adicionar arestas (transições) entre os estados
    def add_edges(node):
        current_state = [int(bit) for bit in node]
        next_state = [0] * n
        for i in range(n):
            left = current_state[(i - 1) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            next_state[i] = apply_rule(left, center, right, rule_binary)
        next_state_str = ''.join(map(str, next_state))
        next_canon_state = canonical_form(next_state_str)
        return (node, next_canon_state)

    edges = Parallel(n_jobs=-1)(delayed(add_edges)(node) for node in list(G.nodes()))
    G.add_edges_from(edges)

    # Simular o autômato celular por num_steps passos
    current_state = format(0, f'0{n}b')  # Estado inicial
    states_history = [current_state]
    for _ in range(num_steps):
        next_state = [0] * n
        for i in range(n):
            left = int(current_state[(i - 1) % n])
            center = int(current_state[i])
            right = int(current_state[(i + 1) % n])
            next_state[i] = apply_rule(left, center, right, rule_binary)
        next_state_str = ''.join(map(str, next_state))
        next_canon_state = canonical_form(next_state_str)
        states_history.append(next_canon_state)
        current_state = next_canon_state

    return G, states_history

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

# Função para plotar o grafo
def plot_graph(G):
    pos = nx.spring_layout(G)  # Layout ajustado para uma melhor visualização

    plt.figure(figsize=(12, 8))

    # Plotar o grafo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=10, font_weight='bold', arrowsize=20)
    plt.title('Grafo Direcionado dos Elementos do Conjunto')
    plt.show()

# Parâmetros do autômato celular
n = 8  # Número de células
num_steps = 1  # Número de passos de simulação
rule = 110  # Regra do autômato celular

# Gerar o grafo e simular o autômato celular
G, states_history = generate_cellular_automaton(n, num_steps, rule)

# Calcular o grupo de automorfismo
automorphisms = calculate_automorphism_group(G)

# Obter rótulos dos nós para mapeamento
node_labels = list(G.nodes())

# Identificar o grupo de automorfismos
automorphism_group = identify_automorphism_group(automorphisms, node_labels)

# Exibir informações sobre o grupo de automorfismos
unique_generators = set(tuple(gen) for gen in automorphism_group.generators)
print("Número de geradores diferentes do grupo de automorfismos:", len(unique_generators))
print("Ordem do grupo de automorfismos:", automorphism_group.order())

# Plotar o grafo dos elementos do conjunto
plot_graph(G)
