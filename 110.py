import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher

# Função para aplicar a Regra 110
def apply_rule_110(left, center, right):
    if left == 1 and center == 1 and right == 1:
        return 0
    if left == 1 and center == 1 and right == 0:
        return 1
    if left == 1 and center == 0 and right == 1:
        return 1
    if left == 1 and center == 0 and right == 0:
        return 0
    if left == 0 and center == 1 and right == 1:
        return 1
    if left == 0 and center == 1 and right == 0:
        return 1
    if left == 0 and center == 0 and right == 1:
        return 1
    if left == 0 and center == 0 and right == 0:
        return 0
    return -1  # caso de erro, se ocorrer

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
def generate_cellular_automaton(n, num_steps):
    G = nx.DiGraph()

    seen_states = set()

    # Adicionar nós (estados) para cada combinação de n células
    for i in range(2 ** n):
        state = format(i, f'0{n}b')  # Representação binária do estado
        canon_state = canonical_form(state)
        if canon_state not in seen_states:
            seen_states.add(canon_state)
            G.add_node(canon_state)

    # Adicionar arestas (transições) entre os estados
    for node in list(G.nodes()):
        current_state = [int(bit) for bit in node]
        next_state = [0] * n

        # Aplicar a Regra 110 para determinar o próximo estado
        for i in range(n):
            left = current_state[(i - 1) % n]
            center = current_state[i]
            right = current_state[(i + 1) % n]
            next_state[i] = apply_rule_110(left, center, right)

        next_state_str = ''.join(map(str, next_state))
        next_canon_state = canonical_form(next_state_str)
        G.add_edge(node, next_canon_state)

    # Simular o autômato celular por num_steps passos
    current_state = format(0, f'0{n}b')  # Estado inicial
    states_history = [current_state]
    for _ in range(num_steps):
        next_state = [0] * n

        # Aplicar a Regra 110 para determinar o próximo estado
        for i in range(n):
            left = int(current_state[(i - 1) % n])
            center = int(current_state[i])
            right = int(current_state[(i + 1) % n])
            next_state[i] = apply_rule_110(left, center, right)

        next_state_str = ''.join(map(str, next_state))
        next_canon_state = canonical_form(next_state_str)
        states_history.append(next_canon_state)
        current_state = next_canon_state

    return G, states_history

# Função para plotar o grafo do autômato celular
def plot_cellular_automaton(G, states_history):
    pos = nx.spring_layout(G)  # Layout ajustado para uma melhor visualização
    labels = {node: node for node in G.nodes()}

    plt.figure(figsize=(12, 8))

    # Plotar o grafo
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # Animação da evolução dos estados
    for i in range(len(states_history) - 1):
        edge_labels = {(states_history[i], states_history[i + 1]): apply_rule_110(int(states_history[i][-1]), int(states_history[i][0]), int(states_history[i][1]))}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)

    plt.title('Autômato Celular Circular - Regra 110 (1 passo)')
    plt.axis('off')
    plt.show()

# Função para calcular o grupo de automorfismo
def calculate_automorphism_group(G):
    GM = GraphMatcher(G, G)
    automorphisms = list(GM.isomorphisms_iter())
    return automorphisms

# Parâmetros do autômato celular
n = 10  # Número de células
num_steps = 1  # Número de passos de simulação

# Gerar o grafo e simular o autômato celular
G, states_history = generate_cellular_automaton(n, num_steps)

# Calcular o grupo de automorfismo
automorphisms = calculate_automorphism_group(G)
print("Número de automorfismos:", len(automorphisms))
for i, aut in enumerate(automorphisms):
    print(f"Automorfismo {i+1}: {aut}")

# Plotar o autômato celular
plot_cellular_automaton(G, states_history)
