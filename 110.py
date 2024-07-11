import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup, AlternatingGroup, CyclicGroup, DihedralGroup

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

# Função para identificar o grupo de automorfismos
def identify_group(automorphism_group):
    n = automorphism_group.degree
    order = automorphism_group.order()
    
    # Verificar se é um grupo simétrico
    if automorphism_group.is_isomorphic(SymmetricGroup(n)):
        return f"O grupo de automorfismos é isomórfico ao Grupo Simétrico S{n}."
    # Verificar se é um grupo alternado
    elif automorphism_group.is_isomorphic(AlternatingGroup(n)):
        return f"O grupo de automorfismos é isomórfico ao Grupo Alternado A{n}."
    # Verificar se é um grupo cíclico
    for k in range(1, n+1):
        if automorphism_group.is_isomorphic(CyclicGroup(k)):
            return f"O grupo de automorfismos é isomórfico ao Grupo Cíclico C{k}."
    # Verificar se é um grupo diedral
    for k in range(2, n+1):
        if automorphism_group.is_isomorphic(DihedralGroup(k)):
            return f"O grupo de automorfismos é isomórfico ao Grupo Diedral D{k}."
    
    return "O grupo de automorfismos não corresponde a um grupo bem conhecido."

# Função para plotar o grafo do grupo de automorfismos
def plot_automorphism_group(automorphisms):
    aut_group = nx.DiGraph()

    # Adicionar nós representando automorfismos
    for i, aut in enumerate(automorphisms):
        aut_group.add_node(i, permutation=aut)

    # Adicionar arestas representando a composição de automorfismos
    for i in range(len(automorphisms)):
        for j in range(len(automorphisms)):
            composed = compose_automorphisms(automorphisms[i], automorphisms[j])
            for k, aut in enumerate(automorphisms):
                if composed == aut:
                    aut_group.add_edge(i, k)

    pos = nx.spring_layout(aut_group)  # Layout ajustado para uma melhor visualização
    labels = {i: f'Aut {i+1}' for i in aut_group.nodes()}

    plt.figure(figsize=(12, 8))

    # Plotar o grafo do grupo de automorfismos
    nx.draw_networkx_nodes(aut_group, pos, node_color='lightgreen', node_size=700)
    nx.draw_networkx_edges(aut_group, pos, edge_color='black', arrows=True)
    nx.draw_networkx_labels(aut_group, pos, labels=labels, font_size=12)

    plt.title('Grupo de Automorfismos do Grafo')
    plt.axis('off')
    plt.show()

# Função para compor dois automorfismos
def compose_automorphisms(aut1, aut2):
    return {k: aut2[v] for k, v in aut1.items()}

# Função para imprimir a tabela de Cayley
def print_cayley_table(group):
    elements = list(group.generate(af=False))
    table = []
    for x in elements:
        row = []
        for y in elements:
            row.append(x * y)
        table.append(row)
    
    header = "  | " + " | ".join(map(str, elements)) + " |"
    print(header)
    print("-" * len(header))
    for elem, row in zip(elements, table):
        print(f"{elem} | " + " | ".join(map(str, row)) + " |")

# Parâmetros do autômato celular
n = 8  # Número de células
num_steps = 1  # Número de passos de simulação

# Gerar o grafo e simular o autômato celular
G, states_history = generate_cellular_automaton(n, num_steps)

# Calcular o grupo de automorfismo
automorphisms = calculate_automorphism_group(G)

# Obter rótulos dos nós para mapeamento
node_labels = list(G.nodes())

# Identificar o grupo de automorfismos
automorphism_group = identify_automorphism_group(automorphisms, node_labels)

# Exibir informações sobre o grupo de automorfismos
print("Número de automorfismos:", len(automorphism_group.generators))
print("Ordem do grupo de automorfismos:", automorphism_group.order())
print("Geradores do grupo de automorfismos:", automorphism_group.generators)

# Plotar o grupo de automorfismos
plot_automorphism_group(automorphisms)

# Identificar o grupo
group_type = identify_group(automorphism_group)
print(group_type)

# Imprimir a tabela de Cayley
print("Tabela de Cayley:")
print_cayley_table(automorphism_group)
