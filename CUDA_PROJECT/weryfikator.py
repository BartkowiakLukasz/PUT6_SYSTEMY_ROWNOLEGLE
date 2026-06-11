import networkx as nx
import numpy as np
import sys

def generate_dokuwiki_graphviz(G, label=""):
    # NetworkX domyślnie indeksuje od 0 (0-16). Dla DokuWiki chcemy indeksować od 1 (1-17), więc dodajemy 1 do każdego wierzchołka.
    nodes_str = ";".join(str(n + 1) for n in G.nodes()) + ";"
    
    # Generowanie listy krawędzi (np. 1--14)
    edges_str = ";".join(f"{u + 1}--{v + 1}" for u, v in G.edges()) + ";"
    
    # Składanie wszystkiego w gotowy blok DokuWiki
    wiki_block = (
        "<graphviz circo>\n"
        f'graph g{{{nodes_str}{edges_str}label="{label}";}}\n'
        "</graphviz>"
    )
    return wiki_block

def format_list(lst):
    # Konwertuje pythonową listę [1, 2, 3] na ciąg bez spacji {1,2,3}
    return "{" + ",".join(map(str, lst)) + "}"

def get_graph6_string(G):
    # networkx domyślnie dodaje nagłówek '>>graph6<<', więc go usuwamy
    g6_bytes = nx.to_graph6_bytes(G, header=False)
    return g6_bytes.decode('ascii').strip()

def verify_graphs(filename):
    try:
        # Wczytujemy plik linia po linii, aby zachować oryginalny ciąg Graph6(G)
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Nie znaleziono pliku: {filename}")
        return

    # Wyczyść zawartość pliku przed pętlą
    with open("doku_wiki_znalezione_grafy.txt", "w") as f_wiki:
        pass

    rejected_graphs = []

    for g6_str in lines:
        try:
            # Parsowanie oryginalnego ciągu graph6
            G = nx.from_graph6_bytes(g6_str.encode('ascii'))
        except Exception as e:
            print(f"Błąd parsowania grafu '{g6_str}': {e}")
            continue

        # 1. Parametry podstawowe (n, e)
        n = G.number_of_nodes()
        e = G.number_of_edges()

        # 2. Sprawdzenie spójności grafu
        is_conn = nx.is_connected(G)

        # 3. Ciąg stopni (deg) posortowany malejąco
        deg_seq = sorted([d for node, d in G.degree()], reverse=True)

        # 4. Liczba trójkątów (t)
        t = sum(nx.triangles(G).values()) // 3

        # 5. Widmo grafu G (Sp) oraz weryfikacja całkowitości
        A = nx.to_numpy_array(G)
        eigenvalues = np.linalg.eigvalsh(A)
        rounded_eigenvalues = np.round(eigenvalues)
        
        # Jeśli zaokrąglone wartości są identyczne z rzeczywistymi (z bardzo małą tolerancją na błąd obliczeń zmiennoprzecinkowych) -> graf jest całkowity
        is_integral = np.allclose(eigenvalues, rounded_eigenvalues, atol=1e-9)
        Sp = sorted([int(val) for val in rounded_eigenvalues], reverse=True)

        if not (is_conn and is_integral):
            rejected_graphs.append(g6_str)
            continue

        # 6. Graf dopełniający (G^C)
        G_C = nx.complement(G)
        
        # 7. Widmo grafu dopełniającego Sp(G^C)
        A_C = nx.to_numpy_array(G_C)
        eigenvalues_C = np.linalg.eigvalsh(A_C)
        Sp_C = sorted([int(round(val)) for val in eigenvalues_C], reverse=True)

        # 8. Format graph6 dla dopełnienia Graph6(G^C)
        g6_C_str = get_graph6_string(G_C)

        wiki_G_block = generate_dokuwiki_graphviz(G, g6_str)
        wiki_GC_block = generate_dokuwiki_graphviz(G_C, g6_C_str)

        # Wypisywanie wyników w zadanym formacie
        print(f"n={n}")
        print(f"e={e}")
        print(f"deg={format_list(deg_seq)}")
        print(f"t={t}")
        print(f"Sp={format_list(Sp)}")
        print(f"Graph6(G)={g6_str}")
        print(f"Sp(G_C)={format_list(Sp_C)}")
        print(f"Graph6(G_C)={g6_C_str}")
        print(f"Spojny={'TAK' if is_conn else 'NIE'}")
        print(f"Calkowity={'TAK' if is_integral else 'NIE'}")
        print()
        print()

        # Zapisz informacje oraz kod dokuwiki do pliku
        with open("doku_wiki_znalezione_grafy.txt", "a") as f_wiki:
            info_str = (
                f"|n=|{n}|\n"
                f"|e=|{e}|\n"
                f"|deg=|{format_list(deg_seq)}|\n"
                f"|t=|{t}|\n"
                f"|Sp=|{format_list(Sp)}|\n"
                f"|Graph6(G)=|<nowiki>{g6_str}</nowiki>|\n"
                f"|Sp(G<nowiki>^</nowiki>C)=|{format_list(Sp_C)}|\n"
                f"|Graph6(G<nowiki>^</nowiki>C)=|<nowiki>{g6_C_str}</nowiki>|\n"
            )
            f_wiki.write(info_str + "\n" + wiki_G_block + "\n\n" + "----" + "\n\n")

    if rejected_graphs:
        print("\n=== Grafy niespełniające warunków (nie są jednocześnie spójne i całkowite) ===")
        for g in rejected_graphs:
            print(g)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python3 weryfikator.py <plik_z_grafami>")
    else:
        verify_graphs(sys.argv[1])