import networkx as nx
from Bio.PDB import PDBParser, is_aa

from ThermoGNN.utils.features import get_node_feature, read_hhm_file, read_scoring_functions


def get_CA(res):
    return res["CA"]


def make_graph(record, aa_features, out_dir, is_wt=True, split="train", contact_threshold=5, local_radius=12):

    if len(record.strip().split()) == 5:  # with known ddG
        pdb_name, mut_pos, wt, mut, ddG = record.strip().split()
        ddG = float(ddG)
        G = nx.Graph(y=ddG)
    else:
        pdb_name, mut_pos, wt, mut = record.strip().split()
        G = nx.Graph()

    p = PDBParser()

    pdb_id, chain = pdb_name[:-1], pdb_name[-1]
    mut_pos = int(mut_pos)

    if is_wt:
        suffix = pdb_name
        out_path = f"{out_dir}/{split}/{pdb_name}_{wt}{mut_pos}{mut}_wt.pkl"
    else:
        suffix = f"{pdb_name}_{wt}{mut_pos}{mut}"
        out_path = f"{out_dir}/{split}/{pdb_name}_{wt}{mut_pos}{mut}_mut.pkl"

    pdb_path = f"data/pdbs/{split}/{pdb_name}/{suffix}_relaxed.pdb"
    hhm_path = f"data/hhm/{split}/{suffix}.hhm"

    structure = p.get_structure(pdb_name, pdb_path)
    chain = structure[0][chain]

    mut_res = chain[mut_pos]
    mut_center = get_CA(mut_res)

    for res in chain:
        if is_aa(res.get_resname(), standard=True):
            center = get_CA(res)
            distance = center - mut_center
            if distance <= local_radius:
                G.add_node(res.id[1], name=res.get_resname())

    num_nodes = len(G.nodes)
    nodes_list = list(G.nodes)
    mut_index = nodes_list.index(mut_res.id[1])
    G.graph['mut_pos'] = mut_index

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            m = nodes_list[i]
            n = nodes_list[j]
            distance = get_CA(chain[m]) - get_CA(chain[n])
            if distance <= contact_threshold:
                G.add_edge(m, n, weight=contact_threshold / distance)

    mat = read_hhm_file(hhm_path)

    scoring = read_scoring_functions(pdb_path)

    G = nx.convert_node_labels_to_integers(G)

    features = get_node_feature(nodes_list, mat, scoring, aa_features, chain)

    for i, node in enumerate(G.nodes.data()):
        node[1]['x'] = features[i]

    nx.write_gpickle(G, out_path)
