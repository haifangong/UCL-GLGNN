from Bio.PDB import *
from Bio.PDB.Polypeptide import PPBuilder, three_to_one


def read_next_nline(f, n):
    for i in range(n):
        line = f.readline()
    return line


def profile2freq(value):
    if value == "*":
        return 0
    else:
        return 2 ** (-int(value) / 1000)


def read_hhm_file(hhm):
    step = 1
    profile = []
    with open(hhm, "r") as f:
        line = f.readline()
        while line and not line.startswith("//"):
            if line.startswith("HMM "):
                step = 3
            line = read_next_nline(f, step)
            if step == 3 and not line.startswith("//"):
                data = [profile2freq(v) for v in line.split()[2:-1]]
                profile.append(data)

        return profile


def read_scoring_functions(pdb):
    scoring = False
    profile = []
    for line in open(pdb):
        if line.startswith("VRT"):
            scoring = False
        if scoring:
            data = [float(v) for v in line.split()[1:-1]]
            profile.append(data)
        if line.startswith("pose"):
            scoring = True
    return profile


def load_aa_features(feature_path):
    aa_features = {}
    for line in open(feature_path):
        line = line.strip().split()
        aa, features = line[0], line[1:]
        features = [float(feature) for feature in features]
        aa_features[aa] = features
    return aa_features


def get_node_feature(nodes_list, profile, scoring, aa_features, chain):
    features = []

    ppb = PPBuilder()
    pp = ppb.build_peptides(chain)
    res_list = []
    for p in pp:
        res_list.extend(p)

    for node in nodes_list:
        res = chain[int(node)]
        data = list(profile[res_list.index(res)])   # positional encoding
        score = list(scoring[res_list.index(res)])  # rosetta scoring function
        aa_feature = aa_features[three_to_one(res.get_resname())]   # sequence encoding
        features.append(data + score + aa_feature)

    return features
    