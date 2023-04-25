import os
import argparse
import warnings
from tempfile import NamedTemporaryFile

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder


def pdb2seq(pdb_dir):

    ppb = PPBuilder()
    records = []

    for pdb_path in os.listdir(pdb_dir):

        if pdb_path.endswith('.pdb'):

            pdb_path = os.path.join(pdb_dir, pdb_path)
            structure = PDBParser().get_structure('pdb', pdb_path)

            pdb = os.path.splitext(os.path.basename(pdb_path))[0]
            pdb = pdb.replace('_relaxed', '')

            chain_name = pdb[4]

            chain = structure[0][chain_name]

            pp = ppb.build_peptides(chain)

            sequence = ''.join([str(p.get_sequence()) for p in pp])
            record = SeqRecord(Seq(sequence), id=pdb, description='')
            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Use hhblits to generate .hhm files")
    parser.add_argument('-i', '--input-pdb-dir', type=str, dest='input_pdb_dir', required=True,
                        help='The directory storing the PDB files.')
    parser.add_argument('-db', '--hhsuite-db', type=str, dest="hhsuite_db", required=True,
                        help='Path to HHsuite database.')
    parser.add_argument('-o', '--output-dir', type=str, dest="output_dir", required=True,
                        help='The directory to store all output data.')
    parser.add_argument('--cpu', type=str, default=4,
                        help='number of CPUs to use (default: 4)')

    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    records = pdb2seq(args.input_pdb_dir)

    for record in records:
        f = NamedTemporaryFile(prefix='tmp', suffix='.fasta')
        SeqIO.write([record], f.name, "fasta")
        hhblits_cmd = ' '.join(['hhblits', '-i', f.name, '-o', '/dev/null',
                                '-ohhm', os.path.join(args.output_dir,
                                                      record.id + ".hhm"),
                                '-d', args.hhsuite_db, '-n 3', '-cpu', args.cpu])

        print(hhblits_cmd)
        os.system(hhblits_cmd)


if __name__ == "__main__":
    main()
