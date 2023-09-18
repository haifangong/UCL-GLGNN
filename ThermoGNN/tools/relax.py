import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Use rosetta to relax the protein structure according to the mutant list.")
    parser.add_argument('-l', '--mutant-list', type=str, dest='mutant_list', required=True,
                        help='A list of mutants, one per line in the format "PDBCHAIN POS WT MUT"')
    parser.add_argument('-i', '--input-pdb-dir', type=str, dest='input_pdb_dir', required=True,
                        help='The directory storing the original PDB files.')
    parser.add_argument('--rosetta-bin', type=str, dest="rosetta_bin", required=True,
                        help='Rosetta FastRelax binary executable.')
    parser.add_argument('-o', '--output-dir', type=str, dest="output_dir", required=True,
                        help='The directory to store all output data.')

    args = parser.parse_args()

    mutants = []
    for l in open(args.mutant_list, 'r'):
        pdb_chain, pos, w, m = l.strip().split()
        mutants.append((pdb_chain, w + pos + m))

    output_dir = os.path.abspath(args.output_dir)
    input_dir = os.path.abspath(args.input_pdb_dir)

    for pdb_chain, mutant in mutants:
        # create and change to necessary directory
        chain_dir = os.path.join(output_dir, pdb_chain)
        if not os.path.exists(chain_dir):
            os.makedirs(chain_dir)

        os.chdir(chain_dir)

        # create a resfile
        mutant_resfile = pdb_chain + '_' + mutant + '.resfile'
        with open(mutant_resfile, 'wt') as opf:
            opf.write('NATAA\n')
            opf.write('start\n')
            opf.write(mutant[1:-1] + ' ' + pdb_chain[-1] +
                      ' PIKAA ' + mutant[-1])

        wild_type_struct = os.path.join(input_dir, pdb_chain + ".pdb")

        if os.path.exists(wild_type_struct):
            rosetta_relax_cmd = ' '.join([args.rosetta_bin, '-in:file:s', wild_type_struct, '-in:file:fullatom',
                                         '-relax:constrain_relax_to_start_coords', '-out:suffix', '_relaxed',
                                          '-out:no_nstruct_label', '-relax:ramp_constraints false', '-detect_disulf false',
                                          '-out:path:score', chain_dir])
            os.system(rosetta_relax_cmd)
        else:
            raise FileNotFoundError(
                f"Require wild-type structure for {pdb_chain} at {wild_type_struct}")

        start_struct = os.path.join(
            output_dir, pdb_chain, pdb_chain + '_relaxed.pdb')
        rosetta_relax_cmd = ' '.join([args.rosetta_bin, '-in:file:s', start_struct, '-in:file:fullatom',
                                      '-relax:constrain_relax_to_start_coords',
                                      '-out:no_nstruct_label', '-relax:ramp_constraints false',
                                      '-relax:respect_resfile', '-detect_disulf false',
                                      '-packing:resfile', mutant_resfile,
                                      '-out:file:scorefile', os.path.join(
                                          chain_dir, pdb_chain + '_relaxed.sc'),
                                      '-out:suffix', '_' + mutant + '_relaxed'])

        os.system(rosetta_relax_cmd)
        os.system('rename _relaxed_ _ * _relaxed_*.pdb')


if __name__ == "__main__":
    main()
