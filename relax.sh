#!/bin/bash


python ThermoGNN/tools/relax.py -i input-pdb \
                                -l mutant_list.txt \
                                --rosetta-bin relax.static.linuxgccrelease \
                                -o data/pdbs/demo/
