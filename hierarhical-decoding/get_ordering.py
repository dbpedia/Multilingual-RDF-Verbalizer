__author__='thiagocastroferreira'

"""
Author: Thiago Castro Ferreira
Date: 26/03/2019
Description:
    This script is responsible to retrieve the triples based on the output of the Discourse Ordering
    and Text Structuring steps of the pipeline approach. Moreover, it also wikifies a Lexicalization template
    ARGS:
        [1] Path to the textual file where the inputs of Discourse Ordering/Text Structuring are
        [2] Path to the textual file where the outputs should be saved
        [3] Flag to indicate the pipeline step: Discourse Ordering -> ordering / Text Structuring -> structing / Lexicalization -> lexicalization
        [4] File name to save the output
    EXAMPLE:
        python3 mapping.py ordering/dev.eval ordering/dev.ordering.postprocessed ordering ordering/dev.ordering.mapped
"""

import sys

def run(out_path):
    ordering_lines = []
    with open(out_path) as f:
        outputs = f.read().split('\n')
        for output in outputs:
            tokens = [token for token in output.split() if token not in ["<SNT>", "</SNT>"]] 
            ordering_triples = ' '.join(tokens)
            ordering_lines.append(ordering_triples)
    return ordering_lines


if __name__ == '__main__':
    out_path = sys.argv[1]
    write_path = sys.argv[2]
    result = run(out_path=out_path)

    with open(write_path, 'w') as f:
        f.write('\n'.join(result))
