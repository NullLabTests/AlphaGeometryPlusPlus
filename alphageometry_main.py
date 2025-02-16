#!/usr/bin/env python3
import argparse
import logging
import os
import time

import ddar  # Existing DDAR module
from alphageometry import graph
from alphageometry import lm_inference  # LM inference module
from alphageometry import pretty
from alphageometry import problem
from alphageometry import trace_back
import jax  # JAX for model inference

# WAM and additional modules
from alphageometry import wam_compiler
from alphageometry import wam_geometry
from alphageometry import neuro_symbolic_bridge
from alphageometry import meta_learning

# Global flags variable
FLAGS = None

def main(argv):
    if FLAGS.alsologtostderr:
        logging.getLogger().setLevel(logging.INFO)
    print('Command line arguments:')
    print(argv)

    start_time = time.time()
    problems = problem.load_problems(FLAGS.problems_file)
    problem_names = problems[0]
    theorems = problems[1]

    g = graph.Graph()
    g.load_problems(problem_names, theorems, FLAGS.problem_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Problem Loaded in {elapsed_time:.4f} seconds")

    if FLAGS.mode == 'ddar':
        proof = ddar.deduce(g)
    elif FLAGS.mode == 'alphageometry':
        proof = alphageometry_solve(g)
    else:
        raise ValueError('Unsupported mode: %s.' % FLAGS.mode)

    pretty_proof = pretty.pretty_print(proof, FLAGS.out_file)

def alphageometry_solve(g):
    """Combines the neural LM with the symbolic DDAR solver."""
    # 1. Attempt solving with DDAR
    proof = ddar.deduce(g)
    if proof:
        print("DD+AR solved the problem.")
        return proof
    else:
        print("DD+AR failed to solve the problem.")

    # 2. Iteratively use LM suggestions combined with WAM deductions
    num_iterations = 2  # Tune as needed
    for i in range(num_iterations):
        print(f"Iteration {i + 1}/{num_iterations}")

        # Get LM suggestions
        lm_suggestion = lm_inference.get_lm_suggestion(g)

        # Translate LM suggestion into WAM instructions
        wam_code = neuro_symbolic_bridge.translate_to_wam(lm_suggestion)

        # Execute the generated WAM code
        wam_result = wam_geometry.execute_wam(wam_code)

        # Integrate any new facts deduced by the WAM engine
        if wam_result:
            print(f"WAM deduced new facts: {wam_result}")
            g.add_facts(wam_result)

        # Try DDAR again with the augmented graph
        proof = ddar.deduce(g)
        if proof:
            print("DD+AR solved the problem with LM/WAM assistance.")
            return proof

        # Meta-learning step: update strategy based on failure
        meta_learning.update_strategy(lm_suggestion, success=False)

    print("Could not solve after LM/WAM attempts.")
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems_file', type=str, required=True,
                        help='Path to the file containing the problem definitions.')
    parser.add_argument('--problem_name', type=str, required=True,
                        help='Name of the problem to solve.')
    parser.add_argument('--mode', type=str, required=True,
                        help='Solving mode: "ddar" or "alphageometry".')
    parser.add_argument('--defs_file', type=str, default='defs.txt',
                        help='Path to the file containing the geometric definitions.')
    parser.add_argument('--rules_file', type=str, default='rules.txt',
                        help='Path to the file containing the deduction rules.')
    parser.add_argument('--beam_size', type=int, default=2,
                        help='Beam size for the LM search.')
    parser.add_argument('--search_depth', type=int, default=2,
                        help='Search depth for the DDAR solver.')
    parser.add_argument('--ckpt_path', type=str, default='ag_ckpt_vocab',
                        help='Path to the LM checkpoint.')
    parser.add_argument('--vocab_path', type=str, default='ag_ckpt_vocab/geometry.757.model',
                        help='Path to the vocabulary file.')
    parser.add_argument('--out_file', type=str,
                        help='Output file for the proof (optional).')
    parser.add_argument('--alsologtostderr', action='store_true',
                        help='Also log to stderr')
    FLAGS, unparsed = parser.parse_known_args()
    jax.config.config_with_absl()
    main(argv=[__file__] + unparsed)
