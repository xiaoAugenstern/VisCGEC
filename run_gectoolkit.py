# -*- encoding: utf-8 -*-

import argparse
import sys
import os
from gectoolkit.quick_start import run_toolkit
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='nlpcc18', help='name of datasets')
    parser.add_argument('--model', '-m', type=str, default='T5', help='name of models')
    parser.add_argument('--augment', type=str, default='none', choices=['none', 'noise', 'translation'],
                        help='use data augmentation or not')

    parser.add_argument('--use_llm', action='store_true', help='use llm or not')
    parser.add_argument('--example_num', type=int, default='0', help='number of examples used for LLMs')


    args, _ = parser.parse_known_args()
    config_dict = {}

    method_name = None

    run_toolkit(method_name, args.model, args.dataset, args.augment, args.use_llm, args.example_num, config_dict)
