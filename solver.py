"""
File: solveer.py
Date: 10.12.2021 (created)
      10.13.2021 (updated)
Author: Shih-Che Sun
Synopsis:
    An interface that handles simulator, problem definition, and visualizer.
"""

import common.simulator
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib


def main(p, m, v, o, H, gamma, thr):
    #sim = simulator.Environment.fromDict(p.env)
    #sim.policyIteration()
    #sim.valueIteration()

    pi, V = m.solve(p.env, H, gamma, thr)
    v.visualize(pi, V, imageOutput=o)

    '''
    a snippet for evaluating the system from a initial state
    sim.initialize((2, 4))
    sim.step('right')
    sim.step('down')
    sim.step('stay')
    sim.step('left')
    sim.step('stay')
    sim.step('down')
    sim.step('left')
    print(sim.state)
    '''



if __name__ == '__main__':
    # usage: python solver.py problem -p problem_definition -s solving_method -v visualizer
    # when "problem" is given, it searches modules under the three directories
    # (problems, method, visualizer) with matching name.
    # When any of -p, -s, or -v is given, the default definition/method/visualizer
    # is overriden.
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("-p", "--problem-def", dest="p")
    parser.add_argument("-m", "--solving-method", dest="m", default="pi",
        choices=["pi", "policy-iteration", "vi", "value-iteration", "gs", "graph-search"])
    parser.add_argument("-v", "--visualizer", dest="v")
    parser.add_argument("-o", "--image-output", dest="o")
    parser.add_argument("--horizon", type=int, default=-1)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--converge-threshold", dest="thr", type=float, default=0.2)
    parser.add_argument("--function-approximation", dest="fa", action="store_true")
    args = parser.parse_args()

    problem = args.p or args.problem
    method = ((args.m == "pi" or args.m == "policy-iteration") and "policyIteration") or \
        ((args.m == "vi" or args.m == "value-iteration") and "valueIteration") or \
        ((args.m == "gs" or args.m == "graph-search") and "graphSearch")
    if args.fa:
        method = "valueIterationFA"
    visualizer = args.v or args.problem
    o = args.o or ('images/' + problem)

    p = importlib.import_module("problem." + problem)
    m = importlib.import_module("method." + method)
    v = importlib.import_module("visualizer." + visualizer)

    main(p, m, v, o, args.horizon, args.gamma, args.thr)





















#
