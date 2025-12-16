#!/usr/bin/env python3
"""
CMFO CLI - Command Line Interface
==================================
Simple CLI tool for CMFO operations.
"""

import sys
import argparse
from cmfo import CMFOIntegrated, __version__

def main():
    parser = argparse.ArgumentParser(
        description=f"CMFO CLI v{__version__} - Fractal Computation Tool"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Parse command
    parse_parser = subparsers.add_parser('parse', help='Parse text to vector')
    parse_parser.add_argument('text', help='Text to parse')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve equation')
    solve_parser.add_argument('equation', help='Equation to solve')
    
    # Distance command
    dist_parser = subparsers.add_parser('distance', help='Calculate distance')
    dist_parser.add_argument('word1', help='First word')
    dist_parser.add_argument('word2', help='Second word')
    
    # Version command
    subparsers.add_parser('version', help='Show version')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'version':
        print(f"CMFO v{__version__}")
        return 0
    
    with CMFOIntegrated() as cmfo:
        if args.command == 'parse':
            vec = cmfo.parse(args.text)
            print(f"{args.text} = {vec}")
        
        elif args.command == 'solve':
            solution = cmfo.solve(args.equation)
            print(solution)
        
        elif args.command == 'distance':
            v1 = cmfo.parse(args.word1)
            v2 = cmfo.parse(args.word2)
            d = cmfo.distance(v1, v2)
            print(f"d_φ({args.word1}, {args.word2}) = {d:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
