#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for process mining
"""

import sys
from termcolor import colored

def main():
    """
    Main CLI entry point
    """
    from processmine.config import parse_arguments
    from process_mining.core.runner import run_analysis
    
    # Parse arguments
    args = parse_arguments()
    
    # Convert arguments to dictionary
    config = vars(args)
    
    try:
        # Run analysis
        run_analysis(args.data_path, **config)
    except KeyboardInterrupt:
        print(colored("\n\n⚠️ Process mining interrupted by user", "yellow"))
        sys.exit(1)
    except Exception as e:
        print(colored(f"\n\n❌ Unexpected error: {e}", "red"))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()