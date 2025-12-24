
import unittest
import sys
import os

def run_suite():
    print("##################################################")
    print("#   CMFO v3.1 - SYSTEM VERIFICATION SUITE        #")
    print("##################################################")
    
    loader = unittest.TestLoader()
    start_dir = 'tests_v3'
    suite = loader.discover(start_dir)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ GLOBAL SUCCESS: ALL SYSTEMS NOMINAL.")
        return 0
    else:
        print(f"\n❌ FAILURES DETECTED: {len(result.failures)} Errors, {len(result.errors)} Crashes.")
        return 1

if __name__ == '__main__':
    sys.exit(run_suite())
