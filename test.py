import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--a")
args, u = parser.parse_args()
print(args, u)

parser.add_argument("--b")
args, u = parser.parse_known_args()
print(args, u)
