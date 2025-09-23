#!/bin/bash

echo "MLP"
python3 TwoMoonTauMLP.py

echo "tau 0"
python3 TwoMoonTau.py --tau 0

echo "tau 0.001"
python3 TwoMoonTau.py --tau 0.001

echo "tau 0.01"
python3 TwoMoonTau.py --tau 0.01

echo "tau 0.1"
python3 TwoMoonTau.py --tau 0.1

echo "tau 0.5"
python3 TwoMoonTau.py --tau 0.5
