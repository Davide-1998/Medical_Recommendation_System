#!/bin/bash

# Useful variables
dataset=../data/datasetB.json
patients=../data/datasetB_cases.txt
python=python3

# Start recommendation system over professor's dataset
$python Medical_recommendations.py -d $dataset --recommend $patients

# Start evaluation of recommendation system oer professor dataset
# 5 runs used to save time
$python Medical_recommendations.py -d $dataset --e 5

# Start evaluation over the randomly generated datasets
$python Medical_recommendations.py --e 5

# Longer evaluation: Uncomment to run
# $python Medical_recommendations.py -d $dataset --evaluate 30
# $python Medical_recommendations.py --evaluate 30
