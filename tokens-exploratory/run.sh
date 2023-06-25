#!/bin/sh
#SBATCH -c 16               # Request 1 CPU core
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 mins (this requests 2 hours)
#SBATCH -p dl               # Partition to submit to (should always be dl, for now)
#SBATCH --mem=100G          # Request 100G of memory
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written (%j inserts jobid)
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written (%j inserts jobid)
#/home/angus/anaconda3/envs/nlp-research/bin/python /home/angus/summer-2023/tokens-exploratory/bootstrap_final.py  # Command you want to run on the cluster
