#!/bin/bash


echo 
echo "*******************"
echo "TRAINING MODELS"
echo "*******************"

# set environmental variables
MODELS=./sklearn_models/run_all/*
LOG=./sklearn_models/run_all_results.txt
SCRIPT=./run_models.py
# if necessary edit pythonpath for the current run 
# PP=~/PycharmProjects/Bike-sharing-kaggle-competition/BikeSharing/

# clear the log
echo > $LOG

# run through all the models
for f in $MODELS
do
  echo >> $LOG
  echo $f >> $LOG
  echo >> $LOG
  echo "---------------------------"
  echo "Running model $f"
  echo "---------------------------"
  echo >> $LOG
  # env PYTHONPATH=$PP
  python $SCRIPT $f >> $LOG
done