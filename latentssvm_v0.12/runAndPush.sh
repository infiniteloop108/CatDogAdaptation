#!/bin/bash
./svm_latent_learn -c 1 finalCatTrain finalCatModel
purge
./svm_latent_learn -c 1 finalDogTrain finalDogModel
purge
git add .
git commit -m "final cat and dog learnt"
pushitbaby
