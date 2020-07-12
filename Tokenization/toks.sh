#!/bin/bash  
export CLASSPATH=/home/devansh/Documents/EMNLP/stanford-corenlp-latest/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0.jar
echo "$(<$1)" | java edu.stanford.nlp.process.PTBTokenizer > out.txt
