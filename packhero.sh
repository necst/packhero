#!/bin/bash
if [ $# -eq 0 ]
then
    echo "No arguments supplied"
    exit 1
elif [ "$1" == "--dir" -o "$1" == "--file" ] && [ "$2" == "--majority" -o "$2" == "--mean" -o $2 == "--clustering" ]
then
    python -W ignore packhero.py "$2" "$3" "$4"
else
    echo "Invalid argument"
    exit 1
fi