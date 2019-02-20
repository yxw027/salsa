#!/bin/bash

for file in ../build/test*
do
  if [ -x $file ]; then
    ./$file
  fi
done
