#!/bin/bash

EXIT_CODE=0

function print_result() {
  if [ $1 -eq 0 ]; then
    echo_green "[Passed]"
  else
    echo_red "[Failed]"
    EXIT_CODE=1
  fi
  echo ""
}

FAILED=()
for file in ../build/test*
do
  if [ -x $file ]; then
    ./$file
    if [ $? -ne 0 ]; then
       FAILED=("${FAILED[@]}" $file)
    fi
  fi
done

for i in "${FAILED[@]}"; do echo_red "FAILED $i"; done
