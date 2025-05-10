#!/bin/bash

run_and_check() {
    "$@"
    status=$?
    if [ $status -ne 0 ]; then
        echo "Error with \"$@\": Exited with status $status"
        exit $status
    fi
    return $status
}

echo "I: Installing tools required for pre-commit checks..."
run_and_check apt install clang-format-12

echo "I: Installing pre-commit itself..."
run_and_check pip3 install pre-commit
run_and_check pre-commit install

echo "I: Installation successful."
