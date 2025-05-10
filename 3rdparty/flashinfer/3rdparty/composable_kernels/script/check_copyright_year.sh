#!/bin/bash

current_year=$(date +%Y)
exit_code=0

for file in $@; do
    if grep -q "Copyright (c)" $file
    then
        if ! grep -q "Copyright (c).*$current_year" $file
        then
            echo "ERROR: File $file has a copyright notice without the current year ($current_year)."
            exit_code=1
        fi
    fi
done

exit $exit_code
