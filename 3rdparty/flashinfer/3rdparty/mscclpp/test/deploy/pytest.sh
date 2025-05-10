#!/bin/bash
set -e

if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
  pytest /root/mscclpp/python/test/test_mscclpp.py -x -v
else
  pytest /root/mscclpp/python/test/test_mscclpp.py -x 2>&1 >/dev/null
fi
