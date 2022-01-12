#!/bin/bash

WORKDIR=`pwd`
MODULEDIR="${WORKDIR}/src/ucsgnet"

echo ${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:${MODULEDIR}
echo ${PYTHONPATH}