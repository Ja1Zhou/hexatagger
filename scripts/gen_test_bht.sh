#!/usr/bin/env bash
pushd data > /dev/null
python dep2bht.py example.conll
popd > /dev/null