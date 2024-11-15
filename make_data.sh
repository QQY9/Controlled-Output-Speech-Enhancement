#!/bin/bash


path=eggs/val/tr
if [[ ! -e $path ]]; then
    mkdir -p $path
fi
python -m denoiser.audio ***/val/noisy > $path/noisy.json
python -m denoiser.audio ***/val/clean > $path/clean.json

