#!/bin/bash
cd "$(dirname "$0")"
inkscape -f HyperFETConstruction.svg -e HyperFETConstruction.png -y 1
inkscape -f HFvsGeo.svg -e HFvsGeo.png -y 1
