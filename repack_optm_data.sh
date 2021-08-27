#!/bin/bash

cp ./optimization_archive/dcai_gcb_10_*.zip ./ --force

for I in {1..31}
do
    rm -r ./temp
    mkdir ./temp
    echo dcai_gcb_10_${I}.zip
    unzip ./dcai_gcb_10_${I}.zip -d ./temp
    mv ./temp/dcai_gcb_10/ ./temp/dcai_gcb_10_${I}
    cd ./temp
    zip -r ./dcai_gcb_10_${I}.zip ./dcai_gcb_10_${I}
    mv ./dcai_gcb_10_${I}.zip ../ --force
    cd ../
done
