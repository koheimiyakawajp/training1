#!/bin/bash

if [ ! -e "./spectra" ];then
    mkdir "./spectra"
fi

while read url
do
    curl -O ${url}
    echo "curl -O ${url}"
    fname=`echo ${url} | cut -f 10 -d "/"`
    mv ${fname} "./spectra/"
done < $1

gzip -d "./spectra/"*
