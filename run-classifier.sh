#!/bin/bash

if [[ "$1" =~ .*".zip".* ]]; then
  unzip $1 -d temp
elif [[ "$1" =~ .*".7z".* ]]; then
  7z x $1 -o./temp
elif [[ "$1" =~ .*".tar".* ]]; then
  tar -xf $1 -C temp
else
  echo "Cannot extract file. Formats accepted: .zip, .7z, .tar"
  exit
fi

docker cp temp/* license-classifier-container:/usr/src/app/extracted

docker commit license-classifier-container license-classifier

docker run -i license-classifier

rm -rf temp/*
