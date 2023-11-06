#!/bin/bash

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade

kaggle datasets download -d anasmohammedtahir/covidqu -f train_hq.zip
unzip train_hq.zip
mv train_hq/* data/
rm -d train_hq
rm train_hq.zip