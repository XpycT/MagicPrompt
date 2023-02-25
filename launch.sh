#!/usr/bin/env bash

function locate {
        dirname -- "$( readlink -f -- "$0"; )"
}

cd $( locate )

if [[ -a $( locate )/venv/bin/activate ]]
then
    source $( locate )/venv/bin/activate
elif [[ ! -d $( locate )/venv ]]
then
    python3 -m venv $( locate )/venv
    source $( locate )/venv/bin/activate
else
    echo "Could not locate venv/bin/activate";
    exit
fi

python3 -m pip install -r requirements.txt | grep -v 'already satisfied'
python3 MagicPrompt.py
