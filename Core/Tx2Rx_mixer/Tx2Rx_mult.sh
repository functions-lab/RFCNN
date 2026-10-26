#!/bin/bash

time=$1
sync=$2

deviceTx=$3
rateTx=$4
clockTx=$5
fileTx=$6
carrierTx=$7
gainTx=$8
saveTx=$9

deviceRx=${10}
rateRx=${11}
clockRx=${12}
fileRx=${13}
carrierRx=${14}
gainRx=${15}
saveRx=${16}

python "$(dirname -- "${BASH_SOURCE[0]}")"/USRP_mult.py \
    --addr $deviceTx --rate $rateTx --time 100 --sync $sync --clock $clockTx --save $saveTx \
    --fileTx $fileTx --freqTx $carrierTx --gainTx $gainTx &
pid=$!
sleep 4s
python "$(dirname -- "${BASH_SOURCE[0]}")"/USRP_mult.py \
    --addr $deviceRx --rate $rateRx --time $time --sync $sync --clock $clockRx --save $saveRx \
    --fileRx $fileRx --freqRx $carrierRx --gainRx $gainRx
kill -9 $pid
