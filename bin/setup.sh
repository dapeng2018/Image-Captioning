#!/bin/sh

# Var paths
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LIB_DIR="$DIR/../lib"
STV_DIR="$LIB_DIR/stv"

# Create lib dir and stv dir if they do not exist
if [[ ! -d $LIB_DIR ]]; then mkdir $LIB_DIR; fi
if [[ ! -d $STV_DIR ]]; then mkdir $STV_DIR; fi

# Downloads dependencies
wget -P /tmp "http://msvocds.blob.core.windows.net/coco2014/train2014.zip"
wget -P /tmp "http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip"
wget -P /tmp "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
wget -P $LIB_DIR "ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy"

# Extract files to their appropriate paths
tar -xvf "/tmp/train2014.zip" -C $LIB_DIR
tar -xvf "/tmp/caption_train-val2014.zip" -C $LIB_DIR
tar -xvf "/tmp/skip_thoughts_uni_2017_02_02.tar.gz" -C $STV_DIR

exit 0
