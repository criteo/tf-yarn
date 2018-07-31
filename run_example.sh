#!/bin/bash

set -xe

export HADOOP_HDFS_HOME=/usr/lib/hadoop-hdfs
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
export LD_LIBRARY_PATH=${JAVA_HOME}/jre/lib/amd64/server
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/lib/hadoop-criteo/hadoop/lib/native

PYTHONPATH=`git rev-parse --show-toplevel` python $@
