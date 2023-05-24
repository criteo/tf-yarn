#!/bin/bash
#
# Run this script from root dir ./tf_yarn/examples/run_pytorch_examples.sh
#
# Run all examples available with file name pattern *_example.py
# This script prepares a virtual environment and example specific setup like downloading data.

echo "run examples .."

exit_code=0

# the following packages must be installed in your (ubuntu) env for this to work
# sudo apt install liblzma-dev lzma

for pytorch_version in "1.13" "2.0"
do
    echo "running with pytorch ${pytorch_version} .."

    # Cleanup old artefacts
    rm -rf tf-yarn_test_env_pytorch
    hdfs dfs -rm -r -f tf_yarn_test/tf_yarn_*

    # Setup environment
    python3.9 -m venv tf-yarn_test_env_pytorch
    . tf-yarn_test_env_pytorch/bin/activate
    python3.9 -m pip install --upgrade pip setuptools wheel
    if [[ $pytorch_version == "1.13" ]]; then
        python3.9 -m pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
    else
        python3.9 -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
    fi
    # Workaround for https://github.com/pytorch/pytorch/issues/97258
    python3.9 -m pip install tensorflow==2.12.0 tensorflow_io==0.32.0

    python3.9 -m  pip install -e .
    python3.9 -m pip install webdataset==0.2.48 mlflow-skinny

    python3.9 -m pip freeze |grep -e torch -e pex -e tensor

    # Execute examples
    pushd tf_yarn/examples/pytorch
        for example in *_example.py; do
            echo "executing $example with torch=${pytorch_version} .."
            python $example
            if ! [ $? -eq 0 ]; then
                exit_code=1
                echo "error $example with torch=${pytorch_version}"
            else
                echo "done $example with torch=${pytorch_version}"
            fi
            echo "============================================="
        done
    popd
    deactivate
done

exit $exit_code
