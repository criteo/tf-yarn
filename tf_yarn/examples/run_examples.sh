#!/bin/bash
#
# Run this script from root dir ./examples/run_examples.sh
#
# Run all examples available with file name pattern *_example.py
# This script prepares a virtual environment and example specific setup like downloading data.

echo "run examples .."

exit_code=0

for tf_version in "1.15.2" "2.5.2"
do
    echo "running with tensorflow ${tf_version} .."

    # Cleanup old artefacts
    rm -rf tf-yarn_test_env
    hdfs dfs -rm -r -f tf_yarn_test/tf_yarn_*

    # Setup environment
    python3.6 -m venv tf-yarn_test_env
    . tf-yarn_test_env/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install -e .
    if [[ $tf_version == "1.15.2" ]]; then
        pip install tensorflow-io==0.8.1 #also installs tensorflow==1.15.5
        pip install tensorflow==${version} # force the correct version of tf after install of tfio

        # https://github.com/pantsbuild/pex/issues/913
        # only pex 2.1.1 is supported for tf 1.15
        pip install pex==2.1.1

        #no version available for tf==2.5.2
        pip install horovod==0.19.2+criteo.${tf_version}
    else
        pip install tensorflow-io==0.19.1 # also installs tensorflow==2.5.2
    fi
    pip install mlflow-skinny
    export CRITEO_MLFLOW_TRACKING_URI="https://mlflow.da1.preprod.crto.in"
    echo ' '
    pip freeze |grep -e tensor -e pex -e horovod
    echo ' '

    # Setup specific to examples
    # Get wine dataset for linear_classifier_example
    hdfs dfs -test -e tf_yarn_test/winequality-red.csv
    if [[ $? == 1 ]]; then
        echo "downloading winequality.zip .."
        curl http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip -o tf_yarn/examples/winequality.zip
        python -c "import zipfile; zip_ref = zipfile.ZipFile('tf_yarn/examples/winequality.zip', 'r'); zip_ref.extractall('tf_yarn/examples'); zip_ref.close()"
        hdfs dfs -mkdir tf_yarn_test
        hdfs dfs -put -f tf_yarn/examples/winequality/winequality-red.csv tf_yarn_test/winequality-red.csv
    fi

    # Execute examples
    pushd tf_yarn/examples
        for example in *_example.py; do
            if [[ "$example" == "native_keras_with_gloo_example.py" && $tf_version == "1.15.2" ]]; then
                continue
            fi
            if [[ "$example" == "collective_all_reduce_example.py" && $tf_version == "2.5.2" ]]; then
                continue
            fi
            echo "executing $example with tf=${tf_version} .."
            python $example
            if ! [ $? -eq 0 ]; then
                exit_code=1
                echo "error $example with tf=${tf_version}"
            else
                echo "done $example with tf=${tf_version}"
            fi
            echo "============================================="
        done
    popd
done

exit $exit_code
