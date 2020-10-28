#!/bin/bash
#
# Run this script from root dir ./examples/run_examples.sh
#
# Run all examples available with file name pattern *_example.py
# This script prepares a virtual environment and example specific setup like downloading data.

echo "run examples .."

exit_code=0

for tf_version in "1.15.2" "2.2.0"
do
    echo "running with tensorflow ${tf_version} .."

    # Cleanup old artefacts
    rm -rf tf-yarn_test_env
    hdfs dfs -rm -r -f tf_yarn_test/tf_yarn_*

    # Setup environment
    python3.6 -m venv tf-yarn_test_env
    . tf-yarn_test_env/bin/activate
    pip install --upgrade pip setuptools
    pip install tensorflow==${tf_version}
    if [[ $tf_version == "1.15.2" ]]; then
        # https://github.com/pantsbuild/pex/issues/913
        # only pex 2.1.1 is supported for tf 1.15
        pip install pex==2.1.1
    fi
    pip install -e .
    pip install mlflow
    pip install horovod==0.19.2+criteo.${tf_version}

    # Setup specific to examples
    # Get wine dataset for linear_classifier_example
    hdfs dfs -test -e tf_yarn_test/winequality-red.csv
    if [[ $? == 1 ]]; then
        echo "downloading winequality.zip .."
        curl http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip -o examples/winequality.zip
        python -c "import zipfile; zip_ref = zipfile.ZipFile('examples/winequality.zip', 'r'); zip_ref.extractall('examples'); zip_ref.close()"
        hdfs dfs -mkdir tf_yarn_test
        hdfs dfs -put -f examples/winequality/winequality-red.csv tf_yarn_test/winequality-red.csv
    fi

    # Execute examples
    pushd examples
        for example in *_example.py; do
            if [[ "$example" == "native_keras_with_gloo_example.py" && $tf_version == "1.15.2" ]]; then
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
        done
    popd
done

exit $exit_code
