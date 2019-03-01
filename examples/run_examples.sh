#!/bin/bash
#
# Run this script from root dir ./tf-yarn/examples/run_examples.sh
#
# Run all examples available with file name pattern *_example.py
# This script prepares a virtual environment and example specific setup like downloading data.
# We assume that a hadoop cluster is setup correctly and authentication is handled with kerberos
# It is necessary to setup some environment variables to run this script. Ex:
# export user=username
# export DOMAIN=@mydomain.com
# export KEYTAB=keytab_path/username.keytab

echo "run examples .."

export USER=$user
export LOGNAME=$user
sudo chown -R $user:$user tf-yarn

kinit $user$DOMAIN -k -t $KEYTAB

# Cleanup old artefacts
rm -rf tf-yarn_test_env
rm -f tf-yarn/examples/example.pex
hdfs dfs -rm -r -f tf_yarn_test/tf_yarn_*

# Setup environment
python3.6 -m venv tf-yarn_test_env
. tf-yarn_test_env/bin/activate
pip install -e tf-yarn
pip install pex==1.5.2

# Setup pex
pex tf-yarn -o tf-yarn/examples/tf-yarn.pex

# Setup specific to examples
# Get wine dataset for linear_classifier_example
hdfs dfs -test -e tf_yarn_test/winequality-red.csv
if [[ $? == 1 ]]; then
    echo "downloading winequality.zip .."
    curl http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip -o tf-yarn/examples/winequality.zip
    python -c "import zipfile; zip_ref = zipfile.ZipFile('tf-yarn/examples/winequality.zip', 'r'); zip_ref.extractall('tf-yarn/examples'); zip_ref.close()"
    hdfs dfs -put -f tf-yarn/examples/winequality/winequality-red.csv tf_yarn_test/winequality-red.csv
fi

# Execute examples
exit_code=0
pushd tf-yarn/examples
    for example in *_example.py; do
        echo "executing $example .."
        python $example
        if ! [ $? -eq 0 ]; then
            exit_code=1
            echo "error on $example"
        fi
    done
popd
exit $exit_code
