#!/bin/bash
#
# Run all examples available with file name pattern *_example.py
# This script prepares a virtual environment and example specific setup like downloading data.
# We assume that a hadoop cluster is setup correctly and authentication is handled with kerberos
# It is necessary to setup some environment variables to run this script. Ex:
# export user=username
# export DOMAIN=@mydomain.com
# export KEYTAB=keytab_path/username.keytab

export USER=$user
export LOGNAME=$user
sudo chown -R $user:$user tf-yarn
MAINPATH=$(dirname $(dirname $(dirname $(realpath $0))))

kdestroy
kinit $user$DOMAIN -k -t $KEYTAB
klist

if [ ! -d "tf-yarn_test_env" ];
then
    # Setup environment
    python3.6 -m venv $MAINPATH/tf-yarn_test_env
	. $MAINPATH/tf-yarn_test_env/bin/activate
    pip install -r $MAINPATH/tf-yarn/tests-requirements.txt
    pip install -e $MAINPATH/tf-yarn
    pip install pex
    # Setup pex
    pex cryptography==2.1.4 tf-yarn -o tf-yarn/examples/example.pex
    hdfs dfs -put $MAINPATH/tf-yarn/examples/example.pex
    # Setup specific to examples
    # Get wine dataset for linear_classifier_example
    curl http://www3.dsi.uminho.pt/pcortez/wine/winequality.zip -o $MAINPATH/tf-yarn/examples/winequality.zip
    python -c "import zipfile; zip_ref = zipfile.ZipFile('tf-yarn/examples/winequality.zip', 'r'); zip_ref.extractall('tf-yarn/examples'); zip_ref.close()"
    hdfs dfs -put $MAINPATH/tf-yarn/examples/winequality/winequality-red.csv
else
    . $MAINPATH/tf-yarn_test_env/bin/activate
fi

# Execute examples
exit_code = 0
pushd $MAINPATH/tf-yarn/examples
    for example in *_example.py; do
        python $example; if ! [ $? -eq 0 ]; then exit_code=1; fi
    done
popd
exit $exit_code
