import os
import sys
import tempfile
import uuid
import argparse
import logging
import tensorflow as tf
import skein
import subprocess
import time

from typing import Tuple, List

import cluster_pack


logger = logging.getLogger(__name__)

PATH_ON_HDFS = f"{cluster_pack.get_default_fs()}/tmp/{uuid.uuid4()}"
FILENAME_ON_HDFS = "hello_tf_yarn.txt"
FILEPATH_ON_HDFS = f"{PATH_ON_HDFS}/{FILENAME_ON_HDFS}"
EXPECTED_CONTENT = "Hello tf-yarn!"
RESULT_CHECK_FILE = 'check_hadoop_env.log'


def write_dummy_file_to_hdfs() -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        new_file = f"{temp_dir}/hello.txt"
        with open(new_file, 'w') as fd:
            fd.write(EXPECTED_CONTENT)
        logger.info(f"write dummy file to hdfs {FILEPATH_ON_HDFS}")
        tf.io.gfile.makedirs(PATH_ON_HDFS)
        tf.io.gfile.copy(new_file, FILEPATH_ON_HDFS, overwrite=True)
    return FILEPATH_ON_HDFS


def read_by_tensorflow(file: str) -> str:
    gfile = tf.io.gfile.GFile(file)
    return '\n'.join(gfile.readlines())


def check_hadoop_tensorflow() -> bool:
    success = True
    try:
        file = write_dummy_file_to_hdfs()
        success = read_by_tensorflow(file) == EXPECTED_CONTENT
    except Exception as exc:
        logger.warn('an exception occured accessing hdfs '
                    f'with tensorflow: {exc}')
        success = False
    if not success:
        logger.info(f'env variables: {os.environ}')
    return success


def launch_remote_check(file: str) -> Tuple[bool, str]:
    logging.info('Launching remote check')
    zip_hdfs, _ = cluster_pack.upload_env(packer=cluster_pack.PEX_PACKER)
    archive_name = os.path.basename(zip_hdfs)
    with skein.Client() as client:
        files = {
            archive_name: zip_hdfs,
            'check_hadoop_env.py': __file__,
        }
        editable_packages = cluster_pack.get_editable_requirements()
        if 'tf_yarn' in editable_packages:
            tf_yarn_zip = cluster_pack.zip_path(editable_packages['tf_yarn'], False)
            logger.info(f"zip path for editable tf_yarn is {tf_yarn_zip}")
            files.update({'tf_yarn': tf_yarn_zip})
        service = skein.Service(
            script=f'./{archive_name} check_hadoop_env.py --file {file}',
            resources=skein.Resources(2*1024, 1),
            env={
                'PEX_ROOT': '/tmp/{uuid.uuid4()}/',
                'PYTHONPATH': '.:',
            },
            files=files,
            instances=1
        )
        spec = skein.ApplicationSpec(
            {'HADOOP_ENV_CHECKER': service},
            acls=skein.model.ACLs(
                enable=True,
                view_users=['*']
            ),
        )
        app = client.submit_and_connect(spec)

        logging.info('Remote check started')
        result = app.kv.wait('result').decode()
        app_id = app.id
        app.shutdown()
        return result == "True", app_id


def check_env(vars: List[str] = []) -> bool:
    var_list = ["JAVA_HOME",
                "HADOOP_CONF_DIR",
                "HADOOP_HOME",
                "LD_LIBRARY_PATH"]
    var_list.extend(vars)
    results = [check_env_variable(var) for var in var_list]
    return all(results)


def check_env_variable(name: str) -> bool:
    if name not in os.environ:
        logger.warning(f"{name} variable not set")
        return False
    return True


def add_file_handler():
    if os.path.exists(RESULT_CHECK_FILE):
        os.remove(RESULT_CHECK_FILE)
    _logger = logging.getLogger()
    fh = logging.FileHandler(RESULT_CHECK_FILE)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    fh.setFormatter(formatter)
    _logger.addHandler(fh)
    logger.info(f'results will be written in {os.getcwd()}/{RESULT_CHECK_FILE}')


def main():
    parser = argparse.ArgumentParser(prog=sys.argv[0])
    parser.add_argument(
        '--file',
        help=argparse.SUPPRESS
    )

    args = parser.parse_args()

    if not args.file:
        logging.basicConfig(level=logging.INFO)
        add_file_handler()

        check_ok = check_env()
        logger.info(f'check environment variables: {check_ok}')

        if check_ok:
            check_ok = check_hadoop_tensorflow()
            logger.info(f'check_local_hadoop_tensorflow: {check_ok}')

            if check_ok:
                check_ok, app_id = launch_remote_check(FILEPATH_ON_HDFS)
                logger.info(f'remote_check: {check_ok}')
                if not check_ok:
                    time.sleep(5)
                    yarn_logs = subprocess.check_output([
                        "yarn",
                        "logs",
                        "-applicationId",
                        app_id]).decode()
                    logger.info(f"yarn logs: {yarn_logs}")

                tf.io.gfile.remove(FILEPATH_ON_HDFS)

        logger.info(f'Hadoop setup: {"OK" if check_ok else "KO"}')
    else:
        # check executed inside a yarn container
        result = False
        try:
            result = read_by_tensorflow(args.file) == EXPECTED_CONTENT
            logger.info(f'check_hadoop_tensorflow {result}')
            client = skein.ApplicationClient.from_current()
        finally:
            client.kv['result'] = str(result).encode()


if __name__ == "__main__":
    main()
