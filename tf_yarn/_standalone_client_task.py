from . import event
from tf_yarn._task_commons import _prepare_container, _process_arguments
from . import cluster


def main() -> None:
    task_type, task_id = cluster.get_task_description()
    client, cluster_spec, cluster_tasks = _prepare_container()
    cluster.setup_tf_config(cluster_spec)
    cluster.start_tf_server(cluster_spec)
    event.wait(client, "stop")


if __name__ == "__main__":
    _process_arguments()
    main()
