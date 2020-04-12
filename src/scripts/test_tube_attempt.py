from src.scripts.training import main

import yaml
from test_tube import SlurmCluster, HyperOptArgumentParser


def optimize_on_cluster(hparams):
    cluster = SlurmCluster(
        hyperparam_optimizer=hparams, log_path=hparams.slurm_log_path,
    )

    cluster.script_name = "-m src.scripts.test_tube_attempt"

    # email for cluster coms
    cluster.notify_job_status(
        email="mathieu.godbout.3@ulaval.ca", on_done=True, on_fail=True
    )

    # configure cluster
    cluster.per_experiment_nb_cpus = 4
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.job_time = "0:10:00"
    cluster.gpu_type = "p100"
    cluster.memory_mb_per_node = int(1e4)
    cluster.minutes_to_checkpoint_before_walltime = 1

    # any modules for code to run in env
    cluster.add_command("source ~/venvs/default/bin/activate")
    cluster.add_slurm_cmd(
        cmd="account", value="def-lulam50", comment="CCDB account for running"
    )

    cluster.optimize_parallel_cluster_gpu(
        main, nb_trials=1, job_name="rl_summarization"
    )


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)
    argument_parser = HyperOptArgumentParser(strategy="random_search")
    for config, value in base_configs.items():
        if type(value) is bool:
            # Hack as per https://stackoverflow.com/a/46951029
            argument_parser.add_argument(
                "--{}".format(config),
                type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
                default=value,
            )
        else:
            argument_parser.add_argument(
                "--{}".format(config), type=type(value), default=value
            )
    hparams = argument_parser.parse_args()
    optimize_on_cluster(hparams)
