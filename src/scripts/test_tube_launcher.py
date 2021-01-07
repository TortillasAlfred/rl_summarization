from src.scripts.training import main

import yaml
from test_tube import SlurmCluster, HyperOptArgumentParser


def optimize_on_cluster(hparams):
    cluster = SlurmCluster(
        hyperparam_optimizer=hparams, log_path=hparams.slurm_log_path,
    )

    cluster.script_name = "-um src.scripts.test_tube_launcher"

    # email for cluster coms
    cluster.notify_job_status(
        email="mathieu.godbout.3@ulaval.ca", on_done=True, on_fail=True
    )

    # configure cluster
    cluster.per_experiment_nb_cpus = 8
    cluster.per_experiment_nb_gpus = 1
    cluster.per_experiment_nb_nodes = 1
    cluster.job_time = "2-00:00:00"
    cluster.gpu_type = "p100"
    cluster.memory_mb_per_node = 64000
    cluster.minutes_to_checkpoint_before_walltime = 2

    # any modules for code to run in env
    cluster.add_command(
        "cp /scratch/magod/summarization_datasets/cnn_dailymail/tarred/sit_dataset.tar $SLURM_TMPDIR/"
    )
    cluster.add_command("tar -xf $SLURM_TMPDIR/sit_dataset.tar -C $SLURM_TMPDIR/")
    cluster.add_command("source ~/venvs/default/bin/activate")
    cluster.add_slurm_cmd(
        cmd="account", value="def-adurand", comment="CCDB account for running"
    )

    cluster.optimize_parallel_cluster_gpu(
        main, nb_trials=15, job_name="rl_summarization"
    )


if __name__ == "__main__":
    base_configs = yaml.load(open("./configs/base.yaml"), Loader=yaml.FullLoader)
    argument_parser = HyperOptArgumentParser(strategy="random_search")

    fine_tuned_items = {}
    fine_tuned_items["n_repeats_per_sample"] = dict(
        default=16, type=int, tunable=True, options=[8, 16, 32],
    )
    fine_tuned_items["seed"] = dict(
        default=1, type=int, tunable=True, options=list(range(5))
    )

    for config, value in base_configs.items():
        if config not in fine_tuned_items:
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

    for name, kwargs in fine_tuned_items.items():
        argument_parser.opt_list(f"--{name}", **kwargs)

    hparams = argument_parser.parse_args()
    optimize_on_cluster(hparams)
