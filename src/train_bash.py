from llmtuner import run_exp

from mmengine.dist import init_dist
def main():
    init_dist('slurm', 'nccl', 'deepspeed')
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
