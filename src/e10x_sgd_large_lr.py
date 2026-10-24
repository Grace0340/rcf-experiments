"""E10x: E10 with plain SGD at learning rates above the E10 grid. Everything else as in E10."""
import e10_optimizer as e10

e10.RUNS = [("sgd", lr) for lr in [3.0, 10.0, 30.0]]
_log = e10.log_run


def log_run(rec):
    rec["exp"] = "E10x"
    _log(rec)


e10.log_run = log_run

if __name__ == "__main__":
    e10.main()
