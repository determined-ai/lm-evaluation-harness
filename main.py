import argparse
import fnmatch
import json
import logging
from typing import Any, Dict

import determined as det

from lm_eval import evaluator, tasks

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", default="")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)
    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main(core_context: det.core.Context, hparams: Dict[str, Any]):
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    trust_remote_code = hparams["model_args"].pop("trust_remote_code", False)
    model_args = f'trust_remote_code={trust_remote_code}'

    uuid = hparams["model_args"]["uuid"]
    if uuid is None:
        model_args += f',pretrained={hparams["model_args"]["pretrained_model_name_or_path"]}'

    # GG_NOTE: task will always be a single string, but it may be a glob-pattern which will get
    # converted to multiple tests.
    assert isinstance(hparams["task"], str)
    task_names = pattern_match([hparams["task"]], tasks.ALL_TASKS)
    assert task_names

    results = evaluator.simple_evaluate(
        model="hf",
        core_context=core_context,
        uuid=uuid,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_base_path=args.output_base_path,
    )

    all_metrics = {}
    assert len(results["results"]) == 1, "Each trial should execute one task only."

    for task_name, metrics in results["results"].items():
        for metric_name, value in metrics.items():
            all_metrics[f"{metric_name}"] = value

    # AC_NOTE: use trial_id as steps completed to scatter point-results
    # in the WebUI plot.
    trial_id = det.get_cluster_info().trial.trial_id
    core_context.train.report_validation_metrics(steps_completed=trial_id, metrics=all_metrics)

    dumped = json.dumps(results, indent=2)
    print(dumped)

    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))


if __name__ == "__main__":
    info = det.get_cluster_info()
    assert info
    hparams = info.trial.hparams
    logging.basicConfig(level=logging.INFO, format=det.LOG_FORMAT)
    try:
        distributed = det.core.DistributedContext.from_torch_distributed()
    except KeyError:
        distributed = None
    with det.core.init(distributed=distributed) as core_context:
        main(core_context, hparams)
