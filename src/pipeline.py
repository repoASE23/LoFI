import os, sys, re, time
import yaml

from utils import write_qa_results_to_csv
from config import config
from log_selection.selection_strategy import select_logs_for_all
from llm_inference.dataset import set_seed, prepare_squad_qa_data
from llm_inference.run_qa import train_eval_qa_model


def main():
    args = config.load_args()
    set_seed(args.seed+1)

    # # Step1: log selection
    if args.log_selection:
        print("[LSSS] Start log selection, method:", args.selection_method)
        args = select_logs_for_all(args, args.selection_method)
        print("[LSSS]", os.path.basename(args.train_filename), os.path.basename(args.dev_filename), os.path.basename(args.test_filename))
        # CUDA_VISIBLE_DEVICES=3 python pipeline.py --offline --log_selection --selection_method HighGroupSimilar-roberta --summarization_method no --log_window_length 3 --qa_ask_with_desc WithDesc
        # CUDA_VISIBLE_DEVICES=3 python pipeline.py --offline --log_selection --selection_method HighGroupSimilar-codebert --summarization_method no --log_window_length 3 --qa_ask_with_desc WithDesc
        # CUDA_VISIBLE_DEVICES=3 python pipeline.py --offline --log_selection --selection_method HighGroupSimilar-labse --summarization_method no --log_window_length 3 --qa_ask_with_desc WithDesc
        # CUDA_VISIBLE_DEVICES=3 python pipeline.py --offline --log_selection --selection_method HighGroupSimilar-unixcoder --summarization_method no --log_window_length 3 --qa_ask_with_desc WithDesc
        # python pipeline.py --offline --do_key_extraction --log_selection --selection_method Highest --summarization_method no --ad_method answer_second --log_second_window 11 --qa_ask_with_desc WithDesc
        # python pipeline.py --offline --do_key_extraction --log_selection --selection_method HighGroup --summarization_method no --ad_method answer_second --log_second_window 11 --qa_ask_with_desc WithDesc
        # python pipeline.py --offline --do_key_extraction --log_selection --selection_method Error --summarization_method no --ad_method answer_second --log_second_window 11 --qa_ask_with_desc WithDesc

    # # Step2: extract key information from the Log Window with Fewshot Learning
    if args.summarization_method == 'fewshot_qa':
        args = prepare_squad_qa_data(args)
        results = train_eval_qa_model(args)

        write_qa_results_to_csv(results, args)

        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)


if __name__ == "__main__":
    main()



