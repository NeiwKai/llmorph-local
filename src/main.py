from mt_main import run_using_config
import argparse


def main():
    parser = argparse.ArgumentParser(prog="PROG", prefix_chars="-", 
                                     description="LLMorph: A framework for testing LLMs with metamorphic relations.",
                                     epilog="Paper: https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf"
                                    )
    parser.add_argument("-l", "--llm", required=True, type=str, help="The name of the LLM to test.")
    parser.add_argument("-t", "--task", required=True, type=str, help="The name of the NLP task to test on.")
    parser.add_argument("-m", "--mr", required=True, type=str, help="The id of the metamorphic relation to test using.")
    parser.add_argument("-i", "--input-data", required=True, type=str, help="The path to the JSON file containing the inputs. Structured as an array of data points.")
    parser.add_argument("-o", "--base-dir", required=True, type=str, help="The path to the directory where caches and outputs will be stored.")
    parser.add_argument("-r", "--replace-perc", required=False, type=float, metavar="PERCENT", nargs="?", default=0.1, help="The ratio value for generating follow up inputs (range: 0.0 - 1.0, default: 0.1).")

    args = parser.parse_args()

    config = {
        "tasks": {args.task: [args["--mr"]]},
        "llm_list": [args["--llm"]],
        "existing_source_inputs": args["--input-data"],
        "dir_base_default": args["--base-dir"],
        "replace_perc": args["--replace-perc"],
    }

    run_using_config(config)

if __name__ == "__main__":
    main()
