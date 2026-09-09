# LLMorph: Metamorphic Testing of Large Language Models

[![DOI](https://zenodo.org/badge/1025446094.svg)](https://doi.org/10.5281/zenodo.16442703)

LLMorph is a tool to automatically test Large Language Models (LLMs) using Metamorphic Testing (MT), thorough their use on Natural Language Processing (NLP) tasks. It leverages the property-based nature MT to uncover faulty behaviours without the need for expensive labelled data. LLMorph is aimed at researchers and developers who want to evaluate the
robustness of LLM-based NLP systems.

This repository is the artifact for our ICSME'25 paper, [Metamorphic Testing of Large Language Models for Natural Language Processing](https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf).
This tool currently utilises Metamorphic Relations (MRs) extracted from academic literature on MT4NLP to test LLMs. 
Currently, LLMorph implements 36 out of the 191 MRs we collected from the literature. More details can be fuond in the mentioned paper.

Video demo: https://youtu.be/sHmqdieCfw4

## Requirements

**Python:** Python 3.11.15

**Dependencies:** Install using

```
pip install -r requirements.txt
```
and
```
nltk.download('punkt')

python -m spacy download en_core_web_trf

from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.')
```

An OpenAI key is needed in `security/token-key.jwt`.

## Usage

### Running the tool

The tool can be run through either Command Line Interface (CLI) or directly from a script. The latter pulls from a configuration file, giving more control.

#### CLI

To run the tool from command line, use

```
python src/main llm task mr input_data base_dir
```

with:

- `llm`: The name of the LLM to test.
- `task`: The name of the NLP task to test on.
- `mr`: The name of the metamorphic relation to test using.
- `input_data`: The path to the JSON file containing the inputs. Structured as an array of data points.
- `base_dir`: The path to the directory where caches and outputs will be stored. Outputs are found in `{base_dir}/results`.

Names of MRs and tasks can be found in `src/config/list_relations.json` and `src/config/list_tasks.json`, respectively.

#### Advanced Config

To run the tool using the advanced config, use

```
python src/mt_main.py
```

This will run the tool based on the configuration file found at `src/config/run_config.json`.

Default config values can be found in `src/run_config_defaults.json`.

### Results

Results are found in `{base_dir}/results` (with `{base_dir}` specified in the configuration; see above). It includes the LLM name, the task name, the metamorphic relation ID, the source and follow-up inputs, the source and follow-up outputs, and the output satisfactions. Results are saved after every relation tested.

### Running multiple data points concurrently

By default LLMorph processes one data point at a time. To have several in flight at once (e.g. splitting a large dataset across a few concurrent SUT/Hermes requests instead of one at a time), set `num_threads` in the config, or pass `-n`/`--num-threads` on the CLI:

```
python src/main.py --llm gemma-3-4b-it-q4_k_m --task question_answering --mr 51 -i data/data-example/source_inputs/data.json -o data/data-example -n 4
```

This parallelizes the SUT calls (`run_sut`), which are normally the dominant per-item cost and are always safe to run concurrently (an LLM call over the network). `input_transformation` and `output_relation` calls are serialized behind a lock instead: some implemented MRs wrap a local model (spaCy, nlpaug, KeyBERT, a SentenceTransformer's fast tokenizer, ...) that is not safe to call from multiple threads at once, and LLMorph has no generic way to tell which MRs do this and which only call an LLM (which would be safe to parallelize too) -- so all of them are serialized as a safe default. Checkpointing and output ordering are unaffected either way; you just get less speedup on MRs that don't use `run_sut` heavily.

If you know the specific MR you're running only calls an LLM in `input_transformation` (e.g. `ITGPT`/`ITGPTSentence`, which just go through the Hermes queue -- true of most `$mr$` config entries whose `func_it.class` ends in `GPT`), set `parallel_input_transformation: true` in the config, or pass `--parallel-input-transformation` on the CLI, to also run that step fully in parallel:

```
python src/main.py --llm gemma-3-4b-it-q4_k_m --task question_answering --mr 51 -i data/data-example/source_inputs/data.json -o data/data-example -n 4 --parallel-input-transformation
```

Leave this off (the default) for any MR using `ITNlpaug` or another local-model-based transformation -- turning it on there risks the same kind of crash (`RuntimeError: Already borrowed`, from a non-thread-safe tokenizer/model) that motivated serializing it by default in the first place. `output_relation` has no such opt-out yet; it is always serialized.

When testing against a local server, keep `num_threads` at or below however many requests it can actually run at once (e.g. llama.cpp's `--parallel`/`n_slots` setting) — going higher just queues requests behind the same bottleneck, and note that a single local model still shares one CPU/GPU across concurrent requests, so speedup from `num_threads` will usually be well short of linear.

### Hermes Queue (Redis)

Calls to Hermes (the `llm_for_transformation` model, used to transform inputs and compare outputs) are not made directly by the main LLMorph process. Instead, each request is pushed onto a Redis queue and a separate worker process picks it up, runs the inference, and sends the result back — this lets requests be handled asynchronously rather than blocking the tool for each call, and lets you scale throughput by running more than one worker. (Calls to the LLM under test, i.e. `llm_list`, are unaffected and still made directly.)

To use it:

1. Have a Redis server running (defaults to `localhost:6379`; configurable via `redis_host`, `redis_port`, `redis_db` in the run config).
2. Start one or more workers, from the repository root:
   ```
   python src/llm_worker.py
   ```
3. Run LLMorph as normal (CLI or `python src/mt_main.py`). Requests needing Hermes will be queued and served by the worker(s).

Other related config values (with defaults) in `src/config/run_config_default.json`: `hermes_queue_name` (the Redis queue name) and `hermes_response_timeout` (seconds a request waits for a worker's reply before retrying, reusing `llm_wait_time`/`llm_max_retries`).

### Changing LLMs

This project uses the `openai` python package to manage LLMs. To change the LLM under test, you can specify the relevant model name in `llm` perameter in the CLI; or, if using the config file, specifying the config value `llm_list` and the API endpoint in `llm_endpoint`. To use a different API, or to use a locally hosted LLM, plese modify `src/llm_runner.py`.

### Adding or modifying tasks

Tasks are currently specified via a zero-shot prompting procedure. To add or modify tasks, go to `src/config/template/sut_prompt_templates.json` to implement the prompt, and `src/config/list_tasks.json` to specify the particular task.

### Adding or modifying metamorphic relations

MRs are implemented as either functions or LLM prompts. To add or modify MRs, go to: `src/relations/func_it.py` and `src/relations/func_or.py` for the implementation of the input transformation and output relation, respectively; `src/config/template/it_prompt_templates.json` or `src/config/template/or_prompt_templates.json` if using a prompted LLM for transformation or comparison; and `src/config/list_relations.json` to specify the particular MR.

## Examples

### Basic Example

As an example on how to run the tool (and to test installation), use, for CLI:

```
python src/main gpt-4o-2024-08-06 question_answering 5 data/data-example/source_inputs/data.json data/data-example
```

This will test the LLM `gpt-4o-2024-08-06` on the `question_answering` task, using the MR with ID `5` (in this case, the "add random spaces" MR), on the single example input value found at `data/data-example/source_inputs/data.json`. Data will be generated in the `data/data-example` directory, with the final output in `data/data-example/results`.

### Datasets

Example datasets for each task are currently being pulled from HuggingFace. To download and clean, you can run 

```
python src/data_with_labels.py
```

Alternatively, set the config value `use_existing_source_inputs` to `false` to automatically download and use the datasets when the tool is run.

### Paper Reproduction

To reproduce the RQ1 data found in our paper, write the following configuration into `src/config/run_config.json`:

```
{
    "run_all": true,
    "llm_list": [
        "nous-hermes-2-mixtral-8x7b-dpo",
        "llama-3.1-70b-instruct",
        "gpt-4-1106"
    ],
    "llm_for_transformation": "nous-hermes-2-mixtral-8x7b-dpo",
    "use_existing_source_inputs": false,
    "dir_base_default": "data/data-reproduction"
}
```

Then, run

```
python src/mt_main.py
```

The results will be found in `data/data-reproduction/results`.

## Contribution

If you would like to contribute to this project by implementing new MRs or tasks, you may follow the instructions outlined above, then open a pull request. Any and all contributions are apprecated, for the furthering of the utility of this tool.

## Contact

If you have any questions, feel free to contact: steven.cho@aucklanduni.ac.nz

## Citation

```
@inproceedings{cho2025metamorphic,
  author = {Cho, Steven and Ruberto, Stefano and Terragni, Valerio},
  title = {Metamorphic Testing of Large Language Models for Natural Language Processing},
  booktitle = {Proceedings of the IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  year = {2025},
  publisher = {IEEE}
}
```
