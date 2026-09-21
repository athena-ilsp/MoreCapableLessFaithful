## More Capable, Less Faithful: A Multilingual Analysis of Mathematical (Un)Solvability Detection in LLMs

Official implementation of ["More Capable, Less Faithful: A Multilingual Analysis of Mathematical (Un)Solvability Detection in LLMs"](https://arxiv.org/abs/2608.30463).


> [!WARNING]
> This codebase is currently under development and not yet ready for general use.


## Abstract

> Solvability detection is one of the most challenging aspects of mathematical reasoning for
Large Language Models (LLMs). While prior
work has studied this capability extensively,
these analyses have been limited to English.
Consequently, it remains unclear whether multilingual failures arise from differences in internal Solvability Belief or from languagedependent failures to express it. To address this
gap, we introduce the first multilingual benchmark of paired solvable and unsolvable mathematical problems, extending ReliableMath
to French and Greek. Using this, we train
multilingual probes predicting Solvability Belief and analyze the solvability detection capabilities of state-of-the-art LLMs behaviorally,
representationally, and in terms of faithfulness. We find that Solvability Belief is encoded
as a largely universal, language-agnostic feature, and that higher-resource languages such
as English, despite achieving stronger mathematical reasoning performance, exhibit lower
solvability-detection faithfulness.


## Installation

*To be updated.*

## Dataset

The file **`data/MultilingualReliableDataset.zip`** contains the multilingual extension of the **ReliableMath** dataset ([Xue et al., 2025](https://arxiv.org/abs/2507.03133)) introduced in our paper. It includes English (original), French, and Greek versions of both solvable and unsolvable mathematical problems. This dataset is released under CC BY 4.0 — see [data/LICENSE](data/LICENSE).

Contents:

```
solve_english_original.json
solve_french.json
solve_greek.json
unsol_english_original.json
unsol_french.json
unsol_greek.json
```

## Code

This repo contains the Python code that produced the paper's reported results:
dataset translation, chain-of-thought generation and hidden-state extraction,
LLM-as-a-judge verbalization annotation, and solvability-belief probing.


## Pipeline stages

### Benchmark Translation (translation/)

   - `translate.py` — local-model (vLLM) translation.
   - `translate_open_router.py` — translation via the OpenRouter API

### Chain of Thought generation and hidden state extraction (cot_generation/)

   - `cot_extract_hidden_states.py` — run a model over the (un)solvable splits
     under all prompts, saving per-token hidden states.
   - `judge.py` — LLM-as-a-judge annotation of each response's verbalized
     solvability verdict.
   - `compute_solvable_accuracy.py` — accuracy on the solvable split.


### Probe Training and Evaluation (probes/)

   - `train_probe.py` — trains per-language and pooled/universal probes.
   - `analyze_faithfulness.py` — agreement rate between the universal SB probe
     and the judge's textual verdict, per model/language.


## Citation

If you use this code or find our work useful, please cite:

```bibtex
@misc{zoumpoulidi2026capablefaithfulmultilingualanalysis,
      title={More Capable, Less Faithful: A Multilingual Analysis of Mathematical (Un)Solvability Detection in LLMs}, 
      author={Maria-Eleni Zoumpoulidi and Nikolaos Xiros and Georgios Paraskevopoulos},
      year={2026},
      eprint={2608.30463},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.30463}, 
}

```

## License

The code in this repository is licensed under the Apache 2.0 License — see [LICENSE](LICENSE) for details.

The dataset (`data/MultilingualReliableDataset.zip`) is a derivative of ReliableMath ([Xue et al., 2025](https://arxiv.org/abs/2507.03133)), which is released under CC BY 4.0; this dataset is likewise released under CC BY 4.0 with attribution to the original authors — see [data/LICENSE](data/LICENSE).


