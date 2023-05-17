# Exploring the Curious Case of Code Prompts

## What is this work about

Recent work has shown that prompting language models with code-like representations of natural language leads to performance improvements on structured reasoning tasks. However, such tasks comprise only a small subset of all natural language tasks. In our work, we seek to answer whether or not code-prompting is the preferred way of interacting with language models in general. We compare code and text prompts across three popular GPT models (`davinci`, `code-davinci-002`, and `text-davinci-002`) on a broader selection of tasks (e.g., QA, sentiment, summarization) and find that with few exceptions, code prompts do not consistently outperform text prompts. Furthermore, we show that the style of code prompt has a large effect on performance for some but not all tasks and that fine-tuning on text instructions leads to better relative performance of code prompts.

## Dependency
The conda environment `/envs` should contain everything needed. If not, please install libraries as prompted. 

## Organization
Each task has a corresponding folder in `/datasets`. Take HellaSWAG for example: in `/datasets/HellaSWAG`: running
> python hellaswag.py --prompt text|code1|code2|code3|code4 --model codex|davinci --max_prompt MAX_TOKENS
produces an output file of a model (`code-davinci-002` or `text-davinci-002`) with a prompt format (text prompt or one of our 4 code prompts) and a context window (2000, 4000, or 8000 tokens). Additionally, it also performs evaluation which can be used to replicate Figure 3, 4 and Table 2 in the paper.

## Citation
If you find our work useful, please cite
```
@misc{zhang2023exploring,
      title={Exploring the Curious Case of Code Prompts}, 
      author={Li Zhang and Liam Dugan and Hainiu Xu and Chris Callison-Burch},
      year={2023},
      eprint={2304.13250},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```