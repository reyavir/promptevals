Large language models (LLMs) are being deployed as part of specialized production data processing pipelines across diverse domains---such as finance, marketing, and e-commerce. However, when running them in production across many inputs, they often fail to follow instructions or meet developer expectations. To improve reliability in these applications, creating assertions or guardrails for LLM outputs to run alongside the pipelines is essential. Yet, determining the right set of assertions that capture developer requirements for a task is challenging. We introduce PromptEvals, a benchmark of 2087 LLM pipeline prompts from different domains, associated with 12623 corresponding assertion criteria. The prompts in our benchmark are written by developers and engineers using our widely-adopted open-source LLM pipeline tools, and PromptEvals is more than $5\times$ larger than previous prompt collections. We evaluated both closed- and open-source models in generating relevant assertion criteria, finding that fine-tuned open-source models can offer reduced latency while providing better performance than closed-source models. We believe that PromptEvals can spur further research in the areas of LLM reliability, alignment, and prompt engineering.

Dataset: https://huggingface.co/datasets/user104/PromptHub

Fine-tuned Mistral 7b Model: https://huggingface.co/user104/promptevals_mistral

Fine-tuned Llama 3 8b Model: https://huggingface.co/user104/promptevals_llama

To generate ground truth criteria:
1. Generate initial criteria using GPT-4o
   Update the OpenAI key and input/output path variables in 'gpt4o-dataset.py' then run 'python gpt4o-dataset.py'
2. Update the OpenAI key and input/output path variables in 'generate_ground_truth.py' (change column index in line 71, depending on the format of your csv file), then run 'python generate_ground_truth.py'

To run the benchmark and calculate scores:
1. Generate criteria using your model and generate ground truth criteria, for all the prompt templates in the PromptEvals test set.
2. Update the input, output, and ground truth paths (and add your OpenAI key) in 'finetuning/src/evaluate_concepts.py' and run 'python finetuning/src/evaluate_concepts.py'
