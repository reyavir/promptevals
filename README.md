
To generate ground truth criteria:
1. Generate initial criteria using GPT-4o
   Update the OpenAI key and input/output path variables in 'gpt4o-dataset.py' then run 'python gpt4o-dataset.py'
2. Update the OpenAI key and input/output path variables in 'generate_ground_truth.py' (change column index in line 71, depending on the format of your csv file), then run 'python generate_ground_truth.py'

To run the benchmark and calculate scores:
1. Generate criteria using your model and generate ground truth criteria, for all the prompt templates in the PromptEvals test set.
2. Update the input, output, and ground truth paths (and add your OpenAI key) in 'finetuning/src/evaluate_concepts.py' and run 'python finetuning/src/evaluate_concepts.py'
