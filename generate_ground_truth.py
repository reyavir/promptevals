import csv
from openai import OpenAI
import re
import json
import time
OPENAI_API_KEY = ""


INITIAL_TEMPLATE = """Here is my prompt template:

```json
{prompt_template}
```

I want to write assertions for my LLM pipeline to run on all pipeline outputs. Here are some categories of constraints I may want the outputs to follow:

- **Structured Output**: Is there a requirement for the output to follow a standardized or custom format, such as markdown, HTML, or a JSON object?
- **Multiple Choice**: Does the output need to select from a predefined list of options?
- **Length Constraints**: Are there instructions regarding the targeted length of the output, such as the number of characters, words, or items in a list?
- **Semantic Constraints**:
  - **Excluding specific terms, items, or actions**: Are there terms, items, or actions that should be excluded from the output?
  - **Including or echoing specific terms or content**: Are there specific terms or content that should be included or echoed in the output?
  - **Covering or staying on a certain topic or domain**: Should the output cover or stay on a specific topic or domain?
  - **Following certain (code) grammar / dialect / context**: Are there requirements to follow certain (code) grammar, dialect, or context in the output?
- **Stylistic Constraints**: Are there requirements for the output to follow a certain style, tone, or persona?
- **Preventing Hallucination (Staying grounded and truthful)**: Should the output stay grounded and truthful, avoiding opinions, beliefs, or hallucinated outputs?
- **Preventing Hallucination (Adhering to Instructions without improvising unrequested actions)**: Should the output strictly adhere to any specific instructions provided, without including content that is not explicitly requested?

Give me a list of constraints to implement for verifying the quality of my LLM output. Each item in the list should contain a string description of a constraint to check for and its corresponding category. Category names are: structured_output, multiple_choice, length_constraints, exclude_terms, include_terms, stay_on_topic, follow_grammar, stylistic_constraints, stay_truthful, adhere_instructions.

Your answer should be a JSON list of objects within ```json ``` markers, where each object has the following fields: "constraint" and "category". Only include constraints that are explicitly mentioned in prompt template, and nothing else. You can return an empty list if no constraints are mentioned in the prompt template.
"""

PROMPT_TEMPLATE = """Here is my prompt template:

```json
{prompt_template}
```

I want to write assertions for my LLM pipeline to run on all pipeline outputs. Here are some categories of constraints I may want the outputs to follow:

- **Structured Output**: Is there a requirement for the output to follow a standardized or custom format, such as markdown, HTML, or a JSON object?
- **Multiple Choice**: Does the output need to select from a predefined list of options?
- **Length Constraints**: Are there instructions regarding the targeted length of the output, such as the number of characters, words, or items in a list?
- **Semantic Constraints**:
  - **Excluding specific terms, items, or actions**: Are there terms, items, or actions that should be excluded from the output?
  - **Including or echoing specific terms or content**: Are there specific terms or content that should be included or echoed in the output?
  - **Covering or staying on a certain topic or domain**: Should the output cover or stay on a specific topic or domain?
  - **Following certain (code) grammar / dialect / context**: Are there requirements to follow certain (code) grammar, dialect, or context in the output?
- **Stylistic Constraints**: Are there requirements for the output to follow a certain style, tone, or persona?
- **Preventing Hallucination (Staying grounded and truthful)**: Should the output stay grounded and truthful, avoiding opinions, beliefs, or hallucinated outputs?
- **Preventing Hallucination (Adhering to Instructions without improvising unrequested actions)**: Should the output strictly adhere to any specific instructions provided, without including content that is not explicitly requested?


Here are some assertion constraints I want the outputs to follow:
{constraints}

{step}
Return only your answer as a numbered list of strings.
"""

def update_constraints(csv_file, output_file, step):
    client = OpenAI(api_key=OPENAI_API_KEY)  
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file, open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(file)
        writer = csv.writer(outfile)
        for row in reader:
            if len(row) > 2:
                prompt_template = row[0]
                # initial_constraints = row[2]
                added_constraints = row[3] # TODO column idx needs to change per step
                
                formatted_prompt = PROMPT_TEMPLATE.format(prompt_template=prompt_template, constraints=added_constraints, step=step)
                message =  [{"role": "user", "content": formatted_prompt}]
                start = time.time()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages = message,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                ) 
                end = time.time()
                runtime = end - start
                
                assertion_constraints = response.choices[0].message.content.strip()
                row.append(assertion_constraints)
                row.append(runtime)
                writer.writerow(row)
        
def count_constraints(constraints, type):
    if type=="base":
        return len(constraints.split('|'))
    elif type=="llm":
        pattern = r'^\d+\.\s'
        matches = re.findall(pattern, constraints, re.MULTILINE)
        return len(matches)


def count_constraints_column():
    with open("added_constraints.csv", mode='r', newline='', encoding='utf-8') as file, open("count_constraints.csv", mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(file)
        writer = csv.writer(outfile)
        for row in reader:
            constraints = row[10]
            count = count_constraints(constraints, type="llm")
            row.append(count)
            writer.writerow(row)

def format_constraints(constraints_file, output_file):
        pattern = r'^\d+\.\s'
        constraints_list = []

        with open(constraints_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                input_data = row[0]
                json_list = []
                constraints = re.split(pattern, input_data, flags=re.MULTILINE)[1:]

                for constraint in constraints:
                    if '|' in constraint:
                        constraint_text, category = constraint.split('|', 1)
                        json_object = {
                            "constraint": constraint_text.strip(),
                            "category": category.strip()
                        }
                        json_list.append(json_object)
                    else:
                        print("constraint:", constraint)
                constraints_list.append(json_list)

        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in constraints_list:
                writer.writerow([json.dumps(row)])


add_step = "Add assertion constraints to the provided list. Add constraints that are stated in the prompt template and not already covered by an existing constraint. Return the combined list. Make sure the constraints are also followed by their corresponding categories"
remove_step = "Remove any assertion constraints that are incorrect, redundant (or already covered by another constraint), not relevant to the prompt template, or difficult to validate. Make sure the constraints are also followed by their corresponding categories"
# modify_step = "Modify any assertion constraints to be more unique, descriptive, aligned with the taxonomy, and/or relevant to the prompt template."


if __name__=="__main__":
    initial_criteria_path = ""
    groundtruth_criteria_path = ""
    update_constraints(initial_criteria_path, "added_constraints.csv", add_step)
    update_constraints("added_constraints.csv", groundtruth_criteria_path, remove_step)