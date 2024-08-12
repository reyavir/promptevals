import csv
from openai import OpenAI
OPENAI_API_KEY = ""

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
        next(reader)
        for row in reader:
            if len(row) > 2:
                prompt_template = row[0]
                initial_constraints = row[8] # TODO column idx needs to change per step
                
                formatted_prompt = PROMPT_TEMPLATE.format(prompt_template=prompt_template, constraints=initial_constraints, step=step)
                message =  [{"role": "user", "content": formatted_prompt}]
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
                
                assertion_constraints = response.choices[0].message.content.strip()
                row.append(assertion_constraints)
                writer.writerow(row)


add_step = "Add assertion constraints to the provided list. Add constraints that are relevant to the prompt template and not already covered by an existing constraint. Return the combined list"
remove_step = "Remove any assertion constraints that are incorrect, redundant (or already covered by another constraint), not relevant to the prompt template, or difficult to validate."
modify_step = "Modify any assertion constraints to be more unique, descriptive, aligned with the taxonomy, and/or relevant to the prompt template."


if __name__=="__main__":
    # update_constraints("prompthub_initial.csv", "added_constraints.csv", add_step)
    # update_constraints("added_constraints.csv", "removed_constraints.csv", remove_step)
    update_constraints("removed_constraints.csv", "final_constraints.csv", modify_step)
