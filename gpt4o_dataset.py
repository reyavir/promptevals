import asyncio
import time
from openai import OpenAI
import csv

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

# ADD_TEMPLATE = "Add assertion constraints to the provided list. Add constraints that are stated in the prompt template and not already covered by an existing constraint. Return the combined list. Make sure the constraints are also followed by their corresponding categories"
# REMOVE_TEMPLATE = "Remove any assertion constraints that are incorrect, redundant (or already covered by another constraint), not relevant to the prompt template, or difficult to validate. Make sure the constraints are also followed by their corresponding categories"


def generate_prompt(prompt_template):
    if len(prompt_template.strip()) == 0:
        return None
    message = INITIAL_TEMPLATE.format(prompt_template=prompt_template)
    return message


async def main(input_path, output_path):
    client = OpenAI(api_key=OPENAI_API_KEY)    
    with open(input_path, 'r') as file:
        reader = csv.reader(file)
        with open(output_path, 'a', newline='') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                prompt_template = row[0]
                prompt = generate_prompt(prompt_template)

                print("Sending new request:", prompt)
                message_text = [{"role": "user", "content": prompt}]
                start = time.time()
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages = message_text,
                    temperature=0.7,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )   
                reply = completion.choices[0].message.content
                end = time.time()
                diff = end-start
                if reply is None:
                    print("reply is None, skipping.")
                    concepts = ""
                else:
                    concepts = reply
                    print("CONCEPTS:", concepts)
                    print("\n\n")
                writer.writerow([concepts, diff])

if __name__=="__main__":
    input_path = "path to prompt templates"
    output_path = ""
    asyncio.run(main(input_path, output_path))