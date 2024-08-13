from InstructorEmbedding import INSTRUCTOR
from openai import OpenAI
import numpy as np
import asyncio
import csv
import ast
import pandas as pd
import json

def embed(sentences):
    model = INSTRUCTOR('hkunlp/instructor-large')
    prompt = ""
    return model.encode([[prompt,sentences]])

def embed_oai(sentence, model="text-embedding-3-large"):

    client = OpenAI(api_key="")
    model="text-embedding-3-large"
    try:
        sentence = str(sentence)
        sentence = str(sentence.replace("\n", " "))
        response = client.embeddings.create(input=[sentence], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(e)
        return Exception
        # return client.embeddings.create(input=["0"], model=model).data[0].embedding

async def get_semantic_similarity(pred, target, model="text-embedding-3-large"):
    # step 1: transform y and y^ into embeddings
    pred_results = [embed_oai(p, model) for p in pred]
    target_results = [embed_oai(t, model) for t in target]
    z_hat = np.array(pred_results)
    z = np.array(target_results)
    
    # step 2: calculate recall
    max_sims = []
    for zi in z:
        max_i = max([np.dot(zi, zj) for zj in z_hat])
        max_sims.append(max_i)
    recall = np.average(max_sims)

    # step 3: calculate precision
    max_sims = []
    for zi in z_hat:
        max_i = max([np.dot(zi, zj) for zj in z])
        max_sims.append(max_i)
    precision = np.average(max_sims)

    # step 4: calculate f1 score
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_precision(pred, target, model="text-embedding-3-large"):
    # transform y and y^ into embeddings
    pred_tasks = [embed_oai(p, model) for p in pred]
    target_tasks = [embed_oai(t, model) for t in target]
    pred_results = pred_tasks
    target_results = target_tasks
    z_hat = np.array(pred_results)
    z = np.array(target_results)

    # calculate precision
    max_sims = []
    for zi in z_hat:
        max_i = max([np.dot(zi, zj) for zj in z])
        max_sims.append(max_i)
    precision = np.average(max_sims)

    return precision

def get_recall(pred, target, model="text-embedding-3-large"):
    # transform y and y^ into embeddings
    pred_results = [embed_oai(p, model) for p in pred]
    target_results = [embed_oai(t, model) for t in target]

    z_hat = np.array(pred_results)
    z = np.array(target_results)

    # calculate recall
    max_sims = []
    for zi in z:
        max_i = max([np.dot(zi, zj) for zj in z_hat])
        max_sims.append(max_i)
    recall = np.average(max_sims)

    return recall

# calculates the number of unique words / num total words
def get_TTR(sentence):
    words = sentence.split()
    num_unique = len(set(words))
    total = len(words)
    if total==0: return 0
    return num_unique/total

def get_hapax_richness(sentence):
    sentence = str(sentence)
    words = sentence.split()
    unique_words = set(words)
    hapax_words = [word for word in unique_words if words.count(word) == 1]
    total = len(words)
    if total==0: return 0
    return len(hapax_words)/total


def get_descriptiveness(pred, metric=get_hapax_richness): # average of TTR/Hapax for each sentence in pred
    return np.average([metric(p) for p in pred])


async def get_uniqueness(concepts, model="text-embedding-3-large"):
    if len(concepts) < 2: return len(concepts) # if 0 concepts, return 0. if 1 concept, return 1
    embeddings = [embed_oai(c, model) for c in concepts]
    # embeddings = await asyncio.gather(*embeddings)
    sims = []
    for i in range(len(embeddings)-1):
        # take max for each i and add to sims
        max_i = float('-inf')
        for j in range(i + 1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j])
            max_i = max(max_i, sim)
        sims.append(max_i)
    return 1-np.average(sims)

async def get_scores(pred, target):
    precision = get_precision(pred, target)
    recall = get_recall(pred, target)
    semantic_similarity = await get_semantic_similarity(pred, target)
    descriptiveness = get_descriptiveness(pred)
    uniqueness = await get_uniqueness(pred)
    return {"Semantic Similarity":semantic_similarity, "Precision":precision, "Recall":recall, "Descriptiveness":descriptiveness, "Uniqueness":uniqueness, "Num Concepts": len(pred)}


async def main(input_path, ground_truth_path, output_path):
    with open(input_path, 'r') as input_file:
        input_reader = csv.reader(input_file)
        input_concepts = []
        for row in input_reader:
            try:
                row = row[0]
                json_row = json.loads(row)
                concept = [item for item in json_row]

            except Exception as e:
                print(e)
                row = row[0]
                # concept = row.replace('\'', '\\\'')
                concept = [s.strip() for s in row[0].split(',') if s]
            input_concepts.append(concept)

    with open(ground_truth_path, 'r') as groundtruth_file:
        groundtruth_reader = csv.reader(groundtruth_file)
        groundtruth_concepts = []
        for row in groundtruth_reader:
            try:
                concept = ast.literal_eval(row[0])
            except:
                row[0].replace('\'', '\\\'')
            groundtruth_concepts.append(concept)
    print("read ground truth concepts")

    with open(output_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        for pred, target in zip(input_concepts, groundtruth_concepts):
            scores = await get_scores(pred, target)
            print("score:", scores)
            writer.writerow([scores])


# normalize time by the number of concepts
def normalize(row):
    try:
        concept = ast.literal_eval(row[0])
    except:
        concept = row[0].replace('\'', '\\\'')
    length = len(concept)
    if length == 0:
        return 0
    else:
        return row[1] / length
    

if __name__=="__main__":
    input_path = ""
    ground_truth_path = ""
    output_path = ""
    asyncio.run(main(input_path, ground_truth_path, output_path))
