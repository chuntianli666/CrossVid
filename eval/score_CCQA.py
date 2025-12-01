import json

from openai import OpenAI
import multiprocessing as mp
from tqdm import tqdm

from utils.extract import extract


SCORE = """
You are asked to score the output of a model, given the following information:
- Question: {QUESTION}
- Standard Answer: {ANSWER}
- Scoring Points: {POINTS}
- Model's Output: {OUTPUT}

Please perform the following two-part scoring:
Part 1: Coverage of Scoring Points
- For each scoring point, determine whether it is covered by the Model's Output.
- Mark as covered (true) **only if** the scoring point is addressed **explicitly, independently, and clearly**.
- If the mention is vague, partial, or ambiguous, consider it **not covered**.

Part 2: Accuracy of Details
- For each covered scoring point, compare the details in the Model's Output to the Standard Answer.
- Mark as correct (true) **only if** the details are **fully accurate and consistent** with the Standard Answer, without any error, omission, or ambiguity.
- If the answer is partially correct, too broad/narrow, or not strictly consistent, mark it as **not correct** (false).
- For scoring points not covered, mark as incorrect.

Format your answer in a json format as follows:
{{
    "coverage": [true, false, true, ...],
    "correctness": [true, false, false, ...]
}}
The length of 'coverage' and 'correctness' lists should match the number of scoring points.
Wrap the json output within <score></score> tags.

Your answer:
"""

QA_path = "QA/CCQA.json"
answer_path = ""
save_path = ""


def chat(messages):
    client = OpenAI(
        api_key='<API-KEY>',
        base_url='http://0.0.0.0:8000/v1'
    )
    model = client.models.list().data[0].id
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=8192,
        temperature=0.0,
        top_p=0.95
    )
    result = completion.choices[0].message.content
    return result


def inference(pair):
    prompt = pair["prompt"]
    for _ in range(3):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        result = chat(messages)
        return pair, result
    return None

def prepare():
    inputs = []
    with open(QA_path, "r") as f:
        pairs = json.load(f)
    with open(answer_path, "r") as f:
        answers = json.load(f)
    for answer in answers:
        idx = int(answer["id"])
        pair = pairs[idx]
        assert pair["id"] == answer["id"]
        response = answer["answer"]
        prompt = SCORE.format(QUESTION=pair["question"], ANSWER=pair["answer"], POINTS=pair["scoring_points"], OUTPUT=response)
        inputs.append({
            "id": answer["id"],
            "prompt": prompt,
            "response": response,
        })
    return inputs


def main():
    inputs = prepare()

    p = mp.Pool(40, )
    pbar = tqdm(total=len(inputs), desc=f'Processing')
    results = []
    num_suc, num_fail = 0, 0

    def update(x):
        nonlocal num_suc, num_fail
        if x is not None:
            try:
                input_pair, response = x
                scoring = extract(response, "score")
                score = json.loads(scoring)
                results.append({
                    "id": input_pair["id"],
                    "answer": input_pair["response"],
                    "coverage": score["coverage"],
                    "correctness": score["correctness"],
                    "score": sum(score["coverage"] + score["correctness"]),
                })
                assert len(score["coverage"]) == len(score["correctness"])
                num_suc += 1
                if num_suc % 100 == 0:
                    with open(save_path, "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                pbar.set_description(f'✅ {num_suc}/{num_fail + num_suc}')
            except Exception as e:
                num_fail += 1
                pbar.set_description(f'❌ {num_suc}/{num_fail + num_suc}')
        else:
            num_fail += 1
            pbar.set_description(f'❌ {num_suc}/{num_fail + num_suc}')
        pbar.update(1)

    for pair in inputs:
        p.apply_async(
            func=inference,
            kwds={'pair': pair},
            callback=update,
            error_callback=lambda e: print(str(e)),
        )
    p.close()
    p.join()
    pbar.close()

    results = sorted(results, key=lambda x: x["id"])
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    sum_score = sum([p["score"] for p in results])
    all_score = sum([2 * len(p["coverage"]) for p in results])
    print(f"The performance on task CCQA is {sum_score / all_score}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Score the output of CCQA task')
    parser.add_argument('--QA_path', type=str, help='QA file path.', default='QA/CCQA.json')
    parser.add_argument('--answer_path', type=str, help='The json of the model output.')
    parser.add_argument('--save_path', type=str, help='File path to save the scoring results.')

    args = parser.parse_args()

    QA_path = args.QA_path
    answer_path = args.answer_path
    save_path = args.save_path

    main()

