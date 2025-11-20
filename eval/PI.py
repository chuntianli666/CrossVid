import json
from openai import OpenAI
import multiprocessing as mp
from tqdm import tqdm
import os

from utils.interval2frames import process_video
from utils.extract import extract

# INFERENCE_EXTRACTABLE = """
# Provide you the beginning and the ending of a movie clip, what is most likely to happen in the middle?
# Watch the video segments carefully, and think about the question based on the context information.
# Select one answer choice, and output the capital letter of your choice within <answer></answer> tags.

# Options:
# {OPTIONS}

# Input frames:
# {FRAMES}

# Your answer:
# """

INFERENCE = """
Provide you the beginning and the ending of a movie clip, what is most likely to happen in the middle?
Watch the video segments carefully, and think about the question based on the context information.
Select one answer choice, and only output the capital letter of your choice.

Options:
{OPTIONS}

Input frames:
{FRAMES}

Your answer:
"""

model = ""
total_frames = 128
length = 360
threads = 20
port = 8000
video_root = "videos"
QA_path = "QA/PI.json"
save_path = ""

def chat(messages):
    client = OpenAI(
        api_key='<API-KEY>',
        base_url=f'http://0.0.0.0:{port}/v1'
    )
    # model = client.models.list().data[0].id
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


def inference(prompt, frames):
    prompt_splits = prompt.split('<frame>')
    contents = []
    for idx, split in enumerate(prompt_splits):
        if split: contents.append({"type": "text", "text": split})
        if idx < len(prompt_splits) - 1:
            contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frames[idx]}"}})
    messages = [
        {"role": "system", "content": "You are a helpful video analyzer."},
        {"role": "user", "content": contents}
    ]
    result = chat(messages)
    return result


def evaluate(pair, max_tries=3):
    for _ in range(max_tries):
        try:
            v = pair["video"]
            path = os.path.join(video_root, v)
            intervals = [pair["beginning"], pair["ending"]]
            interval_frames = process_video(path, total_frames, intervals, length, encode=True)[0]
            frames = []
            frames_str = "Beginning:\n"
            frames_str += "".join(f"<frame>" for _ in range(len(interval_frames[0])))
            frames.extend(interval_frames[0])

            frames_str += "Ending:\n"
            frames_str += "".join(f"<frame>" for _ in range(len(interval_frames[1])))
            frames.extend(interval_frames[1])
            
            prompt = INFERENCE.format(OPTIONS="\n".join(pair["options"]), FRAMES=frames_str)
            result = inference(prompt, frames)
            answer = result
            # If you use INFERENCE_EXTRACTABLE prompt, please extract answers
            # answer = extract(result, "answer")
            return pair, answer
        except:
            continue
    return None


def main():
    with open(QA_path, "r") as f:
        raw_pairs = json.load(f)
    answered_pairs = []
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            answered_pairs = json.load(f)
    answered_ids = set(item['id'] for item in answered_pairs)
    pairs = [item for item in raw_pairs if item.get('id') not in answered_ids]

    p = mp.Pool(threads, )
    pbar = tqdm(total=len(pairs), desc=f'Processing')
    results = answered_pairs
    num_suc, num_fail = 0, 0

    def update(x):
        nonlocal num_suc, num_fail
        if x is not None:
            try:
                input_pair, answer = x
                correct = answer == input_pair["answer"]
                results.append({"id": input_pair["id"], "answer": answer, "correct": correct})

                num_suc += 1
                if num_suc % 10 == 0:
                    with open(save_path, "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                pbar.set_description(f'✅ {model}: {num_suc}/{num_fail + num_suc}')
            except Exception as e:
                num_fail += 1
                pbar.set_description(f'❌ {model}: {num_suc}/{num_fail + num_suc}')
        else:
            num_fail += 1
            pbar.set_description(f'❌ {model}: {num_suc}/{num_fail + num_suc}')
        pbar.update(1)

    for pair in pairs:
        p.apply_async(
            func=evaluate,
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

    num_correct = len([p for p in results if p["correct"]])
    print(f"The performance of {model} on task PI is {num_correct / len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run video analysis with specified parameters.')
    parser.add_argument('--model', type=str, help='Model to use for inference.')
    parser.add_argument('--frames', type=int, default=total_frames, help='Number of total frames per query.')
    parser.add_argument('--length', type=int, default=length, help='Resolution length of the frames.')
    parser.add_argument('--threads', type=int, default=threads, help='Number of threads.')
    parser.add_argument('--port', type=int, default=port, help='Port to send request.')
    parser.add_argument('--video_root', type=str, help='Video root path.', default='videos')
    parser.add_argument('--QA_path', type=str, help='QA file path.', default='QA/PI.json')
    parser.add_argument('--save_path', type=str, help='File path to save the results.')
    args = parser.parse_args()


    model = args.model
    total_frames = args.frames
    length = args.length
    threads = args.threads
    port = args.port
    video_root = args.video_root
    QA_path = args.QA_path
    save_path = args.save_path

    main()
