import json
from openai import OpenAI
import multiprocessing as mp
from tqdm import tqdm
import os

from utils.video2frames import process_video
from utils.extract import extract

# INFERENCE_EXTRACTABLE = """
# Provide you two cooking videos (Video A + Video B) and an open-ended question.
# Watch the videos carefully, and think about the question based on the information from both videos.
# Output your answer within <answer></answer> tags.

# Question:
# {QUESTION}

# Input frames:
# {FRAMES}

# Your answer:
# """


INFERENCE = """
Provide you two cooking videos (Video A + Video B) and an open-ended question.
Watch the videos carefully, and think about the question based on the information from both videos.

Question:
{QUESTION}

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
QA_path = "QA/CCQA.json"
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
            frames_per_video = total_frames // 2
            frame_str = "".join(f"<frame>" for _ in range(frames_per_video))
            frames_str = "Video A:\n" + frame_str + "\nVideo B:\n" + frame_str
            prompt = INFERENCE.format(QUESTION=pair["question"], FRAMES=frames_str)
            videos = [pair["video A"], pair["video B"]]
            frames = []
            for v in videos:
                path = os.path.join(video_root, v)
                frames.extend(process_video(path, frames_per_video, length, encode=True)[0])
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
                results.append({"id": input_pair["id"], "answer": answer})
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

    # For performance evaluation for open-ended task, please refer to score_CCQA.py


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run video analysis with specified parameters.')
    parser.add_argument('--model', type=str, help='Model to use for inference.')
    parser.add_argument('--frames', type=int, default=total_frames, help='Number of total frames per query.')
    parser.add_argument('--length', type=int, default=length, help='Resolution length of the frames.')
    parser.add_argument('--threads', type=int, default=threads, help='Number of threads.')
    parser.add_argument('--port', type=int, default=port, help='Port to send request.')
    parser.add_argument('--video_root', type=str, help='Video root path.', default='videos')
    parser.add_argument('--QA_path', type=str, help='QA file path.', default='QA/CCQA.json')
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
