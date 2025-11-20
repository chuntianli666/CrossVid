import json
from openai import OpenAI
import multiprocessing as mp
from tqdm import tqdm
import os

from utils.video2frames import process_video
from utils.extract import extract

# INFERENCE_EXTRACTABLE = """
# Provide you two cooking videos, which step in Video 2 is functionally equivalent to the step shown between {BEGIN}s and {END}s in Video 1?
# Timestamps of frames sampled from Video 1 are: {TIME1}.
# Timestamps of frames sampled from Video 2 are: {TIME2}.
# Watch the two videos carefully, and think about the question based on the information of the two videos.
# Output a time interval in seconds and separate the beginning and ending time with a comma.
# Wrap your answer within <answer></answer> tags, e.g., "15,23".
#
# Input frames:
# {FRAMES}
#
# Your answer:
# """

INFERENCE = """
Provide you two cooking videos, which step in Video 2 is functionally equivalent to the step shown between {BEGIN}s and {END}s in Video 1?
Timestamps of frames sampled from Video 1 are: {TIME1}.
Timestamps of frames sampled from Video 2 are: {TIME2}.
Watch the two videos carefully, and think about the question based on the information of the two videos.
Only output a time interval in seconds and separate the beginning and ending time with a comma, e.g., "15,23".

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
QA_path = "QA/FSA.json"
save_path = ""


def interval_iou(interval1, interval2):
    """
    Calculate intersection over union between two intervals.

    params:
        interval1 (tuple):  (start1, end1)
        interval2 (tuple):  (start2, end2)

    return:
        float: IOU，[0,1]
    """
    start1, end1 = interval1
    start2, end2 = interval2

    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)

    union = max(end1, end2) - min(start1, start2)
    if union == 0:
        return 0.0

    iou = intersection / union
    return iou


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
            videoA, videoB = pair["video A"], pair["video B"]
            pathA = os.path.join(video_root, videoA)
            pathB = os.path.join(video_root, videoB)
            frames_per_video = total_frames // 2
            frame_str = "".join(f"<frame>" for _ in range(frames_per_video))

            frames1, fps1, time_stamps1 = process_video(pathA, frames_per_video, length, encode=True)
            frames2, fps2, time_stamps2 = process_video(pathB, frames_per_video, length, encode=True)
            frames = frames1 + frames2
            frames_str = "Video 1:\n" + frame_str + "\nVideo 2:\n" + frame_str

            begin, end = pair["ref_segment"]
            prompt = INFERENCE.format(BEGIN=begin, END=end, TIME1=list(map(int, time_stamps1)), TIME2=list(map(int, time_stamps2)), FRAMES=frames_str)
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
                try:
                    answer1, answer2 = answer.split(",")
                    answer1, answer2 = float(answer1), float(answer2)
                    iou = interval_iou((answer1, answer2), (input_pair["answer"][0], input_pair["answer"][1]))
                    results.append({"id": input_pair["id"], "answer": [answer1, answer2], "iou": iou})
                except:
                    results.append({"id": input_pair["id"], "answer": answer, "iou": 0})
                
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

    sum_iou = sum([p["iou"] for p in results])
    print(f"The performance of {model} on task FSA is {sum_iou / len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run video analysis with specified parameters.')
    parser.add_argument('--model', type=str, help='Model to use for inference.')
    parser.add_argument('--frames', type=int, default=total_frames, help='Number of total frames per query.')
    parser.add_argument('--length', type=int, default=length, help='Resolution length of the frames.')
    parser.add_argument('--threads', type=int, default=threads, help='Number of threads.')
    parser.add_argument('--port', type=int, default=port, help='Port to send request.')
    parser.add_argument('--video_root', type=str, help='Video root path.', default='videos')
    parser.add_argument('--QA_path', type=str, help='QA file path.', default='QA/FSA.json')
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
