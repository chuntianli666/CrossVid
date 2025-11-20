import json
import re

from openai import OpenAI
import multiprocessing as mp
from tqdm import tqdm
import os

from utils.extract import extract
from utils.frame_bbox import process_frames

# INFERENCE_EXTRACTABLE = """
# Provide you two synchronized UAV road recording videos, objects' positional information and a single-choice question.
# The positional information contains the bounding box coordinates ([xtl, ytl, xbr, ybr]) of the objects positioned in one appearing frame.
# Watch the videos first, then track the objects in both views and think about the question based on the information.
# Select one answer choice, and output the capital letter of the choice within <answer></answer> tags.

# Question:
# {QUESTION}

# Options:
# {OPTIONS}

# Objects information:
# {BBOX}

# Input frames:
# {FRAMES}

# Your answer:
# """

INFERENCE = """
Provide you two synchronized UAV road recording videos, objects' positional information and a single-choice question.
The positional information contains the bounding box coordinates ([xtl, ytl, xbr, ybr]) of the objects positioned in one appearing frame.
Watch the videos first, then track the objects in both views and think about the question based on the information.
Select one answer choice, and only output the capital letter of your choice.

Question:
{QUESTION}

Options:
{OPTIONS}

Objects information:
{BBOX}

Input frames:
{FRAMES}

Your answer:
"""

model = ""
total_frames = 128
length = 360
threads = 20
port = 8000
QA_path = "QA/MSR.json"
save_path = ""
video_root = "uav"

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


def extract_objects(s):
    results = re.findall(r'\{(.*?)\}', s)
    return results


def process(class_id, queried_obj, objects):
    view_dict = {1: "A", 2: "B"}
    processed_frames = []
    processed_bboxes = []
    obj_id = {}
    for view in [1, 2]:
        obj_info = {}
        for obj in objects:
            assert obj[0] in ["A", "B"] and obj[1] in ["1", "2", "3", "4", "5"]
            if obj[0] != view_dict[view]:
                continue
            idx = int(obj[1]) - 1
            sample_item = queried_obj[idx]
            with open(f"{video_root}/bbox/{view}/{class_id}.json", "r") as f:
                all_data = json.load(f)
            entity = None
            for d in all_data:
                if d["id"] == sample_item["id"]:
                    entity = d
            assert entity is not None
            obj_id[entity["id"]] = obj
            obj_info[entity["id"]] = entity["bbox"]
        frames_dir = f"{video_root}/frames/{view}/{class_id}-{view}"
        frames, bboxes = process_frames(frames_dir, total_frames // 2, obj_info, length,
                                        encode=True)
        formatted_bboxes = []
        for bbox in bboxes:
            temp = {}
            for k, v in bbox.items():
                temp[obj_id[k]] = v
            formatted_bboxes.append(temp)
        processed_frames.append(frames)
        processed_bboxes.append(formatted_bboxes)
    return processed_frames, processed_bboxes


def sample_bbox(bboxes):
    samples = {}
    for view_id, bbox_view in enumerate(bboxes):
        objects = bbox_view[0].keys()
        for obj in objects:
            appeared_frames = [i for i in range(len(bbox_view)) if bbox_view[i][obj] is not None]
            sample_frame = min(appeared_frames)
            view = "A" if view_id == 0 else "B"
            samples[obj] = {"view": view, "frame": sample_frame + 1, "bbox": bbox_view[sample_frame][obj]}
    return samples


def evaluate(pair, max_tries=3):
    for _ in range(max_tries):
        try:
            vid = pair["vid"]
            question = pair["question"]
            objects = extract_objects(question)
            object_name = {}
            for obj_id, obj in enumerate(objects):
                object_name[obj] = f"obj_{obj_id + 1}"

            queried_obj = pair["objects"]

            processed_frames, processed_bboxes = process(vid, queried_obj, objects)
            sample_bboxes = sample_bbox(processed_bboxes)

            obj_info = []
            for obj, bbox in sample_bboxes.items():
                bbox_info = [bbox['bbox']['xtl'], bbox['bbox']['ytl'], bbox['bbox']['xbr'], bbox['bbox']['ybr']]
                if bbox['frame'] == 1:
                    frame_no = "1st"
                elif bbox['frame'] == 2:
                    frame_no = "2nd"
                elif bbox['frame'] == 3:
                    frame_no = "3rd"
                else:
                    frame_no = f"{bbox['frame']}th"
                obj_info.append(
                    f"{object_name[obj]} appears in {frame_no} frame of view {bbox['view']} with bbox {bbox_info}")

            frames_per_video = total_frames // 2
            frame_str = "".join(f"<frame>" for _ in range(frames_per_video))
            frames_str = "View A:\n" + frame_str + "\nView B:\n" + frame_str
            formatted_question = question.format(**object_name)
            formatted_options = [option.format(**object_name) for option in pair["options"]]
            prompt = INFERENCE.format(QUESTION=formatted_question, OPTIONS="\n".join(formatted_options),
                                      FRAMES=frames_str, BBOX="\n".join(obj_info))
            result = inference(prompt, processed_frames[0] + processed_frames[1])
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
    print(f"The performance of {model} on task MSR is {num_correct / len(results)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run video analysis with specified parameters.')
    parser.add_argument('--model', type=str, help='Model to use for inference.')
    parser.add_argument('--frames', type=int, default=total_frames, help='Number of total frames per query.')
    parser.add_argument('--length', type=int, default=length, help='Resolution length of the frames.')
    parser.add_argument('--threads', type=int, default=threads, help='Number of threads.')
    parser.add_argument('--port', type=int, default=port, help='Port to send request.')
    parser.add_argument('--video_root', type=str, help='Video root path.', default='uav')
    parser.add_argument('--QA_path', type=str, help='QA file path.', default='QA/MSR.json')
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
