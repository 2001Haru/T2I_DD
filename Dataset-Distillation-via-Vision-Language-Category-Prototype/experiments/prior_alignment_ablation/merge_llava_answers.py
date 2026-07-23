import argparse
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LLaVA answers into ImageFolder metadata.jsonl")
    parser.add_argument("--questions", required=True)
    parser.add_argument(
        "--answers",
        required=True,
        nargs="+",
        help="One or more LLaVA answer JSONL chunks",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Metadata exists: {output}. Pass --overwrite to replace it.")

    questions = {int(item["question_id"]): item for item in read_jsonl(args.questions)}
    answers = {}
    for answer_path in args.answers:
        for item in read_jsonl(answer_path):
            question_id = int(item["question_id"])
            if question_id in answers:
                raise ValueError(f"Duplicate answer for question_id={question_id}")
            answers[question_id] = str(item["text"]).strip().replace('"', "")

    missing = sorted(set(questions) - set(answers))
    extra = sorted(set(answers) - set(questions))
    if missing or extra:
        raise RuntimeError(
            f"Question/answer mismatch: {len(missing)} missing answers, {len(extra)} unknown answers"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for question_id in sorted(questions):
            record = {
                "file_name": questions[question_id]["image"],
                "text": answers[question_id],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"Wrote {len(questions)} image-text pairs to {output}")


if __name__ == "__main__":
    main()
