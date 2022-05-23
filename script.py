import sys
from action_clips import generate_action_clips

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Specify the raw video's filename (e.g. `python3 script.py raw_vid.MOV`)"
        )
    generate_action_clips(sys.argv[1])