#!/bin/bash

python cnn_dailymail.py --prompt code0 --model codex --max_prompt 4000 --key liam_personal
python cnn_dailymail.py --prompt code1 --model codex --max_prompt 4000 --key liam_personal
python cnn_dailymail.py --prompt code2 --model codex --max_prompt 4000 --key liam_personal
python cnn_dailymail.py --prompt code3 --model codex --max_prompt 4000 --key liam_personal