# Train & MIRFlickr25K
# 16 bits
python main.py --is-train --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 16
# 32 bits
python main.py --is-train --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 32
# 64 bits
python main.py --is-train --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 64

# Test & MIRFlickr25K
# 16 bits
python main.py --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 16 --pretrained=MODEL_PATH
# 32 bits
python main.py --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 32 --pretrained=MODEL_PATH
# 64 bits
python main.py --dataset flickr25k --query-num 2000 --train-num 10000 --result-name "RESULT_MATH_FLICKR" --k-bits 64 --pretrained=MODEL_PATH


# Train & NUSWIDE
# 16 bits
python main.py --is-train --dataset nuswide --query-num 2100 --train-num 10500 --result-name "RESULT_MATH_NUSWIDE" --k-bits 16
# 32 bits
python main.py --is-train --dataset nuswide --query-num 2100 --train-num 10500 --result-name "RESULT_MATH_NUSWIDE" --k-bits 32
# 64 bits
python main.py --is-train --dataset nuswide --query-num 2100 --train-num 10500 --result-name "RESULT_MATH_NUSWIDE" --k-bits 64

# Test & NUSWIDE
# 16 bits
python main.py --dataset nuswide --query-num 2100 --train-num 10500 --result-name "RESULT_MATH_NUSWIDE" --k-bits 16 --pretrained=MODEL_PATH
# 32 bits
python main.py --dataset nuswide --query-num 2100 --train-num 10500 --result-name "RESULT_MATH_NUSWIDE" --k-bits 32 --pretrained=MODEL_PATH
# 64 bits
python main.py --dataset nuswide --query-num 2100 --train-num 10500 --result-name "RESULT_MATH_NUSWIDE" --k-bits 64 --pretrained=MODEL_PATH


# Train & MS COCO
# 16 bits
python main.py --is-train --dataset coco --query-num 5000 --train-num 10000 --result-name "RESULT_MATH_COCO" --k-bits 16
# 32 bits
python main.py --is-train --dataset coco --query-num 5000 --train-num 10000 --result-name "RESULT_MATH_COCO" --k-bits 32
# 64 bits
python main.py --is-train --dataset coco --query-num 5000 --train-num 10000 --result-name "RESULT_MATH_COCO" --k-bits 64

# Test & MS COCO
# 16 bits
python main.py --dataset coco --query-num 5000 --train-num 10000 --result-name "RESULT_MATH_COCO" --k-bits 16 --pretrained=MODEL_PATH
# 32 bits
python main.py --dataset coco --query-num 5000 --train-num 10000 --result-name "RESULT_MATH_COCO" --k-bits 32 --pretrained=MODEL_PATH
# 64 bits
python main.py --dataset coco --query-num 5000 --train-num 10000 --result-name "RESULT_MATH_COCO" --k-bits 64 --pretrained=MODEL_PATH
