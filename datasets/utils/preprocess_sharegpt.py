import json
from transformers import AutoTokenizer
from tqdm import tqdm

# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json?download=true
# ShareGPT format
#   {
#     "id": "XZWuTmr_0",
#     "conversations": [
#       {
#         "from": "human",
#         "value": "I have a set of images, each a scanned attendance sheet containing a list of handwritten student names. Each image is an attendance sheet from a specific day and we have about 20 of those. I want to write some code, using machine learning (or AI) techniques, to decode the names in each image so that I can create a data.frame, one row for each student, and each row having the following columns: the name of the student followed by their attendance status on each of the days (binary variables). I am comfortable with Python and PyTorch (and numpy, etc.)"
#       },
#       {
#         "from": "gpt",
#         "value": "To accomplish this task, you'll need to perform two main steps:\n\n1. Handwritten text recognition (HTR) to extract names from the scanned attendance sheets.\n2. Organize the extracted names and attendance information into a data frame.\n\nHere's a step-by-step guide to achieve this using Python and PyTorch:\n\n1. Install necessary packages:\n\nYou'll need to install pytesseract, pandas, and torchvision for this project:\n```bash\npip install pytesseract pandas torchvision\n```\n1. Import required libraries:\n```python\nimport os\nimport pytesseract\nfrom PIL import Image\nimport pandas as pd\nimport numpy as np\nfrom torchvision import transforms\n```\n1. Configure pytesseract:\n\nMake sure the tesseract binary is in your system's PATH or configure it like this:\n```python\n# For Windows\npytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n```\n1. Create a function to preprocess the image:\n```python\ndef preprocess_image(image_path):\n    image = Image.open(image_path)\n    transform = transforms.Compose([\n        transforms.Resize((256, 256)),\n        transforms.Grayscale(),\n        transforms.ToTensor(),\n        transforms.Normalize(mean=[0.5], std=[0.5]),\n    ])\n    return transform(image)\n```\n1. Create a function to extract names from the attendance sheet:\n```python\ndef extract_names(image_path):\n    image = preprocess_image(image_path)\n    text = pytesseract.image_to_string(image)\n    names = [line.strip() for line in text.split('\\n') if line.strip()]\n    return names\n```\n1. Process each attendance sheet and store the names and attendance status:\n```python\nattendance_sheets = ['path/to/sheet1.jpg', 'path/to/sheet2.jpg', ...] # Add paths to all attendance sheets\nattendance_dict = {}\n\nfor i, sheet_path in enumerate(attendance_sheets):\n    names = extract_names(sheet_path)\n    for name in names:\n        if name not in attendance_dict:\n            attendance_dict[name] = np.zeros(len(attendance_sheets), dtype=int)\n        attendance_dict[name][i] = 1\n```\n1. Create a pandas data frame with the attendance data:\n```python\nattendance_df = pd.DataFrame.from_dict(attendance_dict, orient='index', columns=[f'Day_{i+1}' for i in range(len(attendance_sheets))])\nattendance_df.insert(0, 'Student_Name', attendance_df.index)\nattendance_df.reset_index(drop=True, inplace=True)\n```\nNow you have the `attendance_df` data frame containing student names and their attendance status for each day. You can further manipulate or save the data frame as needed."
#       },
#       {
    
in_f = open("/root/pllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", "r")
data = json.load(in_f)

out_f = open("sharegpt.csv", "w")
out_f.write("TIMESTAMP,ContextTokens,GeneratedTokens\n")

tokenizer = AutoTokenizer.from_pretrained("lmsys/longchat-13b-16k")

num_samples = 100000
collected_samples = 0
for i in tqdm(range(len(data))):
    if len(data[i]["conversations"]) < 2:
        continue
    if data[i]["conversations"][0]["from"] != "human" or data[i]["conversations"][1]["from"] != "gpt":
        continue
    input_tokens = tokenizer.encode(data[i]["conversations"][0]["value"])
    output_tokens = tokenizer.encode(data[i]["conversations"][1]["value"])
    out_f.write(f"XXX,{len(input_tokens)},{len(output_tokens)}\n")
    collected_samples += 1
    if collected_samples >= num_samples:
        break

# ContextTokens      298.798425
# GeneratedTokens    323.726827
# dtype: float64