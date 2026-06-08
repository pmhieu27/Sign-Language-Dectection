import os

person_map = {
    "Person_1": "person_01",
    "Person_2": "person_02",
    "Person_3": "person_03"
}

source_path = "datasets/Data mới"
targer_path = "datasets/dataser"

VIETNAMESE_TO_ENGLISH = {
    'có': 'yes',
    'cảm ơn': 'thanks',
    'hạnh phúc': 'happy',
    'không': 'no',
    'tôi': 'me',
    'tạm biệt': 'goodbye',
    'uống': 'drink',
    'xin chào': 'hello',
    'xin lỗi': 'sorry',
    'ăn': 'eat'
}

VIDEO_EXTENSIONS = {'.mov', '.mp4', '.avi'}

for person in sorted(os.listdir(source_path)):
    print(person)
    for label in sorted(os.listdir(os.path.join(source_path, person))):
        print(label)