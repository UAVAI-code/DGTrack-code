import os
import json

base_dir = '/home/lzw/LEF/AVTrack_FADA_OAMix_TwoPipline/data/uavdt/sequences_aug(小目标增强参数提高)'

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)

    if os.path.isdir(folder_path):
        old_txt_file = os.path.join(folder_path, f"{folder_name.split('_')[0]}_aug_att.txt")
        new_txt_file = os.path.join(folder_path, f"{folder_name.split('_')[0]}_aug1_att.txt")

        if os.path.exists(old_txt_file):
            os.rename(old_txt_file, new_txt_file)
            print(f"Renamed: {old_txt_file} -> {new_txt_file}")
        else:
            print(f"File not found: {old_txt_file}")

        json_file = os.path.join(folder_path, 'meta.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

            for tag in meta_data['tags']:
                if tag['name'] == 'sequence':
                    tag['value'] = f"{folder_name.split('_')[0]}_aug1"

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=4, ensure_ascii=False)
            print(f"Updated: {json_file}")
        else:
            print(f"File not found: {json_file}")

print("All done!")
