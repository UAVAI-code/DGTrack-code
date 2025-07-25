import os
import json


def rename_uavdt_folders_and_files(root_dir):
    for folder_name in os.listdir(root_dir):
        old_folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(old_folder_path):
            continue

        new_folder_name = None
        if folder_name.endswith("_aug1"):
            new_folder_name = folder_name.replace("_aug1", "_aug3")
        elif folder_name.endswith("_aug2"):
            new_folder_name = folder_name.replace("_aug2", "_aug4")
        else:
            continue

        meta_path = os.path.join(old_folder_path, "meta.json")
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                for tag in meta.get("tags", []):
                    if tag.get("name") == "sequence":
                        tag["value"] = new_folder_name
                        break

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"处理 {meta_path} 时出错: {str(e)}")
            continue

        old_att_path = os.path.join(old_folder_path, f"{folder_name}_att.txt")
        new_att_path = os.path.join(old_folder_path, f"{new_folder_name}_att.txt")

        if os.path.exists(old_att_path):
            try:
                os.rename(old_att_path, new_att_path)
            except Exception as e:
                print(f"重命名att文件时出错: {str(e)}")
                continue
        else:
            print(f"警告: 未找到文件 {old_att_path}")
            continue

        new_folder_path = os.path.join(root_dir, new_folder_name)
        try:
            os.rename(old_folder_path, new_folder_path)
            print(f"成功: {folder_name} -> {new_folder_name}")
        except Exception as e:
            print(f"重命名文件夹时出错: {str(e)}")


if __name__ == "__main__":
    root_directory = "/home/lzw/LEF/AVTrack_FADA/data/uavdt/sequences_aug_非小目标"

    rename_uavdt_folders_and_files(root_directory)
