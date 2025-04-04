import os
import zipfile
import tarfile

def extract_folder_from_archive(archive_path, folder_in_archive, output_dir):
    """
    从压缩包中解压指定文件夹及其所有内容到目标目录。
    
    支持 .zip, .tar, .tar.gz, .tgz

    :param archive_path: 压缩包路径
    :param folder_in_archive: 要解压的文件夹路径，例如 'data/images/'
    :param output_dir: 解压输出目录
    """
    print(f"Archive exists: {os.path.exists(archive_path)}")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            print("Zip contents:")
            for member in zip_ref.namelist():
                print(member)
                if member.startswith(folder_in_archive):
                    zip_ref.extract(member, output_dir)
                    print(f"已将 '{member}' 解压到 '{output_dir}'")

    elif archive_path.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            print("Tar contents:")
            for member in tar_ref.getmembers():
                print(member.name)
                if member.name.startswith(folder_in_archive):
                    tar_ref.extract(member, output_dir)
                    print(f"已将 '{member.name}' 解压到 '{output_dir}'")
    else:
        raise ValueError("Unsupported archive format. Use .zip or .tar(.gz/.tgz)")


root = "/Volumes/zhan/DATASET/HOI4D"
# root = "/media/ps/zhan/DATASET/HOI4D"
# task_name = "ZY20210800001/H1/C13/N17/S179/s05/T4"
# task_name = "ZY20210800001/H1/C1/N19/S100/s02/T1"
# task_name = "ZY20210800001/H1/C1/N28/S100/s03/T2"
# task_name = "ZY20210800004/H4/C3/N66/S380/s01/T1"
task_name = "ZY20210800004/H4/C14/N30/S335/s02/T2"

depth_video_zip_path = os.path.join(root, "HOI4D_depth_video.tar")
depth_video_path = os.path.join("HOI4D_depth_video", task_name)
extract_folder_from_archive(depth_video_zip_path, depth_video_path, root)

annotion_zip_path = os.path.join(root, "HOI4D_annotations.zip")
annotion_path = os.path.join("HOI4D_annotations", task_name)
extract_folder_from_archive(annotion_zip_path, annotion_path, root)