import os
import shutil
from urllib.parse import unquote, urlparse

import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import get_file_name, get_file_size
from tqdm import tqdm

import src.settings as s


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    images_path = "/home/alex/DATASETS/TODO/HRF/all/images"
    manuals_path = "/home/alex/DATASETS/TODO/HRF/all/manual1"
    masks_path = "/home/alex/DATASETS/TODO/HRF/all/mask"
    batch_size = 30
    ds_name = "ds"
    masks_ext = "_mask.tif"
    manual_ext = ".tif"

    def create_ann(image_path):
        labels = []

        tag_meta = name_to_tag.get(get_file_name(image_path).split("_")[1])
        tag = sly.Tag(tag_meta)

        mask_path = os.path.join(masks_path, get_file_name(image_path) + masks_ext)
        mask_np = sly.imaging.image.read(mask_path)[:, :, 0]
        img_height = mask_np.shape[0]
        img_wight = mask_np.shape[1]
        obj_mask = mask_np == 255
        curr_bitmap = sly.Bitmap(obj_mask)
        curr_label = sly.Label(curr_bitmap, field)
        labels.append(curr_label)

        manual_path = os.path.join(manuals_path, get_file_name(image_path) + manual_ext)
        mask_np = sly.imaging.image.read(manual_path)[:, :, 0]
        obj_mask = mask_np == 255
        curr_bitmap = sly.Bitmap(obj_mask)
        curr_label = sly.Label(curr_bitmap, vessels)
        labels.append(curr_label)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[tag])

    vessels = sly.ObjClass("vessels", sly.Bitmap)
    field = sly.ObjClass("field of view", sly.Bitmap)

    healthy_meta = sly.TagMeta("healthy", sly.TagValueType.NONE)
    diabetic_meta = sly.TagMeta("diabetic retinopathy", sly.TagValueType.NONE)
    glaucomatous_meta = sly.TagMeta("glaucomatous", sly.TagValueType.NONE)

    name_to_tag = {"h": healthy_meta, "dr": diabetic_meta, "g": glaucomatous_meta}

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[vessels, field], tag_metas=[healthy_meta, diabetic_meta, glaucomatous_meta]
    )
    api.project.update_meta(project.id, meta.to_json())

    dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

    images_names = os.listdir(images_path)

    progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

    for img_names_batch in sly.batched(images_names, batch_size=batch_size):
        images_pathes_batch = [
            os.path.join(images_path, image_name) for image_name in img_names_batch
        ]

        img_infos = api.image.upload_paths(dataset.id, img_names_batch, images_pathes_batch)
        img_ids = [im_info.id for im_info in img_infos]

        anns_batch = [create_ann(image_path) for image_path in images_pathes_batch]
        api.annotation.upload_anns(img_ids, anns_batch)

        progress.iters_done_report(len(img_names_batch))

    return project
