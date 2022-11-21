# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2022. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

import logging
from tempfile import TemporaryDirectory

import numpy as np
import sys
import os

from glob import glob
from PIL import Image
from shapely.geometry import Polygon, Point
from shapely import wkt
from shapely.affinity import affine_transform
from tifffile import imread

from csbdeep.utils import normalize
from stardist.models import StarDist2D

from cytomine import CytomineJob
from cytomine.models import Annotation, AnnotationCollection, ImageInstanceCollection, Job
from sldc_cytomine.dump import dump_region

__author__ = "Maree Raphael <raphael.maree@uliege.be>"


MAX_REGION_AREA = int(1e8) # equivalent to a square region of 10k x 10k pixels


def main(argv):
    with CytomineJob.from_cli(argv) as conn:
        app_logger = conn.logger
        conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization...")

        # Loading pre-trained Stardist model
        np.random.seed(17)

        # use local model file in ~/models/2D_versatile_HE/
        model = StarDist2D(None, name='2D_versatile_HE', basedir='/models/')

        # Select images to process
        images = ImageInstanceCollection().fetch_with_filter(
            "project",
            conn.parameters.cytomine_id_project
        )

        if conn.parameters.cytomine_id_images == 'all':
            list_imgs = [int(image.id) for image in images]
        else:
            list_imgs = [int(id_img) for id_img in conn.parameters.cytomine_id_images.split(',')]

        # Go over images
        for id_image in conn.monitor(list_imgs, prefix="Running detection on image", period=0.1):
            # Dump ROI annotations in img from Cytomine server to local images
            annotation_params = {
                "project": conn.parameters.cytomine_id_project,
                "term": conn.parameters.cytomine_id_roi_term,
                "image": id_image,
                "showWKT": True
            }
            roi_user_annotations = AnnotationCollection(**annotation_params).fetch()
            roi_algo_annotations = AnnotationCollection(**annotation_params, includeAlgo=True).fetch()
            roi_annotations = roi_user_annotations + roi_algo_annotations
            app_logger.debug(roi_annotations)

            # Go over ROI in this image
            for roi in roi_annotations:
                # Get Cytomine ROI coordinates for remapping to whole-slide
                # Cytomine cartesian coordinate system, (0,0) is bottom left corner
                app_logger.debug("----------------------------ROI------------------------------")
                roi_geometry = wkt.loads(roi.location)
                if roi_geometry.area >= MAX_REGION_AREA:
                    app_logger.error(f"ROI from annotation #{roi.id} is too large, skipping to avoid out of memory error.")
                    app_logger.error("Crop area should not be larger 10k x 10k pixels.")
                    continue

                app_logger.debug(f"ROI Geometry from Shapely: {roi_geometry}")
                app_logger.info(f"ROI {roi.id} bounds {roi_geometry.bounds}")

                minx, miny = roi_geometry.bounds[0], roi_geometry.bounds[3]

                # Dump ROI image into local Tiff file, all downloaded files will be deleted after use  
                with TemporaryDirectory() as tmpdir:
                    roi_filepath = os.path.join(tmpdir, f'{roi.id}.tif')
                    tiles_path = os.path.join(tmpdir, 'tiles')
                    app_logger.debug(f"roi_png_filename: {roi_filepath}")
                    dump_region(roi, roi_filepath, working_path=tiles_path)

                    # Processing ROI
                    app_logger.info(f"-- Processing ROI file : {roi_filepath}")
                    img = normalize(
                        imread(roi_filepath),
                        conn.parameters.stardist_norm_perc_low,
                        conn.parameters.stardist_norm_perc_high,
                        axis=(0, 1)  # normalize channels independently
                    )
                    # Stardist model prediction with thresholds
                    _, details = model.predict_instances(
                        img,
                        prob_thresh=conn.parameters.stardist_prob_t,
                        nms_thresh=conn.parameters.stardist_nms_t,
                        n_tiles=model._guess_n_tiles(img)
                    )
                
                app_logger.info(f"Number of detected polygons: {len(details['coord'])}")

                cytomine_annotations = AnnotationCollection()
                # Go over detections in this ROI, convert and upload to Cytomine
                for polygroup in details['coord']:
                    # Converting to Shapely annotation
                    annotation = Polygon(np.vstack(polygroup[::-1]).transpose())
                    # Cytomine cartesian coordinate system, (0,0) is bottom left corner
                    # Mapping Stardist polygon detection coordinates to Cytomine ROI in whole slide image
                    affine_matrix = [1, 0, 0, -1, minx, miny]
                    annotation = affine_transform(annotation, affine_matrix)
                    if not roi_geometry.intersects(annotation): 
                        continue

                    cytomine_annotations.append(
                        Annotation(
                            location=annotation.wkt,
                            id_image=id_image,  # conn.parameters.cytomine_id_image,
                            id_project=conn.parameters.cytomine_id_project,
                            id_terms=[conn.parameters.cytomine_id_cell_term]
                        )
                    )
                
                app_logger.info("upload annotation")
                # Send Annotation Collection (for this ROI) to Cytomine server in batch
                cytomine_annotations.save(chunk=50)

        conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")


if __name__ == "__main__":
    main(sys.argv[1:])
