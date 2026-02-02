import os, math, sys, json, tqdm, statistics, geopandas as gpd, pandas as pd, networkx as nx, shapely as shp
from loguru import logger
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.validation import explain_validity, make_valid

def assemble_coco_json(images, annotations, categories):
    """
    Assemble a COCO JSON dictionary from annotations, images and categories DataFrames.
    Args:
        images (DataFrame or list): Images DataFrame or record list containing the images info.
        annotations (DataFrame or list): Annotations DataFrame or record list containing the annotations info.
        categories (DataFrame or list): Categories DataFrame or record list containing the categories info.
    Returns:
        dict: A dictionary with the COCO JSON structure.
    """
    COCO_dict = {}
    for info_type, entry in {"images":images, "annotations":annotations, "categories":categories}.items():
        if isinstance(entry, pd.DataFrame):
            entry = json.loads(entry.to_json(orient="records"))
        elif not isinstance(entry, list):
            logger.critical(f"Entry {entry} is not a DataFrame or a list.")
            sys.exit(1)
        COCO_dict[info_type] = entry
    return COCO_dict

def find_category(df):
    """
    Ensures that the CATEGORY and SUPERCATEGORY columns are present in the input DataFrame.
    Args:
        df (pandas.DataFrame): DataFrame containing the GT labels.
    Returns:
        pandas.DataFrame: The input DataFrame with the CATEGORY and SUPERCATEGORY columns properly renamed.
    """
    if 'category' in df.columns:
        df.rename(columns={'category': 'CATEGORY'}, inplace = True)
    elif 'CATEGORY' not in df.columns:
        logger.critical('The GT labels have no category. Please produce a CATEGORY column when preparing the data.')
        sys.exit(1)

    if 'supercategory' in df.columns:
        df.rename(columns={'supercategory': 'SUPERCATEGORY'}, inplace = True)
    elif 'SUPERCATEGORY' not in df.columns:
        logger.critical('The GT labels have no supercategory. Please produce a SUPERCATEGORY column when preparing the data.')
        sys.exit(1)
    return df

def grid_over_tile(tile_size, tile_origin, pixel_size_x, pixel_size_y=None, max_dx=0, max_dy=0, grid_width=256, grid_height=256, crs='EPSG:2056', test_shape = None):
    """Create a grid over a tile and save it in a GeoDataFrame with each row representing a grid cell.
    Args:
        tile_size (tuple): tile width and height
        tile_origin (tuple): tile minimum coordinates
        pixel_size_x (float): size of the pixel in the x direction
        pixel_size_y (float, optional): size of the pixels in the y drection. If None, equals to pixel_size_x. Defaults to None.
        max_dx (int, optional): overlap in the x direction. Defaults to 0.
        max_dy (int, optional): overlap in the y direction. Defaults to 0.
        grid_width (int, optional): number of pixels in the width of one grid cell. Defaults to 256.
        grid_height (int, optional): number of pixels in the height of one grid cell. Defaults to 256.
        crs (str, optional): coordinate reference system. Defaults to 'EPSG:2056'.
        test_shape (shapely.geometry.base.BaseGeometry, optional): shape to test against for intersection. Defaults to None.
    Returns:
        GeoDataFrame: grid cells and their attributes
    """
    min_x, min_y = tile_origin
    tile_width, tile_height = tile_size
    number_cells_x = math.ceil((tile_width - max_dx)/(grid_width - max_dx))
    number_cells_y = math.ceil((tile_height - max_dy)/(grid_height - max_dy))

    # Convert dimensions from pixels to meters
    pixel_size_y = pixel_size_y if pixel_size_y else pixel_size_x
    grid_x_dim = grid_width * pixel_size_x
    grid_y_dim = grid_height * pixel_size_y
    max_dx_dim = max_dx * pixel_size_x
    max_dy_dim = max_dy * pixel_size_y

    # Create grid polygons
    polygons = []
    for x in range(number_cells_x):
        for y in range(number_cells_y):
            
            down_left = (min_x + x * (grid_x_dim - max_dx_dim), min_y + y * (grid_y_dim - max_dy_dim))

            # Fasten the process by not producing every single polygon
            if test_shape and not (test_shape.intersects(Point(down_left))):
                continue

            # Define the coordinates of the polygon vertices
            vertices = [down_left,
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + y * (grid_y_dim - max_dy_dim)),
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + (y + 1) * grid_y_dim - y * max_dy_dim),
                        (min_x + x * (grid_x_dim - max_dx_dim), min_y + (y + 1) * grid_y_dim - y * max_dy_dim)]

            # Create a Polygon object
            polygon = Polygon(vertices)
            polygons.append(polygon)

    # Create a GeoDataFrame from the polygons
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    grid_gdf['id'] = [f'{round(min_x)}, {round(min_y)}' for min_x, min_y in [(poly.bounds[0], poly.bounds[1]) for poly in grid_gdf.geometry]]
    return grid_gdf

def segmentation_to_polygon(segm):
    """
    Convert a COCO-style segmentation into a shapely Polygon or MultiPolygon.
    Args:
        segm (list): A list of lists where each sublist contains the x and y coordinates of the polygon's exterior in a flattened format suitable for COCO segmentation.
    Returns:
        shapely.Polygon or shapely.MultiPolygon: A shapely geometry object representing the polygon(s).
    Notes:
        COCO-style segmentation is a list of lists where each sublist contains the x and y coordinates of the polygon's exterior in a flattened format (e.g. [x1, y1, x2, y2, ...]).
        This function will return a Polygon or MultiPolygon depending on the number of polygons in the COCO-style segmentation.
        If the polygon is not valid (e.g. self-intersection), it will be made valid using shapely's make_valid function.
        If the polygon area is 0 or not valid, a warning message will be printed.
    """
    if len(segm)==1:
        if len(segm[0])<5:
            return Polygon()
        poly = Polygon(zip(segm[0][0::2], segm[0][1::2]))
    else:
        parts = []
        for coord_list in segm:
            if len(coord_list)<5:
                    continue
            parts.append(Polygon(zip(coord_list[0::2], coord_list[1::2])))
        if len(parts)==0:
            return Polygon()
        poly = MultiPolygon(parts) if len(parts)>1 else parts[0]

    if not poly.is_valid and 'Self-intersection' in explain_validity(poly):
        valid_poly = make_valid(poly)
        if isinstance(valid_poly, GeometryCollection):
            tmp_list = []
            for multi_geom in valid_poly.geoms:
                if isinstance(multi_geom, MultiPolygon):
                    tmp_list.extend([geom for geom in multi_geom.geoms])
                else:
                    tmp_list.extend([multi_geom])
            poly = MultiPolygon([geom for geom in tmp_list if isinstance(geom, Polygon)])
            if poly.is_empty:
                poly = MultiPolygon([geom for geom in valid_poly.geoms if isinstance(geom, Polygon)])
        else:
            poly = valid_poly

    if poly.area == 0:
        logger.warning(f"Polygon area is 0: {poly}")
    elif not poly.is_valid:
        logger.warning(f"Polygon is not valid: {poly}")
    return poly


def read_image_info(coco_file_path_dict, id_correspondence_df):
    """
    Reads image information from multiple COCO JSON files and merges it with an ID correspondence DataFrame.
    Args:
        coco_file_path_dict (dict): A dictionary where keys are dataset identifiers and values are paths to
                                    COCO JSON files containing image data.
        id_correspondence_df (DataFrame): A DataFrame containing ID correspondence information with columns
                                          for 'dataset', 'original_id', and 'image_id'.
    Returns:
        DataFrame: A DataFrame containing combined image information from all specified COCO JSON files,
                   including a 'basename' column with the base filenames.
    """
    images_df = pd.DataFrame()
    for aoi_key, coco_file in coco_file_path_dict.items():
        with open(coco_file) as fp:
            coco_data = json.load(fp)['images']
        tmp_df = pd.DataFrame(coco_data)
        tmp_df = tmp_df.merge(
            id_correspondence_df[id_correspondence_df.AOI==aoi_key], 
            how='left', left_on='id', right_on='original_id'
        ).drop(columns=['original_id','id'])
        images_df = pd.concat((images_df, tmp_df), ignore_index=True)
    images_df['basename'] = images_df.file_name.apply(lambda x: os.path.basename(x))
    return images_df


def group_annotations(transformed_detections_gdf, verbose=False):
    """
    Groups overlapping annotations in the transformed detections GeoDataFrame.
    This function performs a spatial self-join on the `transformed_detections_gdf` to find 
    overlapping annotation pairs that belong to the same image and dataset. It then groups 
    these overlapping annotations using a graph-based approach, assigning a unique group 
    ID to each connected component of overlapping annotations.
    Args:
        transformed_detections_gdf (GeoDataFrame): A GeoDataFrame containing transformed 
        detections with geometry information.
        verbose (bool, optional): Whether to display progress bars. Defaults to False.
    Returns:
        DataFrame: A DataFrame with the same columns as `transformed_detections_gdf`, 
        but with an additional column 'group_id' indicating the group number each 
        annotation belongs to.
    """
    def assign_groups(row, group_index):
        row['group_id'] = group_index[row['geohash_left']] if row['geohash_left'] in group_index else None
        return row
    
    # Find overlapping pairs
    overlapping_dets_gdf = gpd.GeoDataFrame()
    for image_id in tqdm.tqdm(transformed_detections_gdf.image_id.unique(), desc="Find overlapping pairs", disable=(not verbose)):
        subset_gdf = transformed_detections_gdf[transformed_detections_gdf.image_id==image_id]
        self_join = gpd.sjoin(subset_gdf, subset_gdf, how='inner')
        overlapping_dets_gdf = pd.concat([
            overlapping_dets_gdf,
            self_join[
                (self_join['id_left'] <= self_join['id_right'])
                & (self_join['dataset_left'] == self_join['dataset_right'])
            ]
        ], ignore_index=True)
    # Do groups because of segmentation on more than two tiles
    g = nx.Graph()
    [g.add_edge(row.geohash_left, row.geohash_right) for row in overlapping_dets_gdf[overlapping_dets_gdf.geohash_left.notnull()].itertuples()]
    groups = list(nx.connected_components(g))
    group_index = {node: i for i, group in enumerate(groups) for node in group}
    return overlapping_dets_gdf.apply(lambda row: assign_groups(row, group_index), axis=1)

def make_new_annotation(group,groupped_pairs_df, buffer=1):
    """
    Creates a new annotation based on a group of overlapping detections.
    Args:
        group (int): The group number of the overlapping detections.
        groupped_pairs_df (DataFrame): A DataFrame containing the overlapping pairs
            of detections, with group number assigned to each pair.
        buffer (int, optional): The buffer size to subtract from the new geometry.
            Defaults to 1.
    Returns:
        dict: A dictionary containing the new annotation information.
    """
    def polygon_to_segmentation(multipolygon):
        """
        Convert a shapely Polygon or MultiPolygon into a COCO-style segmentation.
        Args:
            polygon (Polygon or MultiPolygon): A shapely geometry object representing the polygon(s).
        Returns:
            list: A list of lists where each sublist contains the x and y coordinates of the polygon's exterior in 
                a flattened format suitable for COCO segmentation.
        """
        segmentation, polygon_coords = [], []
        if isinstance(multipolygon, Polygon):
            multipolygon = MultiPolygon([multipolygon])

        for poly in multipolygon.geoms:
            exterior_coords = poly.exterior.coords.xy # Missing inner rings if polygon has holes
            for coord_index in range(len(exterior_coords[0])):
                polygon_coords.append(exterior_coords[0][coord_index])
                polygon_coords.append(exterior_coords[1][coord_index])
        segmentation.append(polygon_coords)
        return segmentation

    group_dets = groupped_pairs_df[groupped_pairs_df.group_id==group].copy()

    # Keep lowest id, median score. Calculate new segmentation, area and bbox
    new_geometry = shp.unary_union(
        pd.concat([group_dets.buffered_geometry, gpd.GeoSeries(group_dets.geohash_right.apply(shp.from_wkb))]).drop_duplicates()
    ).buffer(-buffer)
    new_segmentation = polygon_to_segmentation(new_geometry)
    bbox = new_geometry.bounds
    
    ann_dict = {
        'id': int(group_dets.id_left.min()),
        'image_id': int(group_dets.image_id_left.iloc[0]),
        'category_id': int(group_dets.category_id_left.iloc[0]),
        'dataset': group_dets.dataset_left.iloc[0],
        'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
        'segmentation': new_segmentation,
        'area': new_geometry.area,
    }

    if 'score_left' in group_dets.columns:
        # Todo: ensure each score appear only once
        ann_dict['score'] = statistics.median(group_dets.score_left.tolist() + group_dets.score_right.tolist())
    return ann_dict

def transform_annotations(tile_name, annotations_df, images_df, buffer=1, id_field='id', category_field='category_id'):
    """
    Transform COCO annotations on a tile to their original image and pixel coordinates.
    Args:
        tile_name (str): The name of the tile, including the directory and extension.
        annotations_df (DataFrame): A DataFrame containing the COCO annotations for the tile.
        images_df (DataFrame): A DataFrame containing information about the original images.
        buffer (int, optional): The amount of pixels to buffer the geometry of each annotation. Defaults to 1.
        id_field (str, optional): The name of the column in annotations_df containing the annotation ID. Defaults to 'id'.
        category_field (str, optional): The name of the column in annotations_df containing the category ID. Defaults to 'category_id'.
    Returns:
        list: A list of dictionaries, each representing an annotation on the original image.
    Raises:
        ValueError: If no image is found with the same name as the tile.
        ValueError: If multiple images are found with the same name.
    """
    name_parts = tile_name.rstrip('.jpg').split('_')
    original_name = os.path.basename('_'.join(name_parts[:-2]) + '.jpg')
    tile_origin_x, tile_origin_y = int(name_parts[-2]), int(name_parts[-1])

    corresponding_images = images_df.loc[images_df.basename==original_name, 'image_id']

    if len(corresponding_images) == 1:
        image_id = corresponding_images.iloc[0]
    elif len(corresponding_images) > 1:
        raise ValueError(f"Multiple images with the same name: {original_name}")
    else:
        raise ValueError(f"No image with the name: {original_name}")
    
    annotations_on_tiles_df = annotations_df[annotations_df.file_name==tile_name].copy()
    annotations_on_tiles_list = []
    for ann in annotations_on_tiles_df.itertuples():
        ann_segmentation = []
        for poly in ann.segmentation:
            poly_coordinates = []
            for coor_id in range(0, len(poly), 2):
                poly_coordinates.append(poly[coor_id] + tile_origin_x)
                poly_coordinates.append(poly[coor_id + 1] + tile_origin_y)
            ann_segmentation.append(poly_coordinates)
        # Buffer geometry to facilitate overlap in the next step
        buffered_geom = segmentation_to_polygon(ann_segmentation).buffer(buffer)
        annotations_on_tiles_list.append({
            'id': getattr(ann, id_field),
            'image_id': image_id,
            'category_id': getattr(ann, category_field),
            'dataset': ann.dataset,
            'segmentation': ann_segmentation,
            'buffered_geometry': buffered_geom,
            'geohash': shp.to_wkb(buffered_geom)
        })

        if 'score' in annotations_on_tiles_df.columns:
            annotations_on_tiles_list[-1]['score'] = round(ann.score, 3)
    return annotations_on_tiles_list