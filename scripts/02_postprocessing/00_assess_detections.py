import os, sys, math, json, pandas as pd, numpy as np, geopandas as gpd, tqdm, time, plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

sys.path.insert(1, 'scripts')
from utils import misc
from utils import metrics
from utils.logging import get_logger
from utils.config import get_config  

logger = get_logger()
cfg = get_config("This script assesses the quality of detections with respect to ground-truth/other labels and computes the best threshold for the detection.")

ID_CLASSES = [0]

def load_ground_truth(limit_aoi):
    logger.info("Loading ground truth...")
    categories_info_df = pd.DataFrame(cfg['categories'][0], index=[0]).rename(columns={'id': 'label_class', 'name': 'category'})
    tiles_df_dict, labels_gdf_dict, label_segm_df_dict = {}, {}, {}
    nbr_tiles, nbr_labels = 0, 0
    for dataset, labels_file in {"trn":"trn.json", "val":"val.json", "tst":"tst.json"}.items():
        with open(os.path.join(cfg['datasets']['path_ground_truth'], labels_file)) as fp:
            coco_dict = json.load(fp)
        labels_df = pd.DataFrame.from_records(coco_dict['annotations'])

        # Get annotation geometry
        labels_df['geometry'] = labels_df['segmentation'].apply(lambda x: misc.segmentation_to_polygon(x))
        no_geom_tmp = labels_df[labels_df.geometry.isna()]
        if no_geom_tmp.shape[0] > 0:
            logger.warning(f"{no_geom_tmp.shape[0]} labels have no geometry in the {dataset} dataset with a max score of {round(no_geom_tmp['score'].max(), 2)}.")

        # Get annotation class
        labels_df.rename(columns={'category_id': 'label_class', 'id': 'label_id'}, inplace=True)
        labels_df = labels_df.merge(categories_info_df, on='label_class', how='left')
        label_segm_df_dict[dataset] = labels_df[['label_id', 'segmentation', 'bbox']].rename(columns={
            'segmentation': 'segmentation_labels', 'bbox': 'bbox_labels'
        })
        labels_df.drop(columns=['segmentation', 'iscrowd', 'supercategory', 'bbox'], inplace=True, errors='ignore')

        # Format tile info
        all_aoi_tiles_df = pd.DataFrame.from_records(coco_dict['images']).rename(columns={'id': 'image_id'})
        all_aoi_tiles_df['file_name'] = [os.path.basename(path) for path in all_aoi_tiles_df['file_name']]
        if limit_aoi:
            all_aoi_tiles_df = all_aoi_tiles_df[all_aoi_tiles_df['AOI'] == limit_aoi]
            labels_df = pd.merge(labels_df, all_aoi_tiles_df[['image_id']], how='inner', on='image_id')

        tiles_df_dict[dataset] = all_aoi_tiles_df.copy()
        labels_gdf_dict[dataset] = gpd.GeoDataFrame(labels_df)

        nbr_tiles += len(all_aoi_tiles_df)
        nbr_labels += len(labels_df)
    logger.success(f"{nbr_tiles} tiles were found and {nbr_labels} labels were found.")
    return categories_info_df, tiles_df_dict, labels_gdf_dict, label_segm_df_dict

def load_detections(tiles_df_dict, limit_aoi):
    logger.info("Loading detections...")
    dets_gdf_dict, det_segm_df_dict = {}, {}
    nbr_dets = 0
    for dataset, dets_file in cfg['datasets']['detections_files'].items():
        with open(dets_file) as fp:
            dets_dict = json.load(fp)
        if isinstance(dets_dict, dict):
            dets_dict = dets_dict['annotations']
        dets_df = pd.DataFrame.from_records(dets_dict)

        # Format detection info
        if 'image_id' not in dets_df.columns:
            dets_df = pd.merge(dets_df, tiles_df_dict[dataset][['file_name', 'image_id']], how='inner', on='file_name')
        elif limit_aoi:
            dets_df = pd.merge(dets_df, tiles_df_dict[dataset][['image_id']], how='inner', on='image_id')
        if 'det_id' not in dets_df.columns:
            dets_df.rename(columns={'id': 'det_id', 'category_id': 'det_class'}, inplace=True)
        dets_df['geometry'] = dets_df['segmentation'].apply(lambda x: misc.segmentation_to_polygon(x))
        no_geom_condition = dets_df.geometry.isna()
        
        if no_geom_condition.sum() > 0:
            logger.warning(f"{no_geom_condition.shape[0]} detections have no geometry in the {dataset} dataset with a max score of {round(no_geom_condition['score'].max(), 2)}.")
            dets_df = dets_df[~no_geom_condition]

        det_segm_df_dict[dataset] = dets_df[['det_id', 'segmentation', 'bbox']].rename(columns={'segmentation': 'segmentation_dets', 'bbox': 'bbox_dets'})
        dets_df.drop(columns=['segmentation', 'bbox'], inplace=True)

        dets_gdf_dict[dataset] = gpd.GeoDataFrame(dets_df)
        logger.info(f"{len(dets_gdf_dict[dataset])} detections were found in the {dataset} dataset.")
        nbr_dets += len(dets_gdf_dict[dataset])
    logger.success(f"{nbr_dets} detections were found.")
    return dets_gdf_dict, det_segm_df_dict

def compute_validation_metrics(dets_gdf_dict, labels_gdf_dict, iou_threshold, method, datasets_list=['val']):
    metrics_cl_df_dict, metrics_df_dict = {}, {}
    written_files = []
    logger.info("Computing validation metrics...")
    thresholds = np.arange(round(dets_gdf_dict['val'].score.min()*2, 1)/2, 1., 0.05)
    
    # ------ Comparing detections with ground-truth data and computing metrics
    outer_tqdm_log = tqdm.tqdm(total=len(datasets_list), position=0)
    for dataset in datasets_list:
        outer_tqdm_log.set_description_str(f'Current dataset: {dataset}')
        inner_tqdm_log = tqdm.tqdm(total=len(thresholds), position=1, leave=False)
        cur_dets_gdf_dict = dets_gdf_dict[dataset]
        metrics_dict = []
        for threshold in thresholds:
            threshold = round(threshold, 2)
            inner_tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

            tagged_df_dict = metrics.get_fractional_sets(
                cur_dets_gdf_dict[cur_dets_gdf_dict.score >= threshold], labels_gdf_dict[dataset], dataset, iou_threshold
            )
            tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(id_classes=ID_CLASSES, method=method, **tagged_df_dict)

            metrics_dict.append({'threshold':threshold, 'precision':precision, 'recall':recall, 'f1':f1})

            metrics_cl_df_dict[dataset] = pd.DataFrame.from_records([{
                'threshold': threshold,
                'class': id_class,
                'precision_k': p_k[id_class],
                'recall_k': r_k[id_class],
                'TP_k' : tp_k[id_class],
                'FP_k' : fp_k[id_class],
                'FN_k' : fn_k[id_class],
            } for id_class in ID_CLASSES])
            inner_tqdm_log.update(1)

        metrics_df_dict[dataset] = pd.DataFrame.from_records(metrics_dict)
        outer_tqdm_log.update(1)
    inner_tqdm_log.close()
    outer_tqdm_log.close()

    fig = go.Figure()
    for dataset in datasets_list:
        fig.add_trace(go.Scatter(
            x=metrics_df_dict[dataset]['recall'],
            y=metrics_df_dict[dataset]['precision'],
            mode=cfg['scatter_plot_mode'],
            text=metrics_df_dict[dataset]['threshold'], 
            name=dataset
        ))
    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        xaxis=dict(range=[0., 1]), yaxis=dict(range=[0., 1])
    )
    file_to_write = os.path.join(cfg['output_folder'], 'precision_vs_recall.html')
    fig.write_html(file_to_write)
    written_files.append(file_to_write)

    for dataset in datasets_list:
        # Generate a plot of TP, FN and FP for each class
        fig = go.Figure()
        for id_class in ID_CLASSES:
            for y in ['TP_k', 'FN_k', 'FP_k']:
                fig.add_trace(go.Scatter(
                    x=metrics_cl_df_dict[dataset]['threshold'][metrics_cl_df_dict[dataset]['class']==id_class],
                    y=metrics_cl_df_dict[dataset][y][metrics_cl_df_dict[dataset]['class']==id_class],
                    mode=cfg['scatter_plot_mode'],
                    name=f"{y[0:2]}_{id_class}"
                ))
            fig.update_layout(xaxis_title="threshold", yaxis_title="#")
            
        file_to_write = os.path.join(cfg['output_folder'], f'{dataset}_TP-FN-FP_vs_threshold{"_dep_on_class" if len(ID_CLASSES) > 1 else ""}.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)

        fig = go.Figure()
        for y in ['precision', 'recall', 'f1']:
            fig.add_trace(go.Scatter(
                x=metrics_df_dict[dataset]['threshold'],
                y=metrics_df_dict[dataset][y],
                mode=cfg['scatter_plot_mode'],
                name=y
            ))
        fig.update_layout(xaxis_title="threshold")
        file_to_write = os.path.join(cfg['output_folder'], f'{dataset}_metrics_vs_threshold.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)
    return metrics_df_dict, written_files

def main():
    CONFIDENCE_THRESHOLD = cfg['confidence_threshold'] if 'confidence_threshold' in cfg.keys() else None
    IOU_THRESHOLD = cfg['iou_threshold'] if 'iou_threshold' in cfg.keys() else 0.25
    METHOD = cfg['metrics_method'] if 'metrics_method' in cfg.keys() else 'macro-average'
    LIMIT_AOI = cfg['limit_aoi'] if 'limit_aoi' in cfg.keys() else False

    os.makedirs(cfg['output_folder'], exist_ok=True)
    
    written_files = []
    if LIMIT_AOI:
        logger.info(f"Limiting dataset to {LIMIT_AOI}...")
    
    categories_info_df, tiles_df_dict, labels_gdf_dict, label_segm_df_dict = load_ground_truth(LIMIT_AOI)

    dets_gdf_dict, det_segm_df_dict = load_detections(tiles_df_dict, LIMIT_AOI)

    metrics_df_dict = {}
    if 'val' in dets_gdf_dict.keys():
        metrics_df_dict, new_written_files = compute_validation_metrics(dets_gdf_dict, labels_gdf_dict, IOU_THRESHOLD, METHOD)
        written_files = written_files + new_written_files

    logger.info("Tagging detections...")
    # we select the threshold which maximizes the f1-score on the val dataset or the one passed by the user
    if 'val' in metrics_df_dict.keys() and CONFIDENCE_THRESHOLD:
        logger.warning('The confidence threshold was determined over the val dataset, but a confidence threshold is given in the config file.')
        logger.warning(f'confidence threshold: val dataset = {metrics_df_dict["val"].loc[metrics_df_dict["val"]["f1"].argmax(), "threshold"]}, config = {CONFIDENCE_THRESHOLD}')
        logger.warning('The confidence threshold from the config file is used.')
    
    if CONFIDENCE_THRESHOLD:
        selected_threshold = CONFIDENCE_THRESHOLD
        logger.info(f"Tagging detections with threshold = {selected_threshold:.2f}, which is the threshold given in the config file.")
    elif 'val' in metrics_df_dict.keys():
        selected_threshold = metrics_df_dict['val'].loc[metrics_df_dict['val']['f1'].argmax(), 'threshold']
        logger.info(f"Tagging detections with threshold = {selected_threshold:.2f}, which maximizes the f1-score on the val dataset.")
    else:
        raise AttributeError('No confidence threshold can be determined without the validation dataset or the passed value.')

    logger.info(f'Method to compute the metrics : {METHOD}')

    global_metrics_dict = {'dataset': [], 'precision': [], 'recall': [], 'f1': []}
    metrics_cl_df_dict, tagged_dets_gdf_dict = {}, {}
    for dataset in cfg['datasets']['detections_files'].keys():
        tmp_dets_gdf = dets_gdf_dict[dataset][dets_gdf_dict[dataset].score >= selected_threshold].copy()
        logger.info(f'Number of detections = {len(tmp_dets_gdf)}')
        logger.info(f'Number of labels = {len(labels_gdf_dict[dataset])}')

        tagged_df_dict = metrics.get_fractional_sets(tmp_dets_gdf, labels_gdf_dict[dataset], dataset, IOU_THRESHOLD)

        tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(id_classes=ID_CLASSES, method=METHOD, **tagged_df_dict)
        global_metrics_dict['dataset'].append(dataset)
        global_metrics_dict['precision'].append(precision)
        global_metrics_dict['recall'].append(recall)
        global_metrics_dict['f1'].append(f1)
        logger.info(f'Dataset = {dataset} => precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

        # label classes starting at 1 and detection classes starting at 0.
        metrics_cl_df_dict[dataset] = pd.DataFrame.from_records([{
                'threshold': selected_threshold,
                'class': id_cl,
                'precision_k': p_k[id_cl],
                'recall_k': r_k[id_cl],
                'TP_k' : tp_k[id_cl],
                'FP_k' : fp_k[id_cl],
                'FN_k' : fn_k[id_cl],
            } for id_cl in ID_CLASSES])

        tagged_df_dict["fn_df"] = tagged_df_dict["fn_df"].merge(label_segm_df_dict[dataset], how='left', on='label_id').rename(
            columns={'segmentation_labels': 'segmentation', 'bbox_labels': 'bbox'}
        )
        for key in ['tp_df', 'fp_df']:
            tagged_df_dict[key] = tagged_df_dict[key].merge(det_segm_df_dict[dataset], how='left', on='det_id').rename(
                columns={'segmentation_dets': 'segmentation', 'bbox_dets': 'bbox'}
            )
        tagged_dets_gdf_dict[dataset] = pd.concat(tagged_df_dict.values())

    tagged_dets_df = pd.concat([tagged_dets_gdf_dict[x] for x in cfg['datasets']['detections_files'].keys()])
    tagged_dets_df['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'category'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in tagged_dets_df.det_class
    ]


    tagged_detections_file = os.path.join(cfg['output_folder'], 'tagged_detections.json')
    with open(tagged_detections_file, 'w') as fp:
        tagged_dets_df[[
            'dataset', 'tag', 'label_id', 'label_class', 'category', 'det_id', 'score', 'det_class', 'det_category', 'area', 'IOU', 'segmentation', 'bbox', 'image_id'
        ]].to_json(fp, orient='records')
    written_files.append(tagged_detections_file)


    # Save the metrics by class for each dataset
    metrics_by_cl_df = pd.DataFrame()
    for dataset in metrics_cl_df_dict.keys():
        dataset_df = metrics_cl_df_dict[dataset].copy()
        dataset_df['dataset'] = dataset
        dataset_df.drop(columns=['threshold'], inplace=True)
        metrics_by_cl_df = pd.concat([metrics_by_cl_df, dataset_df], ignore_index=True)
    
    metrics_by_cl_df['category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'category'].iloc[0] 
        for det_class in metrics_by_cl_df['class'].to_numpy()
    ] 

    file_to_write = os.path.join(cfg['output_folder'], 'metrics_by_class.csv')
    metrics_by_cl_df[
        ['class', 'category', 'TP_k', 'FP_k', 'FN_k', 'precision_k', 'recall_k', 'dataset']
    ].sort_values(by=['dataset', 'class']).to_csv(file_to_write, index=False)
    written_files.append(file_to_write)

    tmp_df = metrics_by_cl_df[['dataset', 'TP_k', 'FP_k', 'FN_k']].groupby(by='dataset', as_index=False).sum()
    tmp_df2 = pd.DataFrame(global_metrics_dict, index = range(len(dets_gdf_dict.keys())))
    global_metrics_df = tmp_df.merge(tmp_df2, on='dataset')
    global_metrics_df.rename({'TP_k': 'TP', 'FP_k': 'FP', 'FN_k': 'FN', 'precision_k': 'precision', 'recall_k': 'recall'}, inplace=True)

    file_to_write = os.path.join(cfg['output_folder'], 'global_metrics.csv')
    global_metrics_df.to_csv(file_to_write, index=False)
    written_files.append(file_to_write)

    logger.info("Saving the confusion matrix...")
    na_value_category = tagged_dets_df.category.isna()
    sorted_classes = tagged_dets_df.loc[~na_value_category, 'category'].sort_values().unique().tolist() + ['background']
    tagged_dets_df.loc[na_value_category, 'category'] = 'background'
    tagged_dets_df.loc[tagged_dets_df.det_category.isna(), 'det_category'] = 'background'

    for dataset in tagged_dets_df.dataset.unique():
        if isinstance(dataset, float) and math.isnan(dataset) :
            continue

        tagged_dataset_df = tagged_dets_df[tagged_dets_df.dataset == dataset]
        true_class = tagged_dataset_df.category.to_numpy()
        detected_class = tagged_dataset_df.det_category.to_numpy()

        confusion_array = confusion_matrix(true_class, detected_class, labels=sorted_classes)
        confusion_df = pd.DataFrame(confusion_array, index=sorted_classes, columns=sorted_classes, dtype='int64')

        file_to_write = os.path.join(cfg['output_folder'], f'{dataset}_confusion_matrix.csv')
        confusion_df.rename(columns={'background': 'missed labels'}).to_csv(file_to_write)
        written_files.append(file_to_write)

    logger.info("The following files were written :")
    [logger.info(written_file) for written_file in written_files]


if __name__ == "__main__":
    tic = time.time()
    main()
    logger.success(f"Elapsed time: {(time.time()-tic):.2f} seconds")