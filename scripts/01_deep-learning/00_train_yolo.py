import json, os, sys, yaml, torch, ultralytics

def main(cfg, logger):
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")

    best_parameters_path = cfg['best_parameters_path'] if 'best_parameters_path' in cfg.keys() else 'None'
    yolo_default_params = cfg['yolo_training_params']

    if os.path.exists(best_parameters_path):
        with open(best_parameters_path) as fp:
            best_parameters = json.load(fp)
    else:
        logger.info("No best parameters file found.")
        best_parameters = cfg['yolo_best_params']

    if 'batch' in best_parameters.keys():
        yolo_default_params.pop('batch')

    if 'patience' in best_parameters.keys():
        yolo_default_params.pop('patience')

    os.makedirs(cfg['project'], exist_ok=True)
    yolo_default_params['data'] = os.path.join(cfg['project'], "parameters.yaml")
    with open(yolo_default_params['data'], "w") as f:
        yaml.safe_dump({
            "path": cfg['yolo_data'],
            "train": "images/trn",
            "val": "images/val",
            "test": "images/tst",
            "names": {0: "manhole"}
        }, f, sort_keys=False)

    ultralytics.YOLO(cfg['yolo_model']).train(
        project=cfg['project'],
        name=cfg['name'],
        save=True,
        save_period=5,
        plots=True,
        resume=cfg['resume_training'],
        **yolo_default_params,
        **best_parameters
    )
    
if __name__ == '__main__':
    sys.path.insert(1, 'scripts')
    from utils.logging import get_logger
    from utils.config import get_config
    main(get_config("This trains the YOLO model."), get_logger())