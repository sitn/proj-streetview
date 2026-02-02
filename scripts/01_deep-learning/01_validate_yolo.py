import os, sys, time, pandas as pd, ultralytics

sys.path.insert(1, 'scripts')
from utils.logging import get_logger
from utils.config import get_config

logger = get_logger()
cfg = get_config("This script computes test and eval metrics for the given YOLO model.")

def main ():
    model = ultralytics.YOLO(cfg['model'])
    logger.info(f"Perform validation...")
    df_val = model.val(plots=True, project=cfg['project'], name='val', exist_ok=True, **cfg['yolo_training_params']).to_df()
    logger.info(f"Perform test...")
    df_test = model.val(split='test', plots=True, project=cfg['project'], name='tst', exist_ok=True, **cfg['yolo_training_params']).to_df()
    
    metrics_df = pd.concat([df_val.to_pandas(), df_test.to_pandas()], ignore_index=True)
    metrics_df['dataset'] = ['val', 'tst']

    filepath=os.path.join(cfg['project'], 'metrics.csv')
    metrics_df.to_csv(filepath, index=False)
    logger.info(f"Saved metrics to :")
    logger.info(f"{filepath}")
    logger.info(f"{os.path.join(cfg['project'], 'val')}")
    logger.info(f"{os.path.join(cfg['project'], 'tst')}")


if __name__ == '__main__':
    tic = time.time()
    main()
    logger.success(f"Done in {round(time.time() - tic, 2)} seconds.")