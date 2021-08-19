CURRENT_DATE=`date +"%Y%m%d%H"`
python train.py dcai_gcb_00 label_book > ./logs/train_log_${CURRENT_DATE}.log data label_book
