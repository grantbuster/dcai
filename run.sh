CURRENT_DATE=`date +"%Y%m%d%H"`
python train.py > ./logs/train_log_${CURRENT_DATE}.log data label_book
