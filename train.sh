python3 train.py --train_path /home/lab/inseo/cancer/data/train_re.csv --test_path /home/lab/inseo/cancer/data/test_re.csv --weight_path /home/lab/inseo/cancer/ --log_path /home/lab/inseo/cancer/ --model_name regnet_y_32gf    --batch_size 2 --strategy ddp --device 4
python3 train.py --train_path /home/lab/inseo/cancer/data/train_re.csv --test_path /home/lab/inseo/cancer/data/test_re.csv --weight_path /home/lab/inseo/cancer/ --log_path /home/lab/inseo/cancer/ --model_name convnext_base --batch_size 2 --strategy ddp --device 4
python3 train.py --train_path /home/lab/inseo/cancer/data/train_re.csv --test_path /home/lab/inseo/cancer/data/test_re.csv --weight_path /home/lab/inseo/cancer/ --log_path /home/lab/inseo/cancer/ --model_name efficientnet_b7 --batch_size 4 --strategy ddp --device 4
python3 train.py --train_path /home/lab/inseo/cancer/data/train_re.csv --test_path /home/lab/inseo/cancer/data/test_re.csv --weight_path /home/lab/inseo/cancer/ --log_path /home/lab/inseo/cancer/ --model_name efficientnet_v2_l --batch_size 2 --strategy ddp --device 4


