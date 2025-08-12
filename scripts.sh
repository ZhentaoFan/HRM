apt update
apt install nano
pip install -r requirements.txt
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000
OMP_NUM_THREADS=8 torchrun --nproc-per-node 1 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=4000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
