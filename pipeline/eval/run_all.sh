current_dir=$(pwd)

# get all folders start with "eval-" in the current directory

cd ../../datasets/
./gen.sh
cd $current_dir

eval_folders=$(find . -maxdepth 1 -type d -name "eval-*")

for eval_folder in ${eval_folders[@]}; do
    echo "Running $eval_folder"
    cd $eval_folder

    python run.py --trace_base ../../datasets/traces  --executor_base ../utils/

    cd $current_dir
done

python baseline_data.py
python summary.py
python plot_all.py