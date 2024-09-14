current_dir=$(pwd)

# get all folders start with "eval-" in the current directory

eval_folders=$(find . -maxdepth 1 -type d -name "eval-*")

for eval_folder in ${eval_folders[@]}; do
    echo "Running $eval_folder"
    cd $eval_folder

    ./clean.sh

    cd $current_dir
done