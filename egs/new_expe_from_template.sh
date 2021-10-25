new_expe_name="libri360_train_with_awesome_modification"
template="libri360_train"

rsync -av --progress "$template/" "$new_expe_name/" --exclude "log" --exclude "model" --exclude "slurm*"

mkdir -p "$new_expe_name/log"
mkdir -p "$new_expe_name/model"
