declare -a arr=("052218S" "052218L" "031616" "062515" "051215" "043015" "052516" "041818" "022618" "032818" "022415" "031317L" "061715" "102617" "050318L" "031516" "013118S" "032318c" "050917")

for i in "${arr[@]}"
do
    srun -p gpu -t 20:00:00 --mem-per-cpu 16 --gres=gpu:1 python volume_plot.py unet_frames_${i} --sample ${i} &
done
