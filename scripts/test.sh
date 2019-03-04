declare -a arr=("052218S" "052218L" "031616" "062515" "051215" "043015" "052516" "041818" "022618" "032818" "022415" "031317L" "061715" "102617" "050318L" "031516" "013118S" "032318c" "021015" "040417" "083115" "050917")

for i in "${arr[@]}"
do
    srun -p gpu -t 20:00:00 --mem-per-cpu 16 --gres=gpu:1 -J unet_frames_${i} -o unet_frames_${i}.out -e unet_frames_${i}.err python good_frames.py --sample ${i} &
done
