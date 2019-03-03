declare -a arr=("052218S" "052218L" "031616" "062515" "051215" "043015" "052516" "041818" "022618" "032818" "022415" "031317L" "061715" "102617" "050318L" "031516" "013118S" "032318c" "083115" "040417" "021015" "050917")

for i in "${arr[@]}":
do
    scripts/slurm_run.sh unet_frames_${i} --model unet --organ brains --model-file models/unet_frames_full/2000.h5 --split 1 0 --predict-all --good-frames unet3000 --sample ${i}
done
