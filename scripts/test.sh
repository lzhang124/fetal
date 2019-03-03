declare -a arr=("52218S" "52218L" "31616" "62515" "51215" "43015" "52516" "41818" "22618" "32818" "22415" "31317L" "61715" "02617" "50318L" "31516" "13118S" "32318c" "83115" "40417" "21015" "50917")

for i in "${arr[@]}":
do
    scripts/slurm_run.sh unet_frames_${i} --model unet --organ brains --model-file models/unet_frames_full/2000.h5 --split 1 0 --predict-all --good-frames unet3000 --sample ${i}
done
