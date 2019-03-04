import numpy as np
import glob
import util

OVERALL_VOL_DIF = 0.05
OVERALL_DICE = 0.8
SEQ_VOL_DIF = 0.05
SEQ_DICE = 0.9
PERCENT_GOOD = 0.6

models = [i.split('/')[-1] for i in glob.glob('data/predict_cleaned/*')]
for model in models:
    print(model)
    samples = [i.split('/')[-1] for i in glob.glob(f'data/predict_cleaned/{model}/*')]
    good_frames = {}
    num_good_frames = 0

    for s in sorted(samples):
        segs = np.asarray([util.read_vol(f) for f in sorted(glob.glob(f'data/predict_cleaned/{model}/{s}/{s}_*.nii.gz'))])
        label = glob.glob(f'data/labels/{s}/{s}_*_all_brains.nii.gz')
        frames = []
        if label:
            n = label[0].split('/')[-1].split('_')[1]
            vol = util.read_vol(f'data/predict_cleaned/{model}/{s}/{s}_{str(n).zfill(4)}.nii.gz')
            volume = np.sum(vol)
            prev = None
            prev_vol = None
            for i in range(len(segs)):
                seg = segs[i]
                curr_vol = np.sum(seg)
                dif = abs(volume - curr_vol)
                if dif / volume <= OVERALL_VOL_DIF and util.dice_coef(vol, seg) >= OVERALL_DICE:
                    if prev is not None and abs(curr_vol - prev_vol) / prev_vol <= SEQ_VOL_DIF and util.dice_coef(prev, seg) >= SEQ_DICE:
                        frames.append(i)
                prev = seg
                prev_vol = curr_vol
        else:
            volume = np.median(np.sum(segs, axis=tuple(np.arange(1, segs.ndim))))
            prev = None
            prev_vol = None
            for i in range(len(segs)):
                seg = segs[i]
                curr_vol = np.sum(seg)
                dif = abs(volume - curr_vol)
                if dif / volume <= OVERALL_VOL_DIF:
                    if prev is not None and abs(curr_vol - prev_vol) / prev_vol <= SEQ_VOL_DIF and util.dice_coef(prev, seg) >= SEQ_DICE:
                        frames.append(i)
                prev = seg
                prev_vol = curr_vol
        if len(frames)/len(segs) >= PERCENT_GOOD:
            good_frames[s] = frames
        if s in ['052218S', '052218L', '031616', '062515', '051215', '043015', '052516', '041818', '022618', '032818', '022415', '031317L', '061715', '102617', '050318L', '031516', '013118S', '032318c', '021015', '040417', '083115', '050917']:
            print(s, len(frames)/len(segs))
        num_good_frames += len(frames)
        # print(frames)
    print(len(good_frames), num_good_frames)
    print('-----')
