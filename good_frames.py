import numpy as np
import glob
import util
import pickle

samples = [i.split('/')[-1] for i in glob.glob('data/predict_cleaned/unet3000/*')]
good_frames = {}

for s in sorted(samples):
    print(s)
    segs = np.array([util.read_vol(f) for f in sorted(glob.glob(f'data/predict_cleaned/unet3000/{s}/{s}_*.nii.gz'))])
    label = glob.glob(f'data/labels/{s}/{s}_*_all_brains.nii.gz')
    frames = []
    if label:
        n = label[0].split('/')[-1].split('_')[1]
        vol = util.read_vol(f'data/predict_cleaned/unet3000/{s}/{s}_{str(n).zfill(4)}.nii.gz')
        volume = np.sum(vol)
        prev = None
        prev_vol = None
        for i in range(len(segs)):
            seg = segs[i]
            curr_vol = np.sum(seg)
            dif = abs(volume - curr_vol)
            if dif / volume <= 0.1 and util.dice_coef(vol, seg) >= 0.8:
                if prev is not None and abs(curr_vol - prev_vol) / prev_vol <= 0.05 and util.dice_coef(prev, seg) >= 0.9:
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
            if dif / volume <= 0.1:
                if prev is not None and abs(curr_vol - prev_vol) / prev_vol <= 0.05 and util.dice_coef(prev, seg) >= 0.9:
                    frames.append(i)
            prev = seg
            prev_vol = curr_vol
    print(len(frames)/len(segs))
    if len(frames)/len(segs) > 0.6:
        good_frames[s] = frames

print(good_frames)
with open('good_frames.p', 'wb') as f:
    pickle.dump(good_frames, f)
