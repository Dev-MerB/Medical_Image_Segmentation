import numpy as np
import os
import natsort
import nibabel as nib
from tqdm import tqdm
def eval_volume_from_mask(GT_path, pred_path):
    # mask file load
    print("evaluation")
    pred_mask_list = list(natsort.natsorted(os.listdir(pred_path)))
    gt_mask_list = list(natsort.natsorted(os.listdir(GT_path)))

    p_num_l = []
    p_silce = []
    # calculation
    
    if(len(gt_mask_list) == len(pred_mask_list)):
        for i in range(len(gt_mask_list)):
            p_num_l.append(gt_mask_list[i][:8])
        p_num_l = natsort.natsorted(list(set(p_num_l)))
        for p in p_num_l:
            temp = []
            for i in range(len(gt_mask_list)):
                temp.append(gt_mask_list[i].count(p))
            p_silce.append(sum(temp))
        print(p_silce)

        tp = 0
        vol_diff = []
        t_overlab,t_jaccard,t_dice,t_fn,t_fp = [],[],[],[],[]
        for p in p_silce:
            s_sum, t_sum = 0, 0
            intersection, union = 0, 0
            s_diff_t, t_diff_s = 0, 0
            
            for i in tqdm(range(tp, tp+p)):
                gt_slice = nib.load(os.path.join(GT_path, gt_mask_list[i]))
                gt_slice = gt_slice.get_fdata().astype(np.uint16)

                pred_slice = nib.load(os.path.join(pred_path, pred_mask_list[i]))
                pred_slice = pred_slice.get_fdata().astype(np.uint16)

                if len(np.unique(gt_slice)) > 2:
                    print("GT SLICE VALUE ERROR! NOT 0 ,1")
                    print(np.unique(gt_slice))
                if len(np.unique(pred_slice)) > 2:
                    print("PRED SLICE VALUE ERROR! NOT 0 ,1")
                    print(np.unique(pred_slice))

                s_sum += (pred_slice == 1).sum()
                t_sum += (gt_slice == 1).sum()

                intersection += np.bitwise_and(pred_slice, gt_slice).sum()
                union += np.bitwise_or(pred_slice, gt_slice).sum()
                s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
                t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

            overlab = intersection / t_sum
            jaccard = intersection / union
            dice = 2.0*intersection / (s_sum + t_sum)
            fn = t_diff_s / t_sum
            fp = s_diff_t / s_sum
            print('Patient:',p_num_l[p_silce.index(p)])
            print('Overlab: %.4f Jaccard: %.4f Dice: %.4f FN: %.4f FP: %.4f' %(overlab,jaccard,dice,fn,fp))
            print('Volume diff : %.4f'%(abs((s_sum-t_sum)/t_sum)))
            vol_diff.append(abs((s_sum-t_sum)/t_sum))
            t_overlab.append(overlab)
            t_jaccard.append(jaccard)
            t_dice.append(dice)
            t_fn.append(fn)
            t_fp.append(fp)
            tp +=p
        print('Average Overlab: %.4f Jaccard: %.4f Dice: %.4f FN: %.4f FP: %.4f' 
        %(np.average(t_overlab),np.average(t_jaccard),np.average(t_dice),np.average(t_fn),np.average(t_fp)))
        print('Average Vol diff %.4f'%(np.average(vol_diff)))
    else:
        print("GT : %d, Pred : %d" %(len(gt_mask_list), len(pred_mask_list)))

# if __name__ == '__main__':
#     eval_volume_from_mask(GT_path="./result/NestedU_Net/GT", pred_path = "./result/NestedU_Net/Predict")