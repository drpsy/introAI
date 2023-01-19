import glob
import os
from tqdm import tqdm
def get_valid_gt(path,save):
    gt_files = glob.glob(path + "/*")
    for file in tqdm(gt_files):
        fname = os.path.basename(file)
        print(fname)
        with open(file, "r") as gt_file:
            data = gt_file.readlines()
        with open(os.path.join(save,fname),"w+") as new_gt_file:
            for line in data:
                line = line.rstrip().replace("\n", "")
                line = line.split(",")
                lb=line[-1]
                line[-1]="Latin"
                # if lb!="###":
                #     if lb[-1] == ']':
                #         try:
                #             len_lb=int(lb[:-1])
                #         except Exception as e:
                #             continue
                #             raise e
                #         line.append('A'*len_lb)
                #         continue
                #         pass
                #     try:
                #         len_lb=int(lb)
                #     except Exception as e:
                #         continue
                #         raise e
                #     line.append('A'*len_lb)
                # else:
                #     line.append("###")
                line.append(lb)
                new_gt_file.write(','.join(str(l) for l in line)+'\n')

get_valid_gt("../../../disk2/hiepnm/craft/data/Vin_dataset/vietnamese/labels/", "../../../disk2/hiepnm/craft/data/Vin_dataset/vietnamese/gt/")