import vietocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict

import glob
from PIL import Image
from tqdm import tqdm 
config = Cfg.load_config_from_name('vgg_seq2seq')
# config['weights'] = './pretrained/vietocr/transformerocr_18_3_2021_with_real_data.pth'
# config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained']=False
config['device'] = 'cuda'
config['predictor']['beamsearch']=False
detector = Predictor(config)

img_paths = glob.glob('./donthuoc/*.png')
f =  open('label.txt', 'w')
# print(len(img_paths))
for img_path in tqdm(img_paths):
	img = Image.open(img_path)
	txt = detector.predict(img)
	# print(txt)
	f.write(img_path.split('/')[-1]+'\t'+txt+ '\n')

f.close()
