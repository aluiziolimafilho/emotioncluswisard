from ckp import CKP
from skimage import filters
from wisardpkg.wisard import Wisard
ckp = CKP()

ckp.read_data(threshold_func=filters.threshold_sauvola, all_imgs=False)
print(len(ckp.imgs))
