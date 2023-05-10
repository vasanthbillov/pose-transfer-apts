from api import Pose2Pose
from PIL import Image

p2p = Pose2Pose(pretrained=True)


condition = Image.open("D:/LjmuMSc/Projects/Github/PoseTransfer_MS_RnD/Ztesting/c_img.jpg")
condition_seg = Image.open("D:/LjmuMSc/Projects/Github/PoseTransfer_MS_RnD/Ztesting/c_seg.png")
reference = Image.open("D:/LjmuMSc/Projects/Github/PoseTransfer_MS_RnD/Ztesting/r_pose.jpg")
reference_seg = Image.open("D:/LjmuMSc/Projects/Github/PoseTransfer_MS_RnD/Ztesting/r_seg_pose.png")

generated = p2p.temp_transfer_as(condition, reference,condition_seg, reference_seg)
generated.show()

# condition.show()
