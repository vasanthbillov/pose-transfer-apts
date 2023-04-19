from api import Pose2Pose
from PIL import Image

p2p = Pose2Pose(pretrained=True)


condition = Image.open("D:\LjmuMSc\Projects\Github\Z_Clone_multi_scale_attention\pose2pose\pose-transfer\pi-test\condition.jpg")
reference = Image.open("D:\LjmuMSc\Projects\Github\Z_Clone_multi_scale_attention\pose2pose\pose-transfer\pi-test\get_pose_reference - Copy.jpg")
generated = p2p.transfer_as(condition, reference)
generated.show()

# condition.show()
