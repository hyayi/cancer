import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

def get_transforms(h,w) :
    train_transforms = A.Compose([
                                A.HorizontalFlip(),
                                A.VerticalFlip(),
                                A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
                                A.Resize(h,w),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transforms = A.Compose([
                                A.Resize(h,w),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    
    return train_transforms, test_transforms
