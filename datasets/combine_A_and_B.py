# from pdb import set_trace as st
# import os
# import numpy as np
# import cv2
# import argparse

# parser = argparse.ArgumentParser('create image pairs')
# parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
# parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
# parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
# parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=1000000)
# parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
# args = parser.parse_args()

# for arg in vars(args):
#     print('[%s] = ' % arg,  getattr(args, arg))

# splits = os.listdir(args.fold_A)

# for sp in splits:
#     img_fold_A = os.path.join(args.fold_A, sp)
#     img_fold_B = os.path.join(args.fold_B, sp)
#     img_list = os.listdir(img_fold_A)
#     if args.use_AB: 
#         img_list = [img_path for img_path in img_list]

#     num_imgs = min(args.num_imgs, len(img_list))
#     print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
#     img_fold_AB = os.path.join(args.fold_AB, sp)
#     if not os.path.isdir(img_fold_AB):
#         os.makedirs(img_fold_AB)
#     print('split = %s, number of images = %d' % (sp, num_imgs))
#     for n in range(num_imgs):
#         name_A = img_list[n]
#         path_A = os.path.join(img_fold_A, name_A)
#         if args.use_AB:
#             name_B = name_A.replace('_A.', '_B.')
#         else:
#             name_B = name_A
#         path_B = os.path.join(img_fold_B, name_B)
#         if os.path.isfile(path_A) and os.path.isfile(path_B):
#             name_AB = name_A
#             if args.use_AB:
#                 name_AB = name_AB.replace('_A.', '.') # remove _A
#             path_AB = os.path.join(img_fold_AB, name_AB)
#             im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
#             im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
#             im_AB = np.concatenate([im_A, im_B], 1)
#             cv2.imwrite(path_AB, im_AB)

# print("Thư mục fold_A:", args.fold_A)
# print("Danh sách splits:", splits)     

# if not os.path.exists(img_fold_A):
#     print(f"Thư mục {img_fold_A} không tồn tại")
   
# if not os.path.exists(img_fold_B):
#     print(f"Thư mục {img_fold_B} không tồn tại")

from pdb import set_trace as st  # Import for setting breakpoints (debugging)
import os
import numpy as np
import cv2
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='Input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='Input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='Output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='Number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='If true, convert (0001_A, 0001_B) to (0001_AB)', action='store_true')
args = parser.parse_args()

# Print all arguments
for arg in vars(args):
    print(f'[{arg}] = {getattr(args, arg)}')

# Check input directories
print("\n--- Kiểm tra thư mục đầu vào ---")
print(f"Thư mục A: {args.fold_A}")
print(f"Thư mục B: {args.fold_B}")
print(f"Thư mục đầu ra: {args.fold_AB}")

# List all splits in fold_A
splits = os.listdir(args.fold_A)
print("\nDanh sách splits trong thư mục fold_A:", splits)

for sp in splits:
    print("\nProcessing split:", sp)
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)  # Correct the path for fold_B
    img_fold_AB = os.path.join(args.fold_AB, sp)

    # Check if the input directories exist
    if not os.path.exists(img_fold_A):
        print(f"Không tìm thấy {img_fold_A}")
        continue
    if not os.path.exists(img_fold_B):
        print(f"Không tìm thấy {img_fold_B}")
        continue

    # Check if the output directory exists; create if not
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
        print(f"Tạo thư mục đầu ra: {img_fold_AB}")

    # List images in folder A
    img_list = os.listdir(img_fold_A)
    print(f"Danh sách ảnh trong thư mục A: {img_list}")

    # Filter images if using '_A.' for args.use_AB
    if args.use_AB:
        img_list = [img_path for img_path in img_list]
    print(f"Danh sách ảnh sau khi lọc: {img_list}")

    num_imgs = min(args.num_imgs, len(img_list))
    print(f"Tổng số ảnh sử dụng: {num_imgs}")

    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)

        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)

        # Check if both image files exist
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # Remove '_A' from the name
            path_AB = os.path.join(img_fold_AB, name_AB)

            # Read images
            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)

            # Check if images were read correctly
            if im_A is None:
                print(f"Không thể đọc ảnh A: {path_A}")
                continue
            if im_B is None:
                print(f"Không thể đọc ảnh B: {path_B}")
                continue

            # Combine images horizontally
            im_AB = np.concatenate([im_A, im_B], axis=1)

            # Write the combined image to the output directory
            cv2.imwrite(path_AB, im_AB)
            print(f"Ghi ảnh kết hợp vào: {path_AB}")
        else:
            print(f"Không tìm thấy file ảnh hợp lệ: {path_A} hoặc {path_B}")





