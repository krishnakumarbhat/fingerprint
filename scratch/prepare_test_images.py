import os
from PIL import Image

def convert_bmp_to_png(src_path, dest_path):
    if not os.path.exists(src_path):
        print(f"Error: {src_path} does not exist.")
        return False
    try:
        img = Image.open(src_path)
        img.save(dest_path, "PNG")
        print(f"Successfully converted {src_path} -> {dest_path}")
        return True
    except Exception as e:
        print(f"Error converting {src_path}: {e}")
        return False

def main():
    base_dir = "/media/pope/projecteo/github_proj/a_resume/fingerprint"
    real_dir = os.path.join(base_dir, "archive/SOCOFing/Real")
    altered_dir = os.path.join(base_dir, "archive/SOCOFing/Altered")
    dest_dir = os.path.join(base_dir, "web/data/test_prints")
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Define our 3 pairs of interest:
    # Pair 1: Genuine Easy
    # Subject 100, Male, Left Index Finger
    p1_real = os.path.join(real_dir, "100__M_Left_index_finger.BMP")
    p1_alt = os.path.join(altered_dir, "Altered-Easy/100__M_Left_index_finger_CR.BMP")
    
    # Pair 2: Genuine Hard
    # Subject 100, Male, Left Index Finger vs Zcut Alteration (Hard)
    p2_real = p1_real # reuse same real
    p2_alt = os.path.join(altered_dir, "Altered-Hard/100__M_Left_index_finger_Zcut.BMP")
    
    # Pair 3: Impostor
    # Subject 100 vs Subject 101, Male, Left Index Finger
    p3_real_100 = p1_real
    p3_real_101 = os.path.join(real_dir, "101__M_Left_index_finger.BMP")

    # Let's perform the conversions and save them with friendly names
    convert_bmp_to_png(p1_real, os.path.join(dest_dir, "subject100_left_index_real.png"))
    convert_bmp_to_png(p1_alt, os.path.join(dest_dir, "subject100_left_index_altered_easy.png"))
    convert_bmp_to_png(p2_alt, os.path.join(dest_dir, "subject100_left_index_altered_hard.png"))
    convert_bmp_to_png(p3_real_101, os.path.join(dest_dir, "subject101_left_index_real.png"))
    
    # Let's also copy some extra files for user to import/drag-and-drop directly
    # e.g., subject 100 left thumb real and altered
    p4_real = os.path.join(real_dir, "100__M_Left_thumb_finger.BMP")
    p4_alt = os.path.join(altered_dir, "Altered-Medium/100__M_Left_thumb_finger_Obl.BMP")
    convert_bmp_to_png(p4_real, os.path.join(dest_dir, "subject100_left_thumb_real.png"))
    convert_bmp_to_png(p4_alt, os.path.join(dest_dir, "subject100_left_thumb_altered_medium.png"))

if __name__ == "__main__":
    main()
