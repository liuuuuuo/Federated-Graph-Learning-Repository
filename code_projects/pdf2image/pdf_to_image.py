import os
from pdf2image import convert_from_path

# 设置 poppler 路径（假设你解压到了 E:/tools/poppler）
poppler_path = r"E:\tools\poppler\Library\bin"

pdf_dir = os.path.dirname(__file__)
output_dir = os.path.join(pdf_dir, "output_images")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(pdf_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        images = convert_from_path(pdf_path, poppler_path=poppler_path)
        for i, img in enumerate(images):
            img_name = f"{os.path.splitext(filename)[0]}_page{i+1}.png"
            img.save(os.path.join(output_dir, img_name), "PNG")
        print(f"{filename} 转换完成，共 {len(images)} 页。")