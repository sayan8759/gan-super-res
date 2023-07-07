from fastapi import FastAPI, UploadFile, File , Request
import uvicorn
import cv2
from fastapi.responses import FileResponse
import os
import matplotlib.pyplot as plt
import skimage.io
import subprocess
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

model_mapping = {
    "arb": "ArbSR",
    "carn": "CARN",
    "cbl": "ContextualBilateralLoss",
    "cl": "ContextualLoss",
    "espcn": "ESPCN",
    "esrgan": "ESRGAN",
    "liif": "LIIF",
    "metasr": "MetaSR",
    "msrn": "MSRN",
    "srcnn": "SRCNN",
    "srdense": "SRDenseNet"
}
weight_mapping = {
    "arb": "ArbSR_RCAN_x1_x4-DIV2K-8c206342.pth.tar",
    "carn": "CARN_x2-DIV2K-4797e51b.pth.tar",
    "cbl": "SRGAN_CoBi_x4-DIV2K-8c4a7569.pth.tar",
    "cl": "SRGAN_CX_x4-DIV2K-8c4a7569.pth.tar",
    "espcn": "ESPCN_x4-T91-64bf5ee4.pth.tar",
    "esrgan": "ESRGAN_x4-DFO2K-25393df7.pth.tar",
    "liif": "LIIF_EDSR_x4-DIV2K-cc1955cd.pth.tar",
    "metasr": "MetaSR_RDN-DIV2K-8daac205.pth.tar",
    "msrn": "MSRN_x4-DIV2K-572bb58f.pth.tar",
    "srcnn": "srcnn_x2-T91-7d6e0623.pth.tar",
    "srdense": "SRDenseNet_x4-ImageNet-bb28c23d.pth.tar"
}

def preprocess_image(image_path, desired_shape, output_folder):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (desired_shape[1], desired_shape[0]))
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_image)
    output_path = output_path.replace("\\","/")
    return output_path

@app.post("/enhance_image")
async def enhance_image(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as temp_file:
        temp_file.write(await file.read())

    # Perform image enhancement
    path = file_path.replace("\\", "/")
    filename, file_extension = os.path.splitext(path)
    file_name = os.path.basename(filename)
    ext = file_extension[1:]
    image = skimage.io.imread(path)
    score = 10 - (cv2.Laplacian(image, cv2.CV_64F).var() / 1000)

    if score >= 9.5:
        output_path = f"output/{file_name}.{ext}"
        cv2.imwrite(output_path, image)
        return FileResponse(output_path)

    elif score < 9.5 and score >= 9:
        model_id = "cbl"
        image_path = path
        desired_shape = (180, 120, 3)
        output_folder = "preprocessed"
        input_path = preprocess_image(image_path, desired_shape, output_folder)
        output_path = f"output/{model_mapping[model_id]}.{ext}"
        weights_path = f"gan-models/{model_mapping[model_id]}/results/pretrained_models/{weight_mapping[model_id]}"
        command = f"python gan-models/{model_mapping[model_id]}/inference.py --inputs_path {input_path} --output_path {output_path} --weights_path {weights_path}"
        return_code = os.system(command)
        if return_code == 0:
            print("Inference completed successfully.")
            return FileResponse(output_path)
        else:
            print(f"Inference failed with return code: {return_code}.")
            return {"message": f"Inference failed with return code: {return_code}."}

    elif score < 9 and score >= 8.5:
        model_id = "esrgan"
        image_path = path
        desired_shape = (180, 120, 3)
        output_folder = "preprocessed"
        input_path = preprocess_image(image_path, desired_shape, output_folder)
        output_path = f"output/{model_mapping[model_id]}.{ext}"
        weights_path = f"gan-models/{model_mapping[model_id]}/results/pretrained_models/{weight_mapping[model_id]}"
        command = f"python gan-models/{model_mapping[model_id]}/inference.py --inputs_path {input_path} --output_path {output_path} --weights_path {weights_path}"
        return_code = os.system(command)
        if return_code == 0:
            print("Inference completed successfully.")
            return FileResponse(output_path)
        else:
            print(f"Inference failed with return code: {return_code}.")
            return {"message": f"Inference failed with return code: {return_code}."}

    elif score < 8.5:
        model_id = "esrganplus"
        inp_img = path
        out_img = f"output/{filename}-{model_id}.{ext}"
        command = f".\\realesrgan-ncnn-vulkan.exe -i {inp_img} -o {out_img} -n realesrgan-x4plus -g None"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Inference Successful")
            return FileResponse(out_img)
        #{image_path: path , img_emb : [...,..,..]}
        else:
            print(f"Error: {result.stderr}")
            return {"message": f"Error: {result.stderr}"}

if __name__ == "__main__":
    uvicorn.run("gan:app", host="localhost", port=3000)        

