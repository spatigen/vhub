import cv2
import base64
import math
from PIL import Image
from moviepy import *
import tempfile
import numpy as np
import os

def reconstruct_video(base64_frames, audio_path, target_fps, output_video):
    # 创建临时目录存放解码后的图像
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        
        # 解码 base64 并保存为图像文件
        for i, b64_str in enumerate(base64_frames):
            # 解码 base64
            img_data = base64.b64decode(b64_str)
            
            # 用 OpenCV 读取
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"第 {i} 帧解码失败")
            
            # 保存为临时文件
            img_path = os.path.join(temp_dir, f"frame_{i:05d}.jpg")
            cv2.imwrite(img_path, img)
            image_paths.append(img_path)
        
        print(f"已解码 {len(image_paths)} 帧图像")

        # ========== 关键修改：音频可选 ==========
        audio_clip = None
        if audio_path and os.path.exists(audio_path):
            try:
                audio_clip = AudioFileClip(audio_path)
                audio_duration = audio_clip.duration
                print(f"音频时长: {audio_duration:.2f} 秒")
            except Exception as e:
                print(f"音频加载失败: {e}")
                audio_clip = None
        else:
            print("未提供有效音频，将生成无声视频")

        # 创建视频剪辑
        if audio_clip and len(image_paths) > 0 and audio_clip.duration > 0:
            target_fps = len(image_paths) / audio_clip.duration
        video_clip = ImageSequenceClip(image_paths, fps=target_fps)
        video_duration = len(image_paths) / target_fps
        print(f"视频时长: {video_duration:.2f} 秒")

        # 如果有音频且视频比音频短 → 延长最后一帧
        if audio_clip and video_duration < audio_clip.duration:
            last_frame_path = image_paths[-1]
            extra_frames_needed = int((audio_clip.duration - video_duration) * target_fps)
            image_paths.extend([last_frame_path] * extra_frames_needed)
            video_clip = ImageSequenceClip(image_paths, fps=target_fps)

        # ========== 关键：只有音频有效时才绑定 ==========
        if audio_clip:
            video_clip = video_clip.with_audio(audio_clip)
        else:
            print("视频将不包含音频轨道")

        # 导出
        print(f"正在导出视频到: {output_video}")
        video_clip.write_videofile(
            output_video,
            codec='libx264',
            audio_codec='aac',
            fps=target_fps,
            preset='medium',
            threads=4,
            logger=None
        )

        video_clip.close()
        if audio_clip is not None:
            audio_clip.close()

        print(f"视频已成功保存至: {output_video}")
        return output_video
    
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def token_calculate(image, model_h, model_w, model_max_tokens, model_min_tokens):
    # 获取图片的原始尺寸
    height = image.height
    width = image.width
    # 将高度调整为28的整数倍
    h_bar = round(height / model_h) * model_h
    # 将宽度调整为28的整数倍
    w_bar = round(width / model_w) * model_w
    # 图像的Token下限：4个Token
    min_pixels = model_h * model_w * model_min_tokens
    # 图像的Token上限：1280个Token
    max_pixels = model_max_tokens * model_h * model_w
    # 对图像进行缩放处理，调整像素的总数在范围[min_pixels,max_pixels]内
    if h_bar * w_bar > max_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不超过max_pixels
        beta = math.sqrt((height * width) / max_pixels)
        # 重新计算调整后的高度，确保为28的整数倍
        h_bar = math.floor(height / beta / model_h) * model_h
        # 重新计算调整后的宽度，确保为28的整数倍
        w_bar = math.floor(width / beta / model_w) * model_w
    elif h_bar * w_bar < min_pixels:
        # 计算缩放因子beta，使得缩放后的图像总像素数不低于min_pixels
        beta = math.sqrt(min_pixels / (height * width))
        # 重新计算调整后的高度，确保为28的整数倍
        h_bar = math.ceil(height * beta / model_h) * model_h
        # 重新计算调整后的宽度，确保为28的整数倍
        w_bar = math.ceil(width * beta / model_w) * model_w
    # 计算图像的Token数：总像素除以28 * 28
    token = int((h_bar * w_bar) / (model_h * model_w))
    # 系统会自动添加<|vision_bos|>和<|vision_eos|>视觉标记（各1个Token）
    total_token = token + 2
    return total_token

def extract_frames_base64(video_path, nframes, interval):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    # frame_interval = max(1, int(interval))
    base64_frames = []
    count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_str = base64.b64encode(buffer).decode("utf-8")
            base64_frames.append(base64_str)
        count += 1
        if len(base64_frames) >= nframes * 0.8:
            break
    
    cap.release()
    return base64_frames

def get_max_frame_and_interval(max_tokens, cap, frame, model_h, model_w, model_max_tokens, model_min_tokens):
    # Convert the first frame to a Pillow image object for token calculation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    first_frame_pil = Image.fromarray(frame_rgb)
    
    # Use the token_calculate method to get the tokens per frame
    tokens_per_frame = token_calculate(first_frame_pil, model_h, model_w, model_max_tokens, model_min_tokens)

    # Calculate the number of frames that can be sent
    max_nframes = math.floor(max_tokens / tokens_per_frame)
    
#    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    nframes = min(max_nframes, total_video_frames)
    
    # Calculate the frame sampling interval
    if nframes > 1:
        interval = total_video_frames / nframes
    else:
        interval = 1.0

    return nframes, interval
