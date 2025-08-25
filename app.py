import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import time

# 设置页面标题和布局
st.set_page_config(
    page_title="安全检测系统",
    page_icon="⛑️",
    layout="wide"
)

# 应用标题
st.title("⛑️ 安全检测系统")
st.markdown("上传图片或短视频进行安全检测，系统将识别其中的安全装备佩戴情况")

# 定义类别和颜色映射
CLASS_NAMES = ["Hardhat", "Mask", "NO-Hardhat", "No-Mask", "No-Safety Vest",
               "Person", "Safety Cone", "SafetyVest", "machinery", "vehicle"]

# 为不同类别定义颜色 (BGR格式)
COLOR_MAP = {
    "Hardhat": (0, 255, 0),  # 绿色 - 佩戴安全帽
    "NO-Hardhat": (0, 0, 255),  # 红色 - 未戴安全帽
    "SafetyVest": (0, 255, 0),  # 绿色 - 穿着安全背心
    "No-Safety Vest": (0, 0, 255),  # 红色 - 未穿安全背心
    "Mask": (0, 255, 0),  # 绿色 - 戴口罩
    "No-Mask": (0, 0, 255),  # 红色 - 未戴口罩
    "Person": (255, 0, 0),  # 蓝色 - 人员
    "Safety Cone": (0, 165, 255),  # 橙色 - 安全锥
    "machinery": (255, 0, 255),  # 紫色 - 机械设备
    "vehicle": (255, 255, 0)  # 青色 - 车辆
}

# 设置模型路径 - 请根据您的实际路径修改
model_path = "runs/detect/train/weights/best.pt"  # 请修改为您的实际路径


# 创建模型缓存装饰器
@st.cache_resource
def load_model(_model_path):
    """加载YOLOv8模型"""
    try:
        # 检查模型文件是否存在
        if not os.path.exists(_model_path):
            st.error(f"模型文件不存在: {_model_path}")
            return None

        # 加载模型
        model = YOLO(_model_path)
        st.success(f"成功加载模型!")
        return model
    except Exception as e:
        st.error(f"加载模型时出错: {e}")
        return None


# 侧边栏设置
st.sidebar.header("检测设置")

# 允许用户输入模型路径
custom_model_path = st.sidebar.text_input(
    "模型文件路径",
    value=model_path,
    help="请输入best.pt文件的完整路径"
)

# 加载模型
with st.spinner('正在加载检测模型...'):
    model = load_model(custom_model_path)

if model is None:
    st.error("无法加载模型，请检查模型路径是否正确。")

    # 显示常见问题解决方法
    with st.expander("常见问题解决"):
        st.markdown("""
        1. **确保模型路径正确**:
           - 使用绝对路径，如: `C:/Users/YourName/Desktop/best.pt`
           - 或相对路径，如: `models/best.pt`

        2. **确保文件存在**:
           - 检查文件是否在指定位置
           - 检查文件名是否正确（包括扩展名）

        3. **文件权限**:
           - 确保应用有读取该文件的权限
        """)

    st.stop()

confidence_threshold = st.sidebar.slider(
    "置信度阈值",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    help="调整检测的敏感度，值越高要求检测越准确"
)

# 视频处理设置
max_video_duration = st.sidebar.slider(
    "最大处理时长(秒)",
    min_value=5,
    max_value=60,
    value=15,
    help="限制视频处理的最大时长，避免处理时间过长"
)

# 检测类别说明
st.sidebar.markdown("### 检测类别说明")
st.sidebar.markdown("- 🟢 **安全装备**: 安全帽、安全背心、口罩等")
st.sidebar.markdown("- 🔴 **不安全状态**: 未戴安全帽、未穿安全背心、未戴口罩等")
st.sidebar.markdown("- 🔵 **人员**: 检测到的人员")
st.sidebar.markdown("- 🟠 **安全设施**: 安全锥等")
st.sidebar.markdown("- 🟣 **设备**: 机械设备")
st.sidebar.markdown("- 🟡 **车辆**: 检测到的车辆")

# 文件上传区域
uploaded_file = st.file_uploader(
    "选择图片或短视频进行安全检测",
    type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi']
)


# 处理检测结果的函数
def process_detection(results, image):
    """处理检测结果并在图像上绘制边界框"""
    # 创建图像的副本
    image_copy = image.copy()

    # 初始化统计信息
    detection_count = {cls: 0 for cls in CLASS_NAMES}

    # 处理每个检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 获取置信度和类别
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = CLASS_NAMES[cls_id]

            # 只绘制置信度高于阈值的检测结果
            if conf >= confidence_threshold:
                # 更新统计信息
                detection_count[cls_name] += 1

                # 获取类别颜色
                color = COLOR_MAP.get(cls_name, (255, 255, 255))  # 默认为白色

                # 绘制边界框
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)

                # 创建标签文本
                label = f"{cls_name} {conf:.2f}"

                # 获取文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # 绘制文本背景
                cv2.rectangle(
                    image_copy,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # 绘制文本
                cv2.putText(
                    image_copy,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

    return image_copy, detection_count


# 处理视频的函数
def process_video(video_path, max_duration=15):
    """处理视频文件并返回处理后的视频路径"""
    # 创建临时文件保存处理后的视频
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output_path = temp_output.name
    temp_output.close()

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算最大帧数
    max_frames = int(fps * max_duration)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)

    # 处理每一帧
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 处理当前帧
        results = model(frame)
        processed_frame, _ = process_detection(results, frame)

        # 写入处理后的帧
        out.write(processed_frame)

        # 更新进度
        frame_count += 1
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"处理中: {frame_count}/{total_frames} 帧 ({progress:.1%})")

    # 释放资源
    cap.release()
    out.release()

    # 完成进度
    progress_bar.progress(1.0)
    status_text.text("处理完成!")

    return temp_output_path


# 显示统计信息的函数
def display_statistics(detection_count):
    """显示检测统计信息"""
    st.success("检测完成!")

    # 分类显示统计结果
    col_stats1, col_stats2, col_stats3 = st.columns(3)

    with col_stats1:
        st.markdown("**安全装备**")
        if detection_count["Hardhat"] > 0:
            st.markdown(f"🟢 安全帽: {detection_count['Hardhat']}")
        if detection_count["SafetyVest"] > 0:
            st.markdown(f"🟢 安全背心: {detection_count['SafetyVest']}")
        if detection_count["Mask"] > 0:
            st.markdown(f"🟢 口罩: {detection_count['Mask']}")

    with col_stats2:
        st.markdown("**不安全状态**")
        if detection_count["NO-Hardhat"] > 0:
            st.markdown(f"🔴 未戴安全帽: {detection_count['NO-Hardhat']}")
        if detection_count["No-Safety Vest"] > 0:
            st.markdown(f"🔴 未穿安全背心: {detection_count['No-Safety Vest']}")
        if detection_count["No-Mask"] > 0:
            st.markdown(f"🔴 未戴口罩: {detection_count['No-Mask']}")

    with col_stats3:
        st.markdown("**其他检测**")
        if detection_count["Person"] > 0:
            st.markdown(f"🔵 人员: {detection_count['Person']}")
        if detection_count["Safety Cone"] > 0:
            st.markdown(f"🟠 安全锥: {detection_count['Safety Cone']}")
        if detection_count["machinery"] > 0:
            st.markdown(f"🟣 机械设备: {detection_count['machinery']}")
        if detection_count["vehicle"] > 0:
            st.markdown(f"🟡 车辆: {detection_count['vehicle']}")


# 如果上传了文件
if uploaded_file is not None:
    # 获取文件类型
    file_type = uploaded_file.type.split('/')[0]
    file_ext = uploaded_file.name.split('.')[-1].lower()

    # 保存上传的文件到临时位置
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # 处理图片
    if file_type == 'image':
        # 读取图片
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 转换颜色空间 (PIL是RGB, OpenCV需要BGR)
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 创建两列布局
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("原始图片")
            st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("检测结果")

            # 进行预测
            with st.spinner('正在检测...'):
                results = model(image_np)

                # 处理检测结果
                result_image, detection_count = process_detection(results, image_np)

                # 转换颜色空间用于显示
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                # 显示检测结果
                st.image(result_image_rgb, use_container_width=True)

                # 显示统计信息
                display_statistics(detection_count)

        # 添加下载结果图片的功能
        result_pil_image = Image.fromarray(result_image_rgb)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
            result_pil_image.save(tmpfile.name, format='JPEG')

            with open(tmpfile.name, 'rb') as file:
                btn = st.download_button(
                    label="下载检测结果",
                    data=file,
                    file_name="detection_result.jpg",
                    mime="image/jpeg"
                )

        # 清理临时文件
        os.unlink(tmpfile.name)

    # 处理视频
    elif file_type == 'video':
        st.subheader("视频检测")

        # 显示原始视频
        st.markdown("**原始视频**")
        st.video(uploaded_file)

        # 处理视频
        with st.spinner('正在处理视频...'):
            processed_video_path = process_video(tmp_path, max_video_duration)

        # 显示处理后的视频
        #st.markdown("**检测结果视频**")

        # 读取处理后的视频文件
        with open(processed_video_path, 'rb') as video_file:
            video_bytes = video_file.read()

        # 显示视频
        #st.video(video_bytes)

        # 提供下载链接
        st.download_button(
            label="下载处理后的视频",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

        # 清理临时文件
        os.unlink(processed_video_path)

    # 清理上传的临时文件
    os.unlink(tmp_path)

else:
    # 显示示例图片和说明
    st.info("请上传图片或短视频开始检测")

    # 使用选项卡布局
    tab1, tab2 = st.tabs(["图片检测", "视频检测"])

    with tab1:
        st.markdown("### 图片检测示例")

        # 使用三列布局显示示例说明
        col1, col2, col3 = st.columns(3)


    with tab2:
        st.markdown("### 视频检测示例")
        st.markdown("支持上传短视频文件进行安全检测")
        st.markdown("**支持的视频格式**: MP4, MOV, AVI")
        st.markdown("**处理限制**: 默认处理前15秒，可在侧边栏调整")

        # 视频检测说明
        st.markdown("""
        #### 视频检测特点:
        - 逐帧处理视频内容
        - 实时显示处理进度
        - 提供处理后的视频下载
        - 可调整最大处理时长
        """)

    st.markdown("""
    ### 使用说明:
    1. 在左侧输入模型文件的正确路径
    2. 点击"选择图片或短视频进行安全检测"区域上传文件
    3. 调整左侧的置信度阈值和最大处理时长(视频)
    4. 查看检测结果
    5. 可以下载处理后的结果文件
    """)

# 页脚
st.markdown("---")
st.markdown("安全检测系统 | 基于YOLOv8模型开发")