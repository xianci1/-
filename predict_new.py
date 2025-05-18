
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from PIL import Image
import cv2 as cv
import cv2

class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper':3, 'plastic':4, 'trash':5}
# 加载预训练基模型（不包含顶层）
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 冻结预训练层
base_model.trainable = False

# 构建分类头
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names_to_ids), activation='softmax')
])

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_names = list(class_names_to_ids.keys())
    print(f"Predicted class: {class_names[predicted_class]}")

def continuous_detection():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    # 设置摄像头分辨率（与模型输入一致）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取帧")
            break

        # 预处理帧
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR转RGB
        img = image.array_to_img(img)
        img = img.resize((224, 224))  # 调整尺寸
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.inception_resnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # 预测
        prediction = model.predict(img_array, verbose=0)  # 关闭冗余输出
        predicted_class = np.argmax(prediction)
        class_name = list(class_names_to_ids.keys())[predicted_class]
        confidence = np.max(prediction) * 100  # 置信度百分比
        # 在帧上绘制结果
        text = f"{class_name} ({confidence:.1f}%)"
        print(text)
        cv2.putText(frame, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示实时画面
        cv2.imshow('Continuous Detection', frame)

        # 退出条件：按'q'键
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    continuous_detection()