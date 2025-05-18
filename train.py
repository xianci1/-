import os
class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper':3, 'plastic':4, 'trash':5}
data_dir = 'D:\python__project\Garbage_Classification-main\dataset-resized/'
output_path = 'list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close()

# 随机选取样本做训练集和测试集
import random
_NUM_VALIDATION = 505
_RANDOM_SEED = 0
list_path = 'list.txt'
train_list_path = 'list_train.txt'
val_list_path = 'list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:]:
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()

def get_train_test_data(list_file):
    list_train = open(list_file)
    x_train = []
    y_train = []
    for line in list_train.readlines():
        x_train.append(line.strip()[:-2])
        y_train.append(int(line.strip()[-1]))
        #print(line.strip())
    return x_train, y_train
x_train, y_train = get_train_test_data('list_train.txt')
x_test, y_test = get_train_test_data('list_val.txt')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


# 读取list.txt文件到DataFrame
df = pd.read_csv('list.txt', sep=' ', header=None, names=['filepath', 'label'])

# 拆分训练集和验证集
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 数据增强配置
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    # preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    # preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

# 数据流生成
def dataframe_generator(dataframe, datagen, batch_size=32):
    return datagen.flow_from_dataframe(
        dataframe,
        directory='D:\python__project\Garbage_Classification-main\dataset-resized',  # 数据集根目录
        x_col='filepath',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='raw'  # 因为标签已经是数字
    )

train_generator = dataframe_generator(train_df, train_datagen)
val_generator = dataframe_generator(val_df, val_datagen)


from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

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
    layers.Dense(512, activation='relu'),  # 增加神经元数量
    layers.Dropout(0.3),  # 降低Dropout率
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names_to_ids), activation='softmax')
])

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

# 训练参数
epochs = 30
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)
# 编译模型
# 设置余弦退火学习率
lr_schedule = CosineDecay(
    initial_learning_rate=1e-3,  # 初始学习率设为1e-3
    decay_steps=steps_per_epoch * epochs
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),  # 使用动态学习率
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 回调函数
callbacks = [
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,  # 关键修改
        mode='max'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
]

# 训练

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

model.load_weights('D:\python__project\Garbage_Classification-main\Garbage_Classification-main/best_model.keras')
# 使用验证集评估模型
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")


