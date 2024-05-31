# 필요한 라이브러리 import
import tensorflow as tf

# MNIST 데이터 다운로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 이미지(uint8)들을 float32 데이터 타입으로 변경
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784(=28*28)차원으로 flattening
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255](unint8: 0~255) 사이의 값을 [0, 1] 사이의 값으로 Normlize
x_train, x_test = x_train / 255., x_test / 255.
# 레이블 데이터에 one-hot encoding 적용
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# tf.data API를 이용하여 데이터를 섞고 batch 형태로 가져옴
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(100)
train_data_iter = iter(train_data)

# tf.keras.Model을 이용하여 softmax Regression 모델 정의
class SoftmaxRegression(tf.keras.Model):
    def __init__(self):
        super(SoftmaxRegression, self).__init__()
        self.softmax_layer = tf.keras.layers.Dense(10,
                                                   activation=None,
                                                   kernel_initializer='zeros',
                                                   bias_initializer='zeros')

    # 입력 데이터에 대한 소프트맥스 계산
    def call(self, x):
        logits = self.softmax_layer(x)
        return tf.nn.softmax(logits)

# cross-entropy 손실 함수 정의
@tf.function 
def cross_entropy_loss(y_pred, y):
    return tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred), axis=[1]))

# 최적화를 위한 gradient descent optimizer 정의
optimizer = tf.optimizers.SGD(0.5)

# 최적화를 위한 function 정의
@tf.function 
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = cross_entropy_loss(y_pred, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# 모델의 정확도를 출력하는 함수 정의
@tf.function 
def compute_accuracy(y_pred, y):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy

SoftmaxRegression_model = SoftmaxRegression()

for i in range(1000):
    batch_xs, batch_ys = next(train_data_iter)
    train_step(SoftmaxRegression_model, batch_xs, batch_ys)

print(f"정확도(Accuracy): {compute_accuracy(SoftmaxRegression_model(x_test), y_test)} ")