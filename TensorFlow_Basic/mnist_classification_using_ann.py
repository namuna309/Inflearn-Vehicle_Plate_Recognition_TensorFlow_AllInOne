import tensorflow as tf

# MNIST 데이터 다운로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# 이미지(uint8)들을 float32 데이터 타입으로 변경
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# 28*28 형태의 이미지를 784(=28*28)차원으로 flattening
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
# [0, 255](unint8: 0~255) 사이의 값을 [0, 1] 사이의 값으로 Normlize
x_train, x_test = x_train / 255., x_test/255.
# 레이블 데이터에 one-hot encoding 적용
y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)

# 학습을 위한 설정값 정의
learning_rate = 0.001
num_epochs = 30         # 학습횟수
batch_size = 256        # 배치개수
display_step = 1        # 손실함수 출력 주기
input_size = 784        # 입력(28*28) 층 크기
hidden1_size = 256      # hidden layer 층 (1층) 크기
hidden2_size = 256      # hiddne layer 층 (2층) 크기
output_size = 10        # 출력(0~9) 층 크기

# tf.data API를 이용하여 데이터를 섞고 batch 형태로 가져옴
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(60000).batch(batch_size)

# 표준분포로 초기화
def random_normal_initializer_with_stddev_1():
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)

# tf.keras.Model을 이용하여 ANN 모델 정의
class ANN(tf.keras.Model):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_layer1 = tf.keras.layers.Dense(hidden1_size,
                                                   activation='relu',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1()
                                                   )
        self.hidden_layer2 = tf.keras.layers.Dense(hidden2_size,
                                                   activation='relu',
                                                   kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                   bias_initializer=random_normal_initializer_with_stddev_1()
                                                   )
        self.output_layer = tf.keras.layers.Dense(output_size,
                                                activation=None,
                                                kernel_initializer=random_normal_initializer_with_stddev_1(),
                                                bias_initializer=random_normal_initializer_with_stddev_1()
                                                )
        
    def call(self, x):
        H1_output = self.hidden_layer1(x)
        H2_output = self.hidden_layer2(H1_output)
        logits = self.output_layer(H2_output)

        return logits
    


# cross-entropy 손실 함수 정의
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))


# 최적화를 위한 Adam 옵티마이저 정의
optimizer = tf._optimizers.Adam(learning_rate)


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
    accracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accracy

# ANN 모델 선언
ANN_model = ANN()

# 지정된 횟수만큼 최적화 수행
for epoch in range(num_epochs):
    average_loss = 0.
    total_batch = int(x_train.shape[0] / batch_size)

    # 모든 배치들에 대해서 최적화 수행
    for batch_x, batch_y in train_data:
        # 옵티마이저를 실행하여 파라미터 업데이트
        _, current_loss = train_step(ANN_model, batch_x, batch_y), cross_entropy_loss(ANN_model(batch_x), batch_y)
        # 평균 손실 측정
        average_loss += current_loss / total_batch

    if epoch % display_step == 0:
        print("반복(Epoch): %d, 손실 함수(Loss): %f" % ((epoch+1), average_loss))

print(f"정확도(Accuracy): {compute_accuracy(ANN_model(x_test), y_test)} ")