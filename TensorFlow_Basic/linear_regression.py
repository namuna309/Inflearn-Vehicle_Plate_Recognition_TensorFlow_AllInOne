

# 라이브러리 import
import tensorflow as tf

# 선형회귀 모델(Wx + b)을 위한 tf.Variable 선언
W = tf.Variable(tf.random.normal(shape=[1]))    # 가중치
b = w = tf.Variable(tf.random.normal(shape=[1]))    # bias

# 선형회귀 모델 정의
@tf.function
def linear_model(x):
    return W*x + b

# 손실함수 정의: RMSE: sqrt(mean((y' - y)^2))
@tf.function
def loss_function(y_pred, y):
    return tf.math.sqrt(tf.reduce_mean(tf.square(y_pred - y)))

# 최적화를 위한 gradient descent optimizer 정의, learnig rate = 0.01
optimizer = tf.optimizers.SGD(0.01)

# 최적화를 위한 function을 정의
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = linear_model(x)
        loss = loss_function(y_pred, y)
    # gradient 계산
    gradients = tape.gradient(loss, [W, b])
    # 가중치 및 bias 업데이트
    optimizer.apply_gradients(zip(gradients, [W, b]))

# 트레이닝을위 한 train data set 정의
x_train = [1, 2, 3, 4]
y_train = [2, 4, 6, 8]

# 모델 학습 1000번 수행
for i in range(1000):
    train_step(x_train, y_train)

# test data set 정의
x_test = [3.5, 5, 5.5, 6]

# 테스트 데이터를 이용해 학습된 선형회귀 모델이 데이터의 경향성(y=2x)을 잘 학습했는지 측정
# 예상되는 참값 : [7, 10, 11, 12]
print(linear_model(x_test).numpy())