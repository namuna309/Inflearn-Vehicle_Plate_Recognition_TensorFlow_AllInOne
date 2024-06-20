from __future__ import absolute_import, division, print_function, unicode_literals

from absl import app
import tensorflow as tf

import numpy as np
import os
import time

# input데이터와 target 데이터를 생성하는 utility 함수 정의
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]

    return input_text, target_text

# 학습에 필요한 설정값 지정
data_dir = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')  # shakespeare
batch_size = 64
seq_length = 100
embedding_dim = 256
hidden_size = 1024
num_epochs = 10

# 학습에 사용할 txt 파일 읽어오기
text = open(data_dir, 'rb').read().decode(encoding='utf-8')
# 학습데이터에 포함된 모든 character들을 나타내는 변수인 vocab과
# vocab에 id를 부여해 dict 형태로 만든 char2idx를 선언
vocab = sorted(set(text))
vocab_size = len(vocab)
print(f'{vocab_size} unique characters')
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 학습 데이터를 character에서 integer로 변환
text_as_int = np.array([char2idx[c] for c in text])

# split_input_target 함수를 이용하여 input 데이터와 target 데이터 생성
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옴
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)


# RNN 모델 정의
class RNN(tf.keras.Model):
    def __init__(self, batch_size):
        super(RNN, self).__init__()
        # layer 정의
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, 
                                                         embedding_dim, 
                                                        )
        self.hidden_layer_1 = tf.keras.layers.LSTM(hidden_size,
                                             return_sequences=True,
                                             stateful=True,
                                             recurrent_initializer='glorot_uniform')
        self.output_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x):
        embedded_input = self.embedding_layer(x)
        features = self.hidden_layer_1(embedded_input)
        logits = self.output_layer(features)

        return logits
    
# sparse cross-entropy 손실 함수 정의
def sparse_cross_entropy(labels, logits):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))

# 최적화를 위한 Adam 옵티마이저 정의
optimizer = tf.keras.optimizers.Adam()

# 최적화를 위한 function 정의
@tf.function
def train_step(model, input, target):
    with tf.GradientTape() as tape:
        logits = model(input)
        loss = sparse_cross_entropy(target, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def generate_text(model, start_string):
    num_sampling = 4000

    # start_string을 integer 형태로 변환
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 샘플링 결과로 생성된 string을 저장할 배열 초기화
    text_generated = []

    temperature = 1.0

    model.hidden_layer_1.reset_states()

    for i in range(num_sampling):
        predictions = model(input_eval)
        # 불필요한 batch dimension 삭제
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

def main(_):
    RNN_model = RNN(batch_size=batch_size)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = RNN_model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    RNN_model.summary()

    # checkpoint 데이터를 저장할 경로 지정
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

    for epoch in range(num_epochs):
        start = time.time()

        # 매 반복마다 hidden state 초기화
        hidden = RNN_model.hidden_layer_1.reset_states()
        for (batch_n, (input, target)) in enumerate(dataset):
            loss = train_step(RNN_model, input, target)

            if batch_n % 100 == 0:
                template = 'Epoch {} Batch {} Loss {}'
                print(template.format(epoch+1, batch_n, loss))
            
        # 5회 반복마다 파라미터를 checkpoint로 저장
        if (epoch + 1) % 5 == 0:
            RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))
        
        print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))
    print("트레이닝이 끝났습니다!")

    sampling_RNN_model = RNN(batch_size=1)
    sampling_RNN_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    sampling_RNN_model.build(tf.TensorShape([1, None]))
    sampling_RNN_model.summary()

    # 샘플링을 시작합니다.
    print("샘플링을 시작합니다!")
    print(generate_text(sampling_RNN_model, start_string=u' '))

if __name__ == '__main__':
  # main 함수를 호출합니다.
  app.run(main)