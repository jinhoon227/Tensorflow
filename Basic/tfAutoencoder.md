### 비지도 학습  

이때 까지 해온 지도 학습(supervised learning) 은 프로그램에게 입력값과 결과값을 주어서 학습하는 방법이다. 이번에 배울 비지도 학습(unsupervised learning)은 입력값으로부터 데이터의 특징을 찾아내는 학습이다. 이중에서 널리 쓰이는 방법이 오토인코더(Autoencoder) 이다.

### Autoencoder  
입력값과 출력값의 개수가 같으며 그 사이의 중간 계층의 노드 수는 입력값보다 적은것이 특징이다. 그래서 데이터를 압축하는 효과와 노이즈 제거 효과가 있다.  
  
입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 보내고, 은닉층의 데이터를 디코더를 통해 출력층으로 보낸 뒤, 만들어직 출력값을 입력값과 비슷해지는 가중치를 찾아내는것이 목적이다.

  
사용
>tfAutoencoder.ipynb

코드
```
#1
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)
old_v = tf.logging.get_verbosity()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot = True)

#2
learning_rate = 0.01
training_epoch = 20
batch_size = 100
n_hidden = 256
n_input = 28*28

#3
X=tf.placeholder(tf.float32, [None, n_input])

#4
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

#5
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode),b_decode))

#6
cost =tf.reduce_mean(tf.pow(X-decoder, 2))
#7
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#8
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch) :
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer,cost], feed_dict = {X: batch_xs})
        
        total_cost += cost_val
        
    print('Epoch:','%04d' % (epoch + 1),
         'Avg. cost=', '{:4f}'.format(total_cost / total_batch))
    
print('최적화 완료!')

#9
sample_size = 10

samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size,2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
    ax[1][i].imshow(np.reshape(samples[i],(28,28)))

    
plt.show()
```

1. 라이브러리 임포트  
numpy : 계산 도움용  
matplotlib : 그래프 출력용

2. 하이퍼파라미터 옵션 정의  
learning_rate : 최적화 함수에서 사용할 학습률  
training_epoch : 전체 데이터 학습 총횟수  
batch_size : 미니배치로써 한 번에 학습할 데이터 개수  
n_hidden : 은닉층의 뉴런 개수  
n_input : 입력값의 크기(이미지 크기가 28x28 이므로 784)

3. 신경망 모델 구성  
비지도 학습이므로 X의 플레이스홀더만 설정

4. 인코더 작성  
인코더 : n_hidden 개의 뉴런을 가진 은닉층을 만들고 가중치 및 편향 변수 설정 및 활성화 함수 적용(보통 입력값보다 적음 => 압축 및 노이즈 제거하면서 입력값 특징을 찾아냄)  

5. 디코더 작성  
디코더 : 입력값을 은닉층의 크기로, 출력값을 입력층의 크기로 인코더와 같이 작성

6. 손실 함수 작성  
출력값을 입력값과 비슷하게 만드는게 목적  
입력값인 X를 평가하기 위한 실측값으로 사용하고, 디코더가 내보낸 결괏값과의 차이를 손실값으로 설정한다. 

7. 최적화 함수 설정

8. 학습 코드 작성

9. matplotlib 이용한 이미지로 결과값 출력

#### 테스트 결과  
![오토인코더](../Image/B_AE1)
