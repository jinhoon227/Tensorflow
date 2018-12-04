### 텐서
1.텐서는 그래프 연산의 모음  
>텐서의 연산을 미리 정의 후 원하는 시점에서 실행 가능(지연 실해 방식)  
  
2.그래프 실행은 Sessoin 으로 수행

```
//라이브러리 임포트
import tensorflow as tf

//상수를 변수에 저장
hello = tf.constant('Hello, TensorFlow!')
print(hello)

//연산 수행(그래프 생성)
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a,b)
print(c)

//그래프 실행
sess = tf.Session()

print(sess.run(hello))
print(sess.run([a,b,c]))

sess.close()
```

### 플레이스 홀더


```
import tensorflow as tf  

//1
X = tf.plasceholder(tf.float32, [None, 3])

//2
x_data=[[1,2,3],[4,5,6]]

//3
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))

//4
expr=tf.matmul(X,W) + b

//5
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(x_data)
print(sess.run(W))
print(sess.run(b))

print(sess.run(expr, feed_dict={X: x_data})

sess.close()
```

1.플레이스홀더
>그래프에 사용할 입력값 을 나중에 받기 위해 사용하는 매개변수  
(float32 자료형을 가진 두 번째 차원 요소가 3개인 플레이스 홀더)  
  
2.플레이스홀더에 넣을 자료 정의  
3.변수 정의  
>정규분포의 무작위 값(0~1)로 [3, 2], [2 , 1]
  행렬로 초기화  

4.수식 작성
>행렬 곱을 위해 tf.matmul 함수 사용  

5.연산 실행 및 결과 확인
>tf.global_variables_initializer 을 이용해 앞서 정의한 변수 초기화(기존 학습한 값을 가져와 하는게 아닌 처음 실행이면 이 함수를 이용해 '연산' 전 변수 초기화 해야함)  

>feed_dict 매개변수는 그래프를 실행할 때 사용할 입력값을 지정  
(expr 수식에서 X는 플레이스홀더라서 X에 값을 넣어주지 않으면 안되므로 미리 정의한 x_data를 X에 넣어줌)