### Field-aware Factorization Machine(FFM)
* FM에서는 다른 feature들과의 latent effect를 학습하는 latent vector가 모든 feature마다 오직 한개씩 존재하는데, 이러한 한개의 latent vector로 각각 다른 field에 해당하는 feature와 관계를 학습하는 것은 부적절함
    * 다른 field에 해당 하는 feature와는 다른 관계 형성을 이끌어내야 됨
* FM의 latent vector에 차원을 하나 추가해서 각각 다른 field에 해당하는 feature에 다른 latent vector를 사용할 수 있도록 함
    * 상황에 맞는 더욱 정교한 조합이 가능해짐
* Field-aware Factorization Machines for CTR Prediction 논문 리뷰
    * 배경지식
        * CTR
            * Click-Through Rate
            * (클릭 횟수 / 광고 노출 횟수) * 100
        * CVR
            * ConVersion Rate
            * (전환수(ex> 구매) / 클릭 횟수) * 100
        * feature들을 그룹짓고 상위 개념으로 field로 작명함
            * 추가적인 정보를 획득했다고 이해할 수 있음
            * ex> field : Publisher, features : NBC, Vogue, ESPN
            * categorical feature에 대해서는 field를 지정해주는 것이 쉽지만, Numerical Feature, Single-field Features(ex>자연어)에 대해서는 어려움
    * CTR 예측은 광고 산업에서 매우 중요한 역할을 담당하고 있음
        * CTR 예측은 클릭을 했는지 안했는지 binary classification 문제이기 때문에 logistic regression이 CTR 예측에 많이 쓰이는 모델로 언급되었음
        * 이러한 logistic regression을 최적화 하는 방법으로 logistic loss가 언급되었는데, binary classification에 보통 쓰이는 binary cross entropy(= log loss)와 형태가 약간 다름
            * binary cross entropy의 경우 target variable이 log 밖에 있음
            * 논문에 언급된 logistic loss는 target variable이 log 안에 있지만, 학습 자체(특히 그레이디언트)는 binary cross entropy와 같음
            * 논문에 언급된 logistic loss는 y = 0에 대해서는 학습이 이루어지지 않음
                * y = 1일때만 학습에 초점을 맞춘 것으로 생각함
    * CTR 예측에 있어서 중요한 것은 모델이 feature 간에 interaction(conjuction)을 이해하는 것임
        * simple linear model을 이용한 logistic regression은 이러한 conjuction을 잘 이해하지 못함
        * 그래서 이러한 feature간에 conjuction을 학습하도록 고려한 모델들이 poly2, FM임
            * sparse dataset에 있어서 FM이 poly2보다 우수하다고 함
                * FM 모델은 latent vector의 내적으로 model parameter를 산출하기 때문에 잠재적인 coefficient를 찾아낼 수 있음
                * FM 모델은 feature간에 독립성을 극복한 것도 성능 향상에 기여했음
        * FFM은 FM의 변종으로, feature들을 그룹짓는 field 정보를 활용했음
            * FFM이 FM보다 우수한 이유는 위에서 언급했음
                * 다른 field에 해당 하는 feature와는 다른 관계 형성을 이끌어내야 됨
    * FFM 구현의 특징
        * 최적화를 위해서 stochastic gradient를 사용함
            * 구현에 편의성 때문에 SG 썼다고 함
        * 행렬 분해에 효과적이라서 AdaGrad를 적용했다고 함
            * AdaGrad
                * 이전 그레이디언트의 제곱 스케일을 누적한 벡터를 이용해서 그레이디언트의 스케일을 조정함
                    * 누적한 벡터가 sqrt 형태의 분모로 들어가서 누적 벡터가 크면 그레이디언트의 스케일이 작아지고, 누적 벡터가 크면 그레이디언트의 스케일이 커짐
            * 누적 벡터의 초깃값을 1로 설정했다고함
                * sqrt 형태의 분모로 활용되기 때문에 0과 같은 작은 값으로 초기화하면 매우 커지기 때문임
        * Overfitting을 방지하기 위해.. 
            * FM 모델보다 n_factor를 적게 설정함
            * early-stopping를 사용했다고 함
                * 중단한 epoch가 test set에게 적절할지는 확신할 수 없음
                * FFM 모델이 number of epochs에 민감하다고 함
            * matrix factorization 계열의 알고리즘이기 때문에 regularization가 요구됨
        * normalization 해주는 것이 성능 향상에 도움이 되었다고 함
        * 논문에서 언급한 식에서는 없지만, bias와 linear term은 사용했다고 함
            * 일부 데이터에서 성능 향상을 이뤄냈다고 함
        * 초기 latent vector들은 [0,1/sqrt(n_factor)] 사이의 Uniform Distribution에서 랜덤한 값으로 설정했다고 함
        * Hog_WILD! 라는 병렬처리 기법을 활용했다고 함
    * FFM의 최적화 문제는 일반적인 binary classification 문제에서 모델을 FFM을 썼다는 것밖에 차이가 없음
    * 평가
        * logistic loss
            * cost function과 같이 y = 0에서는 적절한 평가가 이루어지지 않음
    * 결과
        * 데이터셋 대부분이 categorical feature로 구성되어 있고 매우 Sparse 할 경우 FFM 성능이 잘 나옴
            * 데이터셋이 sparse 하지 않는 경우 linear 모델과 성능 차이가 나지 않았다고 함
            * Numerical feature에 대해서는 큰 이득을 보기 어려움
                * FFM의 단점이라기 보단 feature engineering이 요구되는 문제라고 생각함