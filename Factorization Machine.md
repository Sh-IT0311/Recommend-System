### Factorization Machine(FM)
* 배경지식
    * 일반적인 데이터 분석에서 categorical feature의 경우 feature 내부에 여러 category로 구성되어 있지만 추천시스템에서는 표현방식이 약간 다름
    * categorical feature 자체가 One-Hot Encoding 되어 feature를 구성하는 category 자체가 feature가 되고, 이러한 feature들의 상위 개념이 field임
* Matrix factorization의 경우 (user, item, rating)으로 이루어진 튜플셋(matrix)를 사용했지만, 실제 real-world에서는 풍부한(feature-rich) 메타 데이터를 가지고 있는 데이터셋이 더 좋은 데이터셋으로 쓰이고 있으며 이러한 메타 데이터를 추천 시스템이 활용할 수 있도록 하는 것이 매우 요구됨
* Factorization Machine은 개인화 추천(personalized recommend)를 위해 polynomial regression을 추천 영역의 기술로 개량한 모델로, 앞에 언급한 feature-rich 데이터셋(meta data를 포함)을 사용하기 좋은 모델이고 그래서 더 많은 extra feature를 모델에 사용할 수 있음
    * FM이 모든 feature 간에 interaction을 계산하기 때문임
* Factorization Machine 구현의 경우, Matrix Factorization 계열 알고리즘의 아이디어와 polynomial regression 계열 알고리즘의 아이디어를 결합했음
    * polynomial regression보다 feature(변수)간에 더 복잡한 관계의 pairwise interaction(conjuction)을 잡아낼 수 있음
        * polynomial regression의 경우, 1개의 model parameter(weight)를 통해 feature간에 interaction(conjuction)을 설명해내야 하기 때문에 한계가 있는 반면에 factorization machine의 경우, feature에 대응되는 latent vector의 내적이라는 선형적인 기법을 통해 model parameter를 추정하기 때문에 feature간에 interaction의 더 복잡한 관계를 잡아낼 수 있다고 생각함
            * 더 복잡한 관계를 잡아낼 수 있다는 장점이 학습이 깊게 이루어지면 Overfitting에 취약할 수 있음으로 직결된다고 생각함
                * 여기서 overfitting이라고 하면, 데이터를 있는 그대로 외워버리는 문제가 발생한다는 의미임
                * 기본적으로 matrix factorization 계열의 알고리즘이기 때문에 overfitting에 취약해서 regularization이 요구됨
            * latent vector간에 내적을 유사도를 구하는 개념으로 해석하여 latent vector간에 유사도를 평가한다고 해석할 수 있음
* 또한 일반적으로 sparse 데이터셋의 경우 feature간에 관계가 매우 독립적으로 나타나는 경향이 있지만, 이러한 sparse 데이터셋의 FM을 사용할 경우 feature 간에 독립성 문제를 극복해서 매우 희소한 데이터에서도 (잠재적인)관계를 추정할 수 있음(cold-start 문제에 대응할 수 있음)
    * 예를 들어, A라는 유저가 X라는 영화를 평가한적이 없다고 할 때, polynomial regression 관점의 접근이라면 A는 X를 평가한 적이 없기 때문에 interaction이 0(= 독립적)이 나오겠지만, factorization machine의 경우 잠재적인 interaction을 얻어낼 수 있음
        * A라는 유저, X라는 영화는 추천시스템에서 feature로 해석됨
            * ∵ categorical feature가 One-Hot Encoding 되었기 때문
    * 여기서 feature 간에 독립성을 극복했다는 의미는 FM은 feature끼리 pairwise를 이루어서 학습하기 때문에 학습과정에서 하나의 feature가 또 다른 feature의 학습에 영향을 줄 수 있는 구조를 가지고 있기 때문임
        * 사람들은 각자 자신만의 행동 패턴이 있기 때문에 행동들 간에 연관성이 있을 것임. 그래서 이러한 연관성이라는 것을 모델이 이해할 수 있도록 feature간에 독립성 극복은 매우 중요하게 작용했을 것으로 생각됨
        * 하지만 이러한 feature간에 영향을 주는 구조가 일반화에 실패하는 overfitting에 취약할 수 있음
            * 그렇기 때문에 latent vector의 dimension을 나타내는 n_factor를 어느정도 작게 설정해야 하고 regularization가 요구됨
                * MF보다 n_factor를 작게 해야 성능이 좋다고 함
                    * FM이 가정하는 데이터셋에 경우 매우 Sparse 하기 때문에 매우 많은 feature가 형성되는데, 이러한 feature에 대응하는 latent vector가 많아진다는 것임. 즉, n_factor가 늘어날수록 parameter가 기하급수적으로 늘어 날 수 밖에 없을 것임
* FM의 model parameter 분석
    * w coef 분석
        * feature 자체의 영향력으로 볼 수 있음
    * v coef (n_factor에 대한)sum 분석
        * feature간에 interaction의 영향력으로 볼 수 있음
    * v coef (n_factor에 대한) 절댓값으로 변환하고 여기서 같은 종류(FFM에서는 Field)의 feature들끼리 모두 sum 분석
        * 절대값의 관점으로 feature간에 interaction의 영향력으로 볼 수 있음