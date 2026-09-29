---
title: "GPT-2의 내부 연산 살펴보기: Residual Stream과 QK·OV의 유효 랭크"
categories: [mi]
tags: [Mechanistic Interpretability, GPT-2, TransformerLens, Residual Stream, Effective Rank]
mathjax: true
lightbox: true
permalink: /mi/gpt2-circuits-residual-stream/
---

GPT-2가 다음 단어를 예측하기까지, 토큰을 나타내는 벡터는 여러 레이어를 거치며 달라집니다. 이 글에서는 그 변화를 직접 추적하고, Attention을 구성하는 행렬의 랭크와 고윳값을 살펴봅니다. 특히 **헤드별 QK·OV는 768×768 행렬이지만 랭크가 최대 64**라는 점에 주목합니다.

<!--more-->

Mechanistic Interpretability 스터디의 첫 실험입니다. [Transformer Circuits 논문](https://transformer-circuits.pub/2021/framework/index.html)의 수식과 분석 방법을 바탕으로, residual stream의 변화와 가중치의 구조를 함께 살펴봅니다. 논문이 주로 다루는 얕은 attention-only 모델과는 달리, 실험에는 MLP를 포함한 사전학습 GPT-2를 사용합니다.

구현과 실행 결과는 [실험 노트북](https://github.com/era-s/Mechanistic-Interpretability-study/blob/main/notebooks/01_Circuits%26ResidualStream.ipynb)에 정리했습니다.

## 실험 설정

| 항목 | 설정 |
| --- | --- |
| 모델 | `openai-community/gpt2` |
| 도구 | PyTorch · TransformerLens `TransformerBridge` |
| 레이어 수 | 12 |
| 레이어당 헤드 수 | 12 |
| Residual 벡터의 차원 | 768 |
| 헤드 차원 | 64 |
| 어휘 크기 | 50,257 |
| 랭크 계산 | CPU, float64 |

수식에서는 입력을 열벡터로 두고 $Wx$ 순서로 계산합니다. TransformerLens에 저장된 가중치는 이 표기에 맞게 전치합니다.

## Mary와 John을 구분하는 문장

실험에는 다음 문장을 사용합니다.

> When Mary and John went to the store, John gave a drink to …

Mary와 John이 가게에 갔고, John이 누군가에게 음료를 주었다는 문장입니다. 모델이 문장 속 역할을 구분한다면, 주는 사람인 John에 이어 받는 사람인 ` Mary`를 예측해야 합니다. 이렇게 간접목적어를 식별하는 과제를 **IOI(Indirect Object Identification)**라고 합니다.

문장 시작 토큰인 BOS를 붙이지 않으면 입력은 14개 토큰으로 나뉩니다. 위치는 0번부터 13번까지이며, 마지막 ` to`의 출력으로 14번 위치에 올 토큰을 예측합니다. 이번 실행에서 ` Mary`의 확률은 **44.6005%**로 전체 어휘 중 가장 높았습니다. 생성할 때는 가장 높은 확률의 토큰을 고르는 greedy decoding을 사용했습니다.

이 문장은 모델 내부를 살펴보기 위한 예시입니다. IOI 과제 전체의 성능을 평가한 결과는 아닙니다.

## Residual stream에는 어떤 변화가 쌓이는가

각 토큰의 residual 벡터는 토큰 임베딩과 위치 임베딩의 합에서 시작합니다. 이후 Attention과 MLP의 출력이 더해지면서 레이어를 따라 변합니다. 이 실험에서는 **처음 임베딩에서 얼마나 달라졌는지**를 측정합니다.

$$
r_t^{(0)}=W_Ee_t+p_t,
\qquad
\Delta r_{\ell,t}=r_{\ell,t}^{\mathrm{post}}-r_t^{(0)}.
$$

$t$는 토큰 위치, $\ell$은 레이어 번호입니다. $e_t$는 해당 토큰을 나타내는 one-hot 열벡터이고, $W_Ee_t$와 $p_t$는 각각 토큰 임베딩과 위치 임베딩입니다. $r_{\ell,t}^{\mathrm{post}}$는 레이어 $\ell$을 통과한 뒤의 벡터입니다. 따라서 $\Delta r_{\ell,t}$는 최초 입력에서 해당 레이어까지 쌓인 변화이며, 최종 LayerNorm을 적용하기 전의 값으로 계산합니다.

아래 그림은 입력 문장을 한꺼번에 처리하는 **prefill**과, KV cache를 사용해 새 토큰 하나를 처리하는 **decode**를 나누어 보여줍니다.

- **Prefill:** 입력 토큰 14개 모두의 변화를 표시합니다.
- **Decode:** 첫 번째 단계로 고정하여, 새로 생성한 ` Mary`를 처리할 때의 변화를 맨 아래 행에 표시합니다. 앞선 토큰은 재계산하지 않으므로 화면에서 0으로 표시합니다. 그 토큰들의 실제 벡터나 누적 변화가 0이라는 뜻은 아닙니다.
- **Layer 슬라이더:** 레이어별 누적 변화를 확인할 수 있습니다. `Input`은 첫 레이어에 들어가기 전이므로 변화량이 0입니다.

가로축은 768개 벡터 좌표, 세로축은 토큰입니다. 큰 값과 작은 값을 함께 볼 수 있도록 색에는 부호를 보존하는 로그 변환을 적용했습니다.

$$
c(\Delta)=\operatorname{sign}(\Delta)\log_{10}(1+\lvert\Delta\rvert).
$$

모든 레이어에 같은 색 범위를 사용합니다. 색 막대와 마우스를 올렸을 때 나타나는 수치는 변환 전의 변화량입니다.

<iframe src="/assets/posts/gpt2-circuits-residual-stream/residual-stream.html" title="Residual stream: layer slider" loading="lazy" style="width:100%;height:760px;border:0;background:white"></iframe>

[인터랙티브 그림을 크게 열기](/assets/posts/gpt2-circuits-residual-stream/residual-stream.html)

이 그림으로 확인하는 것은 **누적 변화**입니다. 레이어 하나가 더한 변위를 비교하려면 그 레이어의 입력과 출력을 빼야 합니다.

$$
\delta r_{\ell,t}=r_{\ell,t}^{\mathrm{post}}-r_{\ell,t}^{\mathrm{pre}}.
$$

두 값을 구분해야 앞선 레이어에서 이미 생긴 변화를 현재 레이어의 기여로 해석하지 않을 수 있습니다.

### 언임베딩 행렬을 거쳐 다음 토큰을 고르기

다음 토큰을 예측할 때는 13번 토큰 ` to`의 마지막 residual 벡터를 사용합니다. 위 그림에 표시한 변화량에 최초 임베딩을 더하면 이 벡터를 복원할 수 있습니다. 여기에 최종 정규화를 적용한 뒤, 언임베딩 행렬 $W_U$를 곱합니다.

$$
\underbrace{r_{13}^{(0)}+\Delta r_{11,13}}_{\text{마지막 residual}}
\xrightarrow{\operatorname{LN}_f}
\underbrace{h_{13}}_{768}
\xrightarrow{\;W_U\;}
\underbrace{z_{13}=W_Uh_{13}}_{50257}
\xrightarrow{\operatorname{softmax}(\cdot/T)}
P_T(x_{14}\mid x_{0:13}).
$$

$W_U$의 크기는 $50257\times768$입니다. 각 행은 어휘의 토큰 하나에 대응하며, 그 행과 최종 표현 $h_{13}$의 내적이 해당 토큰의 **로짓(logit)**이 됩니다. 아래 히트맵에서 세로축은 토큰 ID, 가로축은 residual 좌표, 색은 가중치 값입니다.

![언임베딩 행렬: 토큰 ID × residual 좌표](/assets/posts/gpt2-circuits-residual-stream/unembedding.png)

로짓은 아직 확률이 아닙니다. 전체 어휘의 로짓에 softmax를 적용하면 다음 토큰의 확률 분포를 얻습니다.

$$
P_T(x_{14}=v\mid x_{0:13})=
\frac{\exp(z_{13,v}/T)}{\sum_{u\in\mathcal V}\exp(z_{13,u}/T)},
\qquad x_{14}\sim P_T(\cdot\mid x_{0:13}).
$$

$\mathcal V$는 전체 어휘 집합입니다. $x_{14}\sim P_T$는 이 분포에서 다음 토큰을 샘플링한다는 뜻입니다. 이번 실험은 샘플링 대신 로짓이 가장 큰 ` Mary`를 선택합니다.

아래 막대그래프는 상위 후보의 로짓과 확률을 보여줍니다. 확률은 상위 후보끼리 다시 계산한 값이 아니라, **전체 어휘에 softmax를 적용한 결과**입니다.

![위치 13의 상위 로짓과 다음 토큰 확률](/assets/posts/gpt2-circuits-residual-stream/next-token.png)

전체 어휘의 확률은 아래 그림에 빨간색으로 표시했습니다. 슬라이더로 temperature $T$를 0.1부터 10까지 바꿀 수 있습니다. $T=1$은 원래 확률이며, $T<1$이면 높은 확률의 토큰에 더 집중하고, $T>1$이면 분포가 평탄해집니다.

<iframe src="/assets/posts/gpt2-circuits-residual-stream/temperature.html" title="Full vocabulary probabilities: temperature slider" loading="lazy" style="width:100%;height:465px;border:0;background:white"></iframe>

[인터랙티브 그림을 크게 열기](/assets/posts/gpt2-circuits-residual-stream/temperature.html)

슬라이더는 저장된 로짓으로 확률만 다시 계산합니다. 세로축은 분포에 맞춰 자동으로 조정하며, 위의 상위 후보 막대그래프는 $T=1$로 유지합니다.

### 임베딩부터 출력까지의 계산

지금까지의 흐름을 한 식으로 정리하면 다음과 같습니다.

$$
\begin{aligned}
\text{Embedding:}\quad&r_t^{(0)}=W_Ee_t+p_t,\\
\text{Attention:}\quad&\widetilde r_t^{(\ell)}=r_t^{(\ell)}+
\operatorname{Attn}_{\ell}\!\left(\operatorname{LN}_{\ell,1}(r_{0:t}^{(\ell)})\right)_t,\\
\text{MLP:}\quad&r_t^{(\ell+1)}=\widetilde r_t^{(\ell)}+
W_{\ell,\mathrm{out}}\operatorname{GELU}\!\left(W_{\ell,\mathrm{in}}\operatorname{LN}_{\ell,2}(\widetilde r_t^{(\ell)})+b_{\ell,\mathrm{in}}\right)+b_{\ell,\mathrm{out}},\\
\text{Unembedding:}\quad&z_t=W_U\operatorname{LN}_f(r_t^{(12)}),\\
\text{Next-token probability:}\quad&P_T(x_{t+1}\mid x_{0:t})=\operatorname{softmax}(z_t/T).
\end{aligned}
$$

각 레이어에서 Attention은 앞선 위치와 현재 위치의 정보를 모으고, MLP는 각 토큰의 벡터를 변환합니다. 두 연산의 출력은 차례로 residual에 더해집니다. $W$와 $b$는 학습한 가중치와 편향, GELU는 비선형 활성화 함수입니다. 12개 레이어를 모두 거친 뒤에는 최종 정규화와 언임베딩, softmax를 통해 다음 토큰의 확률을 구합니다.

## Q·K·V·O는 몇 개의 방향을 사용하는가

이제 입력 문장에 따른 변화에서 가중치 행렬 자체의 구조로 시선을 옮깁니다. 먼저 각 레이어의 **전체 $768\times768$ Q·K·V·O 행렬**을 분석합니다.

헤드 하나의 Q·K·V는 768차원 벡터를 64차원으로 옮기고, O는 헤드의 출력을 다시 768차원 공간으로 보냅니다. 헤드 번호를 $h$로 쓰면 각 행렬의 크기는 다음과 같습니다.

$$
W_Q^h,W_K^h,W_V^h\in\mathbb{R}^{64\times768},
\qquad W_O^h\in\mathbb{R}^{768\times64}.
$$

전체 Q·K·V는 12개 헤드의 행렬을 세로로 이어 붙인 것이고, 전체 O는 가로로 이어 붙인 것입니다.

$$
W_X=\begin{bmatrix}W_X^0\\\vdots\\W_X^{11}\end{bmatrix},
\quad X\in\{Q,K,V\},
\qquad
W_O=\begin{bmatrix}W_O^0&\cdots&W_O^{11}\end{bmatrix}.
$$

### 수치 랭크: 독립적인 방향의 수

행렬의 랭크는 출력 공간에서 만들 수 있는 서로 독립적인 방향의 수입니다. 다만 컴퓨터에서는 아주 작은 값과 0을 구분하기 어려우므로, 기준값보다 큰 특잇값의 개수를 셉니다.

$$
\operatorname{rank}_{\mathrm{num}}(W)
=\#\{i:\sigma_i>\tau\},
\qquad
\tau=\sigma_{\max}\max(m,n)\epsilon_{64}.
$$

$\sigma_i$는 특잇값, $\sigma_{\max}$는 그중 최댓값입니다. $\#$는 조건을 만족하는 값의 개수이고, $\epsilon_{64}$는 float64의 머신 정밀도입니다. 이번 실험에서는 CPU float64로 `torch.linalg.matrix_rank`를 실행하고 기본 허용오차를 사용했습니다.

그 결과, **12개 레이어의 전체 Q·K·V·O 행렬은 모두 수치 랭크가 768**이었습니다. 수치 랭크만으로는 이 행렬들의 차이를 구분하기 어렵습니다.

### 유효 랭크: 각 방향에 실린 에너지의 분포

랭크가 같더라도 각 방향의 크기는 다를 수 있습니다. 일부 방향에 큰 값이 몰려 있는지, 여러 방향에 고르게 퍼져 있는지 살펴보기 위해 유효 랭크를 계산합니다.

먼저 특잇값 분해(SVD)로 행렬을 분해합니다.

$$
W=U\Sigma V^{\mathsf T},
\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_q),
\qquad
\sigma_1\geq\cdots\geq\sigma_q\geq0,
\quad q=\min(m,n).
$$

축약 SVD에서 $U$와 $V$의 열은 각각 출력과 입력 쪽의 직교 방향을 나타냅니다. 대각행렬 $\Sigma$에 놓인 특잇값 $\sigma_i$는 각 방향의 크기 배율입니다. $\mathsf T$는 전치를 뜻합니다.

이번 실험은 특잇값을 **제곱한 뒤 정규화**합니다.

$$
p_i=\frac{\sigma_i^2}{\sum_{j=1}^{q}\sigma_j^2},
\qquad
\sum_i p_i=1,
\qquad
\sum_i\sigma_i^2=\lVert W\rVert_F^2.
$$

$\lVert W\rVert_F^2$는 행렬의 모든 원소를 제곱해 더한 값입니다. 따라서 $p_i$는 전체 가중치 에너지에서 $i$번째 방향이 차지하는 비율입니다. 이 분포로 다음 세 지표를 계산합니다.

| 지표 | 정의 | 의미 |
| --- | --- | --- |
| Entropy effective rank | $\exp(-\sum_i p_i\log p_i)$ | 에너지 분포의 엔트로피를 차원 수로 환산합니다. |
| Participation ratio | $1/\sum_i p_i^2$ | 에너지가 적은 수의 방향에 집중될수록 작아집니다. |
| 95% explained-variance dimension | $\min\{k:\sum_{i=1}^{k}p_i\geq0.95\}$ | 전체 에너지의 95%를 설명하는 데 필요한 최소 차원 수입니다. |

$\log$는 자연로그이며 $0\log0=0$으로 둡니다. 에너지가 $r$개 방향에 균등하게 분포하면 앞의 두 지표는 모두 $r$이 됩니다. 영행렬에서는 세 지표를 모두 0으로 정의합니다.

여기서 ‘설명 분산’은 입력 데이터나 활성값의 분산이 아니라 **가중치의 특잇값 에너지**를 뜻합니다.

![전체 768×768 Q·K·V·O의 유효 랭크](/assets/posts/gpt2-circuits-residual-stream/full-effective-rank.png)

### 헤드 단위로 나누어 보기

전체 행렬을 헤드별로 나누면 $64\times768$인 Q·K·V와 $768\times64$인 O를 비교할 수 있습니다. 이 경우 세 지표의 상한은 모두 64입니다.

아래 그래프의 선은 레이어별 12개 헤드의 평균이고, 음영은 최솟값부터 최댓값까지의 범위입니다. 음영은 신뢰구간이 아니라 **같은 레이어 안에서 헤드마다 얼마나 다른지**를 보여줍니다.

![64차원 헤드별 Q·K·V·O의 유효 랭크](/assets/posts/gpt2-circuits-residual-stream/head-effective-rank.png)

## QK·OV는 왜 저랭크 행렬인가

Attention을 두 부분으로 나누어 보면, QK는 어느 토큰에 주목할지 결정하는 점수에 관여하고 OV는 그 토큰에서 가져온 정보를 residual에 더할 형태로 바꿉니다. 원 논문의 표기에 따라 두 회로를 다음과 같이 정의합니다.

$$
W_{QK}^h=(W_Q^h)^{\mathsf T}W_K^h,
\qquad
W_{OV}^h=W_O^hW_V^h.
$$

현재 위치와 정보를 가져올 위치의 벡터를 각각 $x_t,x_s$라 하면, 두 회로가 하는 일을 다음처럼 나타낼 수 있습니다.

$$
\mathrm{score}_{t,s}^h=x_t^{\mathsf T}W_{QK}^h x_s,
\qquad
\mathrm{output}_t^h=\sum_s\alpha_{t,s}^h W_{OV}^h x_s.
$$

$\alpha_{t,s}^h$는 인과 마스크와 softmax를 적용한 Attention 가중치입니다. 여기서는 회로의 가중치 구조를 보기 위해 bias와 LayerNorm을 제외하고, 원 논문처럼 점수의 상수 스케일링도 생략했습니다. 실제 GPT-2의 추론 계산은 그대로 사용합니다.

열벡터 표기에서는 오른쪽 행렬부터 작용하므로, V를 거쳐 O로 돌아오는 순서가 $W_OW_Vx$가 됩니다. 두 회로 모두 768차원 벡터에 작용하지만 중간에 64차원을 거칩니다. 따라서 행렬의 크기는 $768\times768$이어도 랭크는 64를 넘을 수 없습니다.

$$
W_{QK}^h=(W_Q^h)^{\mathsf T}W_K^h,
\qquad
W_{OV}^h=W_O^hW_V^h,
\qquad
\operatorname{rank}(W_{QK}^h),\operatorname{rank}(W_{OV}^h)\leq64\ll768.
$$

이 성질은 외적의 합으로도 확인할 수 있습니다.

$$
W_{QK}^h=\sum_{a=1}^{64}q_a k_a^{\mathsf T},
\qquad
W_{OV}^h=\sum_{a=1}^{64}o_a v_a^{\mathsf T}.
$$

$q_a^{\mathsf T},k_a^{\mathsf T},v_a^{\mathsf T}$는 각각 Q·K·V 행렬의 $a$번째 행이고, $o_a$는 O 행렬의 $a$번째 열입니다. 외적 하나의 랭크는 최대 1이므로, 64개를 더한 행렬의 랭크도 최대 64입니다.

### 행렬곱과 텐서곱의 차이

QK·OV를 계산하는 **행렬곱**은 두 행렬이 공유하는 차원을 합산합니다. $A\in\mathbb R^{m\times k}$, $B\in\mathbb R^{k\times n}$이라면 다음과 같습니다.

$$
(AB)_{ij}=\sum_{a=1}^{k}A_{ia}B_{aj},
\qquad AB\in\mathbb{R}^{m\times n}.
$$

반면 두 열벡터의 **외적**은 모든 성분 쌍을 곱해 행렬을 만듭니다.

$$
(uv^{\mathsf T})_{ij}=u_i v_j.
$$

벡터의 텐서곱 $u\otimes v$를 행렬로 표현하면 이 외적에 대응합니다. 행렬의 텐서곱을 나타내는 Kronecker 곱 $A\otimes B$는 $A$의 각 원소 $A_{ij}$를 블록 $A_{ij}B$로 바꾸는 연산입니다.

원 논문의 $A^h\otimes W_{OV}^h$도 같은 방식으로 읽을 수 있습니다. $A^h$는 토큰 사이에 정보를 어떻게 전달할지 정하고, $W_{OV}^h$는 전달할 벡터를 어떻게 바꿀지 정합니다.

$$
(A^h\otimes W_{OV}^h)
\begin{bmatrix}x_1\\\vdots\\x_T\end{bmatrix}
=
\begin{bmatrix}
\sum_s A_{1s}^hW_{OV}^hx_s\\
\vdots\\
\sum_s A_{Ts}^hW_{OV}^hx_s
\end{bmatrix}.
$$

이 식에서 $T$는 토큰 수이며, 앞서 사용한 temperature와는 다른 기호입니다. $A_{ts}^h=\alpha_{t,s}^h$이고, 전체 연산의 크기는 $(768T)\times(768T)$입니다. 이렇게 토큰 사이의 연산과 벡터 좌표 사이의 연산을 함께 표현할 수 있습니다.

### QK·OV의 랭크를 실제로 계산하기

`torch.linalg.matrix_rank`로 계산한 결과, **모든 레이어와 헤드의 QK·OV 수치 랭크는 64**였습니다. 이어서 앞과 같은 $\sigma^2$ 정규화로 세 유효 랭크를 계산했습니다.

![QK·OV 회로의 유효 랭크](/assets/posts/gpt2-circuits-residual-stream/circuit-effective-rank.png)

선은 헤드 평균, 음영은 헤드별 최솟값과 최댓값의 범위입니다. 수치 랭크가 모두 64여도 에너지의 분포는 서로 다릅니다. 다만 유효 랭크가 작다고 해서 덜 중요한 헤드라는 뜻은 아닙니다. 이 지표는 가중치 에너지의 집중도를 측정하며, 특정 입력에서 그 헤드가 하는 일의 중요성은 별도로 확인해야 합니다.

## OV 고윳값으로 살펴보는 방향의 보존

유효 랭크가 에너지의 분포를 보여준다면, 고윳값은 특정 방향이 변환될 때 크기와 부호가 어떻게 바뀌는지 보여줍니다. 헤드별 OV를 $M=W_O^hW_V^h$라 두고 오른쪽 고유벡터를 구합니다.

$$
Mv=\lambda v,
\qquad v\ne0.
$$

$v$ 방향의 입력은 $\lambda v$로 변합니다. OV는 일반적으로 비대칭이므로 실수 고윳값뿐 아니라 복소 고윳값도 나올 수 있습니다. 따라서 `eig`로 계산하고 복소수 결과도 유지합니다.

- 양의 실수 고윳값은 해당 고유방향의 부호를 보존합니다. $0<\lambda<1$이면 크기를 줄이고, $\lambda>1$이면 키웁니다.
- 음의 실수 고윳값은 해당 방향의 부호를 뒤집습니다.
- 복소 고윳값은 실수부가 양수여도 양의 실수 고윳값으로 분류하지 않습니다.

계산에는 저랭크 구조를 다시 활용할 수 있습니다. $A=W_O^h$, $B=W_V^h$로 두면 다음 관계가 성립합니다.

$$
BAu=\lambda u
\quad\Longrightarrow\quad
AB(Au)=A(BAu)=\lambda(Au).
$$

$AB$와 $BA$의 0이 아닌 고윳값은 같으므로, $768\times768$ 행렬 대신 $64\times64$ 행렬에서 계산할 수 있습니다. 0이 아닌 고윳값에 대응하는 고유벡터는 $v=Au$로 복원합니다. 원래 행렬에 존재하는 최소 704개의 0 고윳값은 그림에서 제외합니다.

첫 번째 그림은 헤드마다 **0이 아닌 고윳값 중 양의 실수 고윳값이 차지하는 비율**을 보여줍니다.

![헤드별 양의 실수 고윳값 비율](/assets/posts/gpt2-circuits-residual-stream/ov-positive-eigenvalues.png)

두 번째 그림은 [원 논문의 원형 산점도](https://transformer-circuits.pub/2021/framework/index.html#copying-matrix) 형식을 따라 전체 레이어와 헤드를 펼쳐 놓은 것입니다. 행은 레이어, 열은 헤드입니다. 점 하나가 0이 아닌 고윳값 하나에 대응하며, 다음 좌표로 표시합니다.

$$
\theta=\arg(\lambda),\qquad \rho=\log_{10}\lvert\lambda\rvert+C.
$$

$\arg(\lambda)$는 복소평면에서의 각도입니다. 양의 실수는 오른쪽, 음의 실수는 왼쪽에 놓입니다. 반지름에는 크기의 로그를 사용하고, 공통 상수 $C$를 더해 양수로 옮깁니다. 동심원의 눈금은 원래 크기 $\lvert\lambda\rvert$를 나타냅니다. 모든 칸에 같은 척도를 적용했으며, 주황색은 양의 실수 고윳값, 파란색은 나머지 고윳값입니다.

![전체 12개 레이어와 12개 헤드의 OV 고윳값](/assets/posts/gpt2-circuits-residual-stream/ov-eigenspectra.svg)

여기서 분석한 행렬은 residual 공간의 $W_OW_V$입니다. 양의 고윳값은 **해당 고유방향을 부호 변화 없이 전달한다는 단서**입니다. 특정 토큰을 복사한다고 결론 내리려면, 임베딩과 언임베딩을 연결한 $W_UW_OW_VW_E$뿐 아니라 실제 Attention 가중치와 로짓의 변화도 함께 확인해야 합니다.

## 실험에서 확인한 점

**첫째, residual의 변화를 임베딩 기준으로 추적할 수 있습니다.** 이번 그림은 $\Delta r_{\ell,t}=r_{\ell,t}^{\mathrm{post}}-r_t^{(0)}$를 표시합니다. 레이어 하나가 더한 변위와는 구분해야 합니다. Prefill은 입력 토큰 전체를 처리하고, KV cache를 쓰는 decode는 새 토큰만 처리합니다.

**둘째, 수치 랭크가 같아도 유효 랭크는 다를 수 있습니다.** 전체 Q·K·V·O의 수치 랭크는 모두 768이었습니다. 12개 레이어의 전체 Q·K·V·O 행렬에서 entropy rank는 약 **160~416**, participation ratio는 **27~331**, 95% 설명 차원은 **202~459**였습니다. 수치 랭크 768보다 작아, 가중치 에너지가 일부 방향에 집중되어 있음을 보여줍니다.

**셋째, QK·OV의 핵심은 저랭크 구조입니다.** 헤드별 회로는 $768\times768$ 행렬이지만 $\operatorname{rank}\leq64$입니다. 이번 실험에서도 모든 헤드의 수치 랭크가 64로 확인됐습니다.

**넷째, 양의 OV 고윳값만으로 토큰 복사를 단정할 수는 없습니다.** 고유방향의 보존과 토큰 복사는 서로 다른 분석 수준입니다. 특정 기능에 대한 인과적 기여는 구성 요소를 제거하는 ablation이나 활성값을 바꾸는 activation patching으로 더 살펴봐야 합니다.

## 다음 실험에서 살펴볼 질문

1. Residual stream에 더해지는 변위는 레이어마다 어떻게 다릅니까?

   a. 차이가 생기는 이유는 무엇입니까?

2. 레이어마다 맡는 역할이 구분되어 있습니까?

3. Q·K·V·O 이외의 행렬은 어떤 유효 랭크를 보입니까?

4. 레이어별 변환과 유효 랭크에는 모델 사이에 공통된 특성이 있습니까? 다음 범위에서 비교해 볼 수 있습니다.

   a. 같은 GPT-2 계열의 모델

   b. 같은 MHA 구조를 사용하는 모델

   c. 같은 트랜스포머 계열의 모델

   d. 같은 데이터셋으로 학습한 모델

## 참고 자료

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [실험 저장소: Mechanistic Interpretability Study](https://github.com/era-s/Mechanistic-Interpretability-study)
- [GPT-2 모델](https://huggingface.co/openai-community/gpt2)

그림은 실험 노트북에 저장된 실행 결과를 사용했습니다.
