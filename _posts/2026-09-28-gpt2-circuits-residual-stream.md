---
title: "GPT-2 내부 들여다보기: Residual Stream, 유효 랭크, QK·OV 회로"
categories: [mi]
tags: [Mechanistic Interpretability, GPT-2, TransformerLens, Residual Stream, Effective Rank]
mathjax: true
lightbox: true
permalink: /mi/gpt2-circuits-residual-stream/
---

트랜스포머의 내부에서는 토큰 벡터가 어떻게 변하고, Attention의 가중치는 얼마나 많은 방향을 사용합니까? GPT-2에서 residual stream을 추적하고 Q·K·V·O와 QK·OV 회로의 랭크를 계산해 살펴봅니다. 전체 projection은 수치적으로 full-rank이지만, 헤드별 QK·OV는 **768×768 크기에 랭크가 64인 저랭크 행렬**입니다.

<!--more-->

이 글은 Mechanistic Interpretability 스터디 1주차 실험을 정리한 기록입니다. [Transformer Circuits 원 논문](https://transformer-circuits.pub/2021/framework/index.html)의 관점을 실제 GPT-2에 적용합니다. 원 논문의 주요 분석 대상인 얕은 attention-only 모델과 달리, 여기서는 MLP를 포함한 사전학습 GPT-2를 사용합니다.

[실험 노트북](https://github.com/era-s/Mechanistic-Interpretability-study/blob/main/notebooks/01_Circuits%26ResidualStream.ipynb)에서 구현을 확인할 수 있습니다.

## 실험 설정

| 항목 | 설정 |
| --- | --- |
| 모델 | `openai-community/gpt2` |
| 도구 | PyTorch · TransformerLens `TransformerBridge` |
| 레이어 / 헤드 | 12 / 레이어당 12 |
| Residual / 헤드 차원 | 768 / 64 |
| 어휘 크기 | 50,257 |
| 랭크 계산 | CPU float64 |

수식은 입력을 열벡터로 두는 $Wx$ 표기를 사용합니다. TransformerLens에 저장된 가중치를 전치하여 이 표기에 맞춥니다.

## IOI 문장으로 다음 토큰 예측하기

> When Mary and John went to the store, John gave a drink to …

IOI(Indirect Object Identification)는 문장에서 받는 사람에 해당하는 간접목적어를 식별하는 과제입니다. 이 예에서는 주는 사람인 John과 받는 사람인 Mary의 역할을 구분하여 ` Mary`가 이어지는지 확인합니다. 하나의 문장에 대한 관찰이므로 IOI 과제 전체의 성능을 평가한 결과는 아닙니다.

BOS를 붙이지 않고 토큰화하면 입력은 위치 0–13을 차지합니다. 마지막 ` to`의 출력으로 위치 14의 토큰을 예측하며, 이번 실행에서 ` Mary`는 전체 어휘 중 1위, 확률 **44.6005%**였습니다. 생성은 greedy decoding으로 수행했습니다.

## Residual Stream: 임베딩 기준 누적 diff
토큰 위치를 $t$, 레이어를 $\ell$로 표시합니다. 최초 residual 입력은 토큰 임베딩과 위치 임베딩의 합입니다.

$$
r_t^{(0)}=W_Ee_t+p_t,
\qquad
\Delta r_{\ell,t}=r_{\ell,t}^{\mathrm{post}}-r_t^{(0)}.
$$

$e_t$는 위치 $t$의 토큰을 나타내는 one-hot 열벡터입니다. $W_Ee_t$는 토큰 임베딩이며, $p_t$는 위치 임베딩입니다.
$r_{\ell,t}^{\mathrm{post}}$는 레이어 $\ell$을 통과한 토큰 $t$의 residual 벡터입니다.
$\Delta r_{\ell,t}\in\mathbb{R}^{768}$은 **최초 임베딩부터 해당 레이어 출력까지의 누적 변화**를 나타내며, 최종 LayerNorm 이전 값으로 계산합니다.

- **Prefill:** 모든 입력 토큰의 누적 diff를 표시합니다.
- **Decode:** step 1로 고정하여 첫 생성 토큰인 맨 아래 행만 누적 diff를 표시합니다. 이전 토큰 행은 재계산하지 않는 위치를 나타내기 위해 화면에서 0으로 처리합니다. 실제 누적 diff가 0이라는 뜻은 아닙니다.
- **슬라이더:** `Layer`로 레이어를 바꾸면 prefill의 모든 행과 decode의 맨 아래 행이 변합니다. `Input`에서는 현재 처리하는 토큰의 diff가 0입니다.

가로축은 벡터 좌표, 세로축은 토큰입니다. 모든 레이어에 공통 색 범위를 적용합니다.
큰 값과 작은 값을 함께 보기 위해 색상에는 다음 변환을 사용합니다.

$$
c(\Delta)=\operatorname{sign}(\Delta)\log_{10}(1+|\Delta|).
$$

$\operatorname{sign}$은 부호를 보존하고, $\log_{10}$은 큰 값의 표시 범위를 압축합니다. 색 막대와 마우스 오버에는 변환 전 diff를 표시합니다.

<iframe src="/assets/posts/gpt2-circuits-residual-stream/residual-stream.html" title="Residual stream: layer slider" loading="lazy" style="width:100%;height:760px;border:0;background:white"></iframe>

[인터랙티브 그림을 크게 열기](/assets/posts/gpt2-circuits-residual-stream/residual-stream.html)

여기서 색은 레이어 하나가 추가한 변화가 아니라 **임베딩부터 누적된 변화**입니다. 레이어 자체의 기여를 비교하려면 연속 레이어의 차이를 구해야 합니다.

$$
\delta r_{\ell,t}=r_{\ell,t}^{\mathrm{post}}-r_{\ell,t}^{\mathrm{pre}}.
$$

또한 decode에서 이전 행을 0으로 표시한 것은 KV cache로 재계산을 생략했다는 시각적 약속입니다. 과거 토큰의 residual이 실제로 0이 된다는 뜻은 아닙니다.

### 언임베딩에서 다음 토큰 선택까지
위에서 추적한 **13번 토큰 ` to`의 마지막 residual 벡터**를 최종 정규화한 뒤, 아래 언임베딩 행렬 $W_U$에 곱해 전체 어휘의 logits를 구합니다. 앞의 그림은 임베딩 기준 diff이므로 실제 입력은 $r_{11,13}^{\mathrm{post}}=r_{13}^{(0)}+\Delta r_{11,13}$입니다.

$$
\underbrace{r_{13}^{(0)}+\Delta r_{11,13}}_{\text{마지막 residual}}
\xrightarrow{\operatorname{LN}_f}
\underbrace{h_{13}}_{768}
\xrightarrow{\;W_U\;}
\underbrace{z_{13}=W_Uh_{13}}_{50257}
\xrightarrow{\operatorname{softmax}(\cdot/T)}
P_T(x_{14}\mid x_{0:13}).
$$

$W_U\in\mathbb R^{50257\times768}$의 각 행과 $h_{13}$의 내적이 해당 토큰의 logit이 됩니다. 히트맵의 세로축은 토큰 ID, 가로축은 residual 좌표이며, 색은 가중치 값입니다.

![언임베딩 행렬: 토큰 ID × residual 좌표](/assets/posts/gpt2-circuits-residual-stream/unembedding.png)

이 행렬 곱으로 얻은 logits와 다음 토큰 확률을 아래에 표시합니다.

$$
P_T(x_{14}=v\mid x_{0:13})=
\frac{\exp(z_{13,v}/T)}{\sum_{u\in\mathcal V}\exp(z_{13,u}/T)},
\qquad x_{14}\sim P_T(\cdot\mid x_{0:13}).
$$

$\mathcal V$는 전체 어휘 집합이며, $\sim$는 이 확률 분포에서 토큰을 샘플링한다는 뜻입니다. **현재 실험에서는 greedy decoding으로 $\arg\max_v z_{13,v}$인 ` Mary`를 선택합니다.**

확률은 빨간색으로 표시합니다. 전체 어휘 그림의 슬라이더로 $T\in[0.1,10]$을 조절합니다. $T=1$은 원래 확률이며, $T<1$이면 분포가 뾰족해지고 $T>1$이면 평탄해집니다. 세로축은 자동 조정하며 상위 후보 막대그래프는 $T=1$로 고정합니다.

![위치 13의 상위 logits와 다음 토큰 확률](/assets/posts/gpt2-circuits-residual-stream/next-token.png)

<iframe src="/assets/posts/gpt2-circuits-residual-stream/temperature.html" title="Full vocabulary probabilities: temperature slider" loading="lazy" style="width:100%;height:465px;border:0;background:white"></iframe>

[인터랙티브 그림을 크게 열기](/assets/posts/gpt2-circuits-residual-stream/temperature.html)

슬라이더는 저장된 logits에 temperature를 적용합니다. 모델을 다시 실행하거나 생성 토큰을 다시 샘플링하지는 않습니다.

### 전체 계산 흐름
열벡터 표기로 GPT-2의 임베딩부터 다음 토큰 확률까지 정리합니다.

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

$e_t$는 토큰의 one-hot 벡터, $p_t$는 위치 임베딩이며, $\ell=0,\ldots,11$입니다. Attention은 위치 $0$부터 $t$까지의 정보를 모으고, MLP는 각 토큰의 벡터를 변환합니다. 두 출력은 residual에 더해집니다. $W,b$는 학습한 가중치와 편향이며, GELU는 비선형 활성화 함수입니다. 마지막으로 언임베딩과 softmax를 거쳐 다음 토큰의 확률을 구합니다.

## Q, K, V, O: Rank → Effective Rank
각 레이어의 **전체 $768\times768$ 가중치 행렬**을 분석합니다. 헤드 번호를 $h=0,\ldots,11$로 표시하면 헤드별 행렬 크기는 다음과 같습니다.

$$
W_Q^h,W_K^h,W_V^h\in\mathbb{R}^{64\times768},
\qquad W_O^h\in\mathbb{R}^{768\times64}.
$$

$\mathbb{R}^{m\times n}$은 실수 원소를 갖는 $m$행 $n$열 행렬을 의미합니다. $q=W_Q^hx$처럼 행렬을 열벡터의 왼쪽에 곱합니다.
Q·K·V는 768차원 입력을 헤드별 64차원으로 보내고, O는 헤드 출력을 768차원 residual 공간으로 보냅니다.
전체 행렬은 Q·K·V를 세로로, O를 가로로 연결하여 복원합니다.

$$
W_X=\begin{bmatrix}W_X^0\\\vdots\\W_X^{11}\end{bmatrix},
\quad X\in\{Q,K,V\},
\qquad
W_O=\begin{bmatrix}W_O^0&\cdots&W_O^{11}\end{bmatrix}.
$$

### 1. torch.linalg.matrix_rank
랭크는 행렬이 출력할 수 있는 서로 독립적인 방향의 수입니다. 수치 계산에서는 충분히 큰 특잇값의 개수로 판정합니다.

$$
\operatorname{rank}_{\mathrm{num}}(W)
=\#\{i:\sigma_i>\tau\},
\qquad
\tau=\sigma_{\max}\max(m,n)\epsilon_{64}.
$$

$\sigma_i$는 $W$의 $i$번째 특잇값, $\sigma_{\max}$는 가장 큰 특잇값입니다. $\#$는 조건을 만족하는 원소의 개수를 의미합니다.
$\epsilon_{64}$는 float64의 머신 정밀도이며, $\tau$보다 작은 값은 수치적으로 0으로 취급합니다.
`torch.linalg.matrix_rank`에 CPU float64의 기본 허용오차를 적용합니다.

실제로 계산한 12개 레이어의 Q·K·V·O 수치 랭크는 **모두 768**입니다.

### 2. Effective Rank
수치 랭크가 방향의 **개수**를 센다면, 유효 랭크는 각 방향에 에너지가 **얼마나 분산되어 있는지**를 측정합니다.
먼저 특잇값 분해(SVD)를 수행합니다.

$$
W=U\Sigma V^{\mathsf T},
\qquad
\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_q),
\qquad
\sigma_1\geq\cdots\geq\sigma_q\geq0,
\quad q=\min(m,n).
$$

축약 SVD에서 $U\in\mathbb{R}^{m\times q}$와 $V\in\mathbb{R}^{n\times q}$의 열은 직교 방향을 나타냅니다. 열벡터 $Wx$ 표기에서는 $V$가 입력 쪽 방향을, $U$가 출력 쪽 방향을 나타냅니다.
$\Sigma$는 대각선에 특잇값을 배치한 행렬이며, $\sigma_i$는 해당 방향의 크기 배율입니다. $\mathsf T$는 행과 열을 바꾸는 전치를 의미합니다.

$$
p_i=\frac{\sigma_i^2}{\sum_{j=1}^{q}\sigma_j^2},
\qquad
\sum_i p_i=1,
\qquad
\sum_i\sigma_i^2=\lVert W\rVert_F^2.
$$

$\lVert W\rVert_F^2$는 모든 행렬 원소를 제곱하여 더한 값입니다. 따라서 $p_i$는 전체 가중치 에너지 중 $i$번째 방향이 차지하는 비율입니다.
이 노트북에서는 다음 세 지표를 사용합니다.

| 지표 | 정의 | 해석 |
|---|---|---|
| Entropy effective rank | $\exp\!\left(-\sum_i p_i\log p_i\right)$ | 에너지 분포의 엔트로피를 차원 수로 환산합니다. $\log$는 자연로그이며 $0\log0=0$으로 정의합니다. |
| Participation ratio | $\displaystyle\frac{1}{\sum_i p_i^2}$ | 에너지가 일부 방향에 집중될수록 작아집니다. |
| 95% explained-variance dimension | $\displaystyle\min\left\{k:\sum_{i=1}^{k}p_i\geq0.95\right\}$ | 큰 특잇값부터 더해 전체 에너지의 95%를 설명하는 최소 차원 수를 구합니다. |

에너지가 $r$개 방향에 균등하게 분포하면 entropy rank와 participation ratio는 모두 $r$이 됩니다.
영행렬의 세 지표는 0으로 정의합니다. 여기서 explained variance는 **가중치의 특잇값 에너지**를 의미합니다.

![전체 768×768 Q·K·V·O의 유효 랭크](/assets/posts/gpt2-circuits-residual-stream/full-effective-rank.png)

### 3. 헤드별 Effective Rank
전체 행렬을 GPT-2의 64차원 헤드 단위로 나누어 분석합니다.

$$
W_Q^h,W_K^h,W_V^h\in\mathbb{R}^{64\times768},
\qquad W_O^h\in\mathbb{R}^{768\times64}.
$$

세 지표의 상한은 64입니다. 선은 레이어별 헤드 평균, 음영은 최솟값–최댓값을 나타냅니다. 색상은 위의 전체 Q·K·V·O 그래프와 동일하게 사용합니다.

![64차원 헤드별 Q·K·V·O 유효 랭크](/assets/posts/gpt2-circuits-residual-stream/head-effective-rank.png)

## QK, OV Circuits
원 논문의 표기에 따라 헤드 $h$의 두 회로를 정의합니다.

$$
W_{QK}^h=(W_Q^h)^{\mathsf T}W_K^h,
\qquad
W_{OV}^h=W_O^hW_V^h.
$$

QK는 **어디에 주목할지**, OV는 **어떤 정보를 residual에 더할지**를 결정하는 가중치 부분입니다.
목적지 토큰과 원본 토큰의 열벡터를 각각 $x_t,x_s\in\mathbb{R}^{768}$로 두면, 원 논문처럼 상수 스케일링을 생략한 표기는 다음과 같습니다.

$$
\mathrm{score}_{t,s}^h=x_t^{\mathsf T}W_{QK}^h x_s,
\qquad
\mathrm{output}_t^h=\sum_s\alpha_{t,s}^h W_{OV}^h x_s.
$$

$\mathrm{score}_{t,s}^h$는 두 토큰 사이의 점수인 스칼라입니다. $\alpha_{t,s}^h$는 인과 마스크와 softmax를 적용한 attention 가중치입니다.
열벡터에서는 오른쪽 행렬부터 적용하므로, **V 다음 O**의 연산 순서를 $W_OW_Vx$로 표현합니다. 회로 분석에서는 bias와 LayerNorm을 제외합니다.

**행렬곱·외적·텐서곱을 구분합니다.** $A\in\mathbb{R}^{m\times k}$와 $B\in\mathbb{R}^{k\times n}$의 행렬곱은 공유 차원 $k$를 합산합니다.

$$
(AB)_{ij}=\sum_{a=1}^{k}A_{ia}B_{aj},
\qquad AB\in\mathbb{R}^{m\times n}.
$$

열벡터 $u\in\mathbb{R}^{m}$와 $v\in\mathbb{R}^{n}$의 **외적**은 합산 없이 모든 성분 쌍을 곱하여 행렬을 만듭니다.

$$
(uv^{\mathsf T})_{ij}=u_i v_j.
$$

벡터의 텐서곱 $u\otimes v$는 이러한 성분 쌍을 갖는 2차 텐서이며, 행렬로 표현하면 외적에 대응합니다.
행렬의 텐서곱을 나타내는 Kronecker 곱 $A\otimes B$는 각 $A_{ij}$를 블록 $A_{ij}B$로 바꾸는 연산입니다.
원 논문의 $A^h\otimes W_{OV}^h$는 **토큰 사이의 이동**과 **벡터 성분의 변환**을 동시에 나타냅니다.

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

$T$는 토큰 수이고 $A_{ts}^h=\alpha_{t,s}^h$입니다. $A^h$는 $T\times T$ 행렬이므로 위 텐서곱의 크기는 $(768T)\times(768T)$입니다.
반면 $W_Q^{\mathsf T}W_K$와 $W_OW_V$ 자체는 **공유 차원을 합산하는 행렬곱**입니다. 각 곱은 64개의 외적을 더한 형태로 풀어 쓸 수 있습니다.

$$
W_{QK}^h=\sum_{a=1}^{64}q_a k_a^{\mathsf T},
\qquad
W_{OV}^h=\sum_{a=1}^{64}o_a v_a^{\mathsf T}.
$$

$q_a^{\mathsf T},k_a^{\mathsf T},v_a^{\mathsf T}$는 각 Q·K·V 행렬의 $a$번째 행이고, $o_a$는 O 행렬의 $a$번째 열입니다.
외적 하나의 랭크는 최대 1이므로, 두 회로는 $768\times768$ 행렬이지만 랭크가 최대 64입니다.

### 1. torch.linalg.matrix_rank
헤드별 QK·OV 행렬에 `torch.linalg.matrix_rank`를 적용하고, 각 레이어의 12개 헤드에 대한 평균과 최솟값·최댓값을 표시합니다.

실험에서는 모든 레이어와 헤드의 QK·OV 수치 랭크가 **64**였습니다.

### 2. Effective Rank
QK·OV에도 $p_i=\sigma_i^2/\sum_j\sigma_j^2$를 적용하여 세 유효 랭크 지표를 계산합니다.
그래프의 **선은 헤드 평균**, **음영은 헤드별 최솟값부터 최댓값까지의 범위**를 나타냅니다.

![QK·OV 회로의 유효 랭크](/assets/posts/gpt2-circuits-residual-stream/circuit-effective-rank.png)

유효 랭크가 작다는 사실만으로 해당 헤드가 중요하지 않다고 판단할 수는 없습니다. 가중치 에너지의 집중도와 실제 입력에서의 기능적 중요성은 서로 다른 측정 대상입니다.

## Copying Behavior: OV 고유분해
헤드별 OV 행렬을 $M=W_O^hW_V^h\in\mathbb{R}^{768\times768}$로 정의합니다.
열벡터 업데이트 $Mx$에서 방향이 어떻게 변하는지 확인하기 위해 **오른쪽 고유벡터**의 관계를 사용합니다.

$$
Mv=\lambda v,
\qquad v\ne0.
$$

$v$는 고유벡터, $\lambda$는 고윳값입니다. 입력 $v$가 $\lambda v$로 변환되는 관계를 나타냅니다.
OV는 일반적으로 비대칭이므로 `eig`를 사용하며 복소 고윳값도 유지합니다.

계산은 작은 행렬로 줄일 수 있습니다. $A=W_O^h\in\mathbb{R}^{768\times64}$,
$B=W_V^h\in\mathbb{R}^{64\times768}$로 두면 $M=AB$이며 다음 관계가 성립합니다.

$$
BAu=\lambda u
\quad\Longrightarrow\quad
AB(Au)=A(BAu)=\lambda(Au).
$$

따라서 0이 아닌 고윳값은 $64\times64$ 행렬 $BA$에서 계산합니다. 고유벡터가 필요하면 $BA$의 고유벡터 $u$로부터 $v=Au$를 계산하여 768차원으로 복원할 수 있습니다.
원래 행렬에 존재하는 최소 704개의 0 고윳값은 그림에서 제외합니다.

- $\lambda>0$인 **실수 고윳값**은 해당 방향의 부호를 보존합니다. $0<\lambda<1$이면 감쇠하고, $\lambda>1$이면 증폭합니다.
- $\lambda<0$인 실수 고윳값은 해당 방향의 부호를 반전합니다.
- 복소 고윳값은 실수부가 양수여도 양의 실수 고윳값으로 세지 않습니다.

첫 그림은 헤드별 **0이 아닌 고윳값 중 양의 실수 고윳값의 비율**을 표시합니다.
두 번째 그림은 [원 논문의 원형 산점도](https://transformer-circuits.pub/2021/framework/index.html#copying-matrix) 형식으로 모든 레이어와 헤드의 고윳값을 표시합니다.
**행은 레이어, 열은 헤드**입니다. 각 점은 0이 아닌 고윳값 하나이며 다음 좌표를 사용합니다.

$$
\theta=\arg(\lambda),\qquad \rho=\log_{10}|\lambda|+C.
$$

$\arg(\lambda)$는 복소평면에서의 각도이며, 양의 실수는 오른쪽, 음의 실수는 왼쪽에 놓입니다.
$C$는 반지름을 양수로 옮기는 공통 상수입니다. 동심원의 눈금은 원래 크기 $|\lambda|$를 나타냅니다.
모든 칸에 동일한 로그 척도를 적용하며, 주황색은 양의 실수 고윳값, 파란색은 나머지 고윳값을 표시합니다.
그림 형식은 논문을 따르며 분석 행렬은 $M=W_OW_V$를 사용합니다.

양의 고윳값은 residual 특징의 보존 성향을 나타냅니다. 특정 토큰의 복사를 확인하려면 $W_U M W_E$와 실제 attention·logit 효과를 함께 분석해야 합니다.
$W_E$는 토큰을 residual 공간으로 보내는 임베딩 행렬이고, $W_U$는 residual을 토큰별 logit으로 보내는 출력 행렬입니다.

![헤드별 양의 실수 고윳값 비율](/assets/posts/gpt2-circuits-residual-stream/ov-positive-eigenvalues.png)

![전체 12개 레이어와 12개 헤드의 OV 고윳값](/assets/posts/gpt2-circuits-residual-stream/ov-eigenspectra.svg)

## 결론 및 요약

**1. Residual stream은 임베딩에서 누적된 변화를 나타냅니다.**

$$
r_t^{(0)}=W_Ee_t+p_t,
\qquad
\Delta r_{\ell,t}=r_{\ell,t}^{\mathrm{post}}-r_t^{(0)}.
$$

Prefill은 모든 토큰을 처리하고, KV cache를 사용하는 decode는 새 토큰만 처리합니다. 그림의 이전 토큰 행은 표시용으로 0 처리합니다.

**2. 수치 랭크와 유효 랭크는 서로 다른 정보를 나타냅니다.**

$$
p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2},
\qquad
r_{\mathrm{entropy}}=\exp\!\left(-\sum_i p_i\log p_i\right),
\qquad
r_{\mathrm{PR}}=\frac{1}{\sum_i p_i^2},
\qquad
 d_{95}=\min\left\{k:\sum_{i=1}^{k}p_i\geq0.95\right\}.
$$

이번 GPT-2 실험에서 전체 Q·K·V·O의 수치 랭크는 모두 768입니다. 유효 랭크는 그보다 작으며, 에너지가 일부 방향에 집중되어 있음을 보여줍니다.

**3. QK·OV는 저랭크(low-rank) 행렬입니다.**

$$
W_{QK}^h=(W_Q^h)^{\mathsf T}W_K^h,
\qquad
W_{OV}^h=W_O^hW_V^h,
\qquad
\operatorname{rank}(W_{QK}^h),\operatorname{rank}(W_{OV}^h)\leq64\ll768.
$$

헤드별 QK·OV는 $768\times768$ 행렬이지만, 64차원 헤드를 거치는 행렬곱이므로 랭크가 최대 64입니다. 이번 실험에서도 모든 헤드의 수치 랭크는 64로 확인됩니다.

**4. 양의 OV 고윳값은 특징 보존의 단서입니다.**

$$
W_{OV}^h v=\lambda v,
\qquad
\lambda>0\ \Rightarrow\ \text{해당 실수 고유방향의 부호 보존}.
$$

OV에는 음의 실수와 복소 고윳값도 존재합니다. 토큰 복사는 $W_UW_{OV}^hW_E$와 실제 attention·logit 효과를 함께 확인해야 합니다.

## 후속 질문

1. Residual stream에 더해지는 변위는 레이어별로 어떤 차이가 있습니까?

   a. 차이가 있다면 그 이유는 무엇입니까?

2. 레이어별 역할이 구분되어 있습니까?

3. Q·K·V·O 이외의 행렬들의 유효 랭크는 어떻습니까?

4. 레이어별 변환과 유효 랭크에는 모델 간 공통점이 있습니까? 다음 범위에서 비교합니다.

   a. 같은 GPT-2 계열의 모델

   b. 같은 MHA 구조를 사용하는 모델

   c. 같은 트랜스포머 계열의 모델

   d. 같은 데이터셋으로 학습한 모델


## 참고 자료

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [실험 저장소: Mechanistic Interpretability Study](https://github.com/era-s/Mechanistic-Interpretability-study)
- [GPT-2 모델](https://huggingface.co/openai-community/gpt2)

그림은 실험 노트북에 저장된 실행 결과를 사용했습니다. 랭크 분석은 가중치에 대한 분석이며, 특정 기능의 인과적 기여를 확인하려면 추가적인 ablation이나 activation patching이 필요합니다.
