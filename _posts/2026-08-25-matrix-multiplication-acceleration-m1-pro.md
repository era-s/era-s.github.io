---
title: "Apple M1 Pro 행렬 곱셈 최적화: 3 GFLOPS에서 47.6 GFLOPS까지"
categories: [mlsys]
tags:
  - Apple Silicon
  - Matrix Multiplication
  - NEON
  - SIMD
  - Cache Blocking
  - Performance Engineering
mathjax: true
lightbox: true
permalink: /mlsys/matrix-multiplication-acceleration-m1-pro/
---

행렬 곱셈은 코드로 쓰면 세 줄짜리 중첩 루프지만, 하드웨어의 최고 성능에
가까워지려면 레지스터와 캐시, 명령어 발행 구조를 모두 고려해야 한다. 이
글에서는 Apple M1 Pro의 단일 P-core에서 FP64 행렬 곱셈을 직접 최적화한
과정을 정리한다. 단순 구현은 3.0037 GFLOPS에 그쳤지만, 10x4 NEON
마이크로커널과 패널화를 적용한 최종 구현은 **47.5751 GFLOPS**를 기록했다.

<!--more-->

코드와 원본 벤치마크 결과는
[Matrix-Multiplication-Acceleration-M1-Pro](https://github.com/era-s/Matrix-Multiplication-Acceleration-M1-Pro)에
정리했다.

## 결과부터 보기

실험은 6개의 P-core와 2개의 E-core가 탑재된 Apple M1 Pro에서 진행했다.
정방 행렬을 다섯 번씩 곱하고, $2N^3 / T$로 GFLOPS를 계산했다.

| 구현 | 최고 성능 | 단일 P-core 이론치 대비 |
| --- | ---: | ---: |
| 직관적인 `i-j-k` 3중 루프 | 3.0037 GFLOPS | 5.82% |
| OpenBLAS, 1 thread | 49.2012 GFLOPS | 95.35% |
| 10x4 NEON + 패널화 | **47.5751 GFLOPS** | **91.20%** |

최종 구현은 같은 조건에서 측정한 OpenBLAS 최고 성능의 **96.6950%**에
도달했다. 이 숫자는 행렬 곱셈 알고리즘 자체를 바꾼 결과가 아니다. 데이터
배치와 연산 순서를 바꿔 이미 존재하는 연산기를 쉬지 않게 만든 결과다.

## 먼저 하드웨어의 상한을 계산한다

M1 Pro의 Firestorm P-core는 128-bit NEON 레지스터 32개와 FP/SIMD
파이프라인 4개를 가진다. 128-bit 벡터 하나에는 double 두 개가 들어가고,
FMA는 곱셈과 덧셈을 합쳐 원소당 2 FLOP으로 센다. 따라서 최대 처리량은
다음과 같다.

$$
4\ \text{pipes}
\times 2\ \text{FP64 lanes}
\times 2\ \text{FLOP/FMA}
=16\ \text{FLOP/cycle}
$$

단일 P-core가 3.228 GHz로 동작하면 이론치는 51.648 GFLOPS다. 여섯 P-core가
동시에 동작할 때 관측한 약 3.036 GHz를 적용하면 전체 이론치는 291.456
GFLOPS가 된다.

| 항목 | 실험에 사용한 M1 Pro |
| --- | ---: |
| 코어 | P-core 6개, E-core 2개 |
| P-core 클럭 | 단일 최대 3.228 GHz, 전체 동작 시 약 3.036 GHz |
| SIMD | ARMv8.4-A Advanced SIMD, 128-bit NEON |
| 벡터 레지스터 | 32개 (`V0`-`V31`) |
| P-core L1 데이터 캐시 | 128 KiB |
| P-cluster L2 | 12 MiB |

초기 보고서에는 P-core L2를 24 MB로 적었지만, 해당 기기에서 macOS
`sysctl hw.perflevel0.l2cachesize`로 다시 확인한 값은 12 MiB였다. 24 MB는
시스템 레벨 캐시(SLC)와 혼동한 표기였으므로 여기서는 바로잡는다.

이 계산은 목표선을 정해준다. 측정값이 3 GFLOPS라면 단순히 "C++가
느리다"고 말할 것이 아니라, 51.6 GFLOPS 중 왜 94%가 비어 있는지 찾아야
한다.

## 기준점: OpenBLAS와 직관적인 구현

$A \in \mathbb{R}^{N \times P}$,
$B \in \mathbb{R}^{P \times M}$,
$C \in \mathbb{R}^{N \times M}$이고 $C=AB$라 하면 필요한 연산량은
다음과 같다.

$$
\text{FLOP}=2NMP
$$

행렬은 모두 row-major 1차원 배열로 저장했다. 이때 leading dimension은
열의 수이고, 원소 접근은 `A[i * lda + j]`가 된다.

가장 직관적인 구현은 다음과 같다.

```cpp
for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
        for (int k = 0; k < P; ++k) {
            C[i * M + j] += A[i * P + k] * B[k * M + j];
        }
    }
}
```

![직관적인 3중 루프의 단일 코어 성능](/assets/images/posts/matrix-multiplication-m1-pro/naive-single-core.png)

*작은 행렬에서 순간적으로 3.0037 GFLOPS를 기록했지만, 크기가 커지면
1.5 GFLOPS 부근까지 내려간다.*

이 구현은 최고 성능을 기준으로 해도 이론치의 5.82%밖에 사용하지 못했다.
반면 OpenBLAS는 행렬 크기 약 200부터 빠르게 45 GFLOPS를 넘었고, 최고
49.2012 GFLOPS를 기록했다.

![OpenBLAS 단일 코어 성능](/assets/images/posts/matrix-multiplication-m1-pro/openblas-single-core.png)

*OpenBLAS는 연산량이 충분해진 뒤 47-49 GFLOPS를 안정적으로 유지한다.*

## 컴파일 옵션만으로 해결되지 않은 이유

먼저 컴파일된 코드를 디스어셈블했다. `-O2`에서 가장 안쪽 루프는 개략적으로
다음 순서였다.

```asm
ldr   d29, [x8]
ldr   d30, [x6], 8
fmadd d31, d30, d29, d31
str   d31, [x10, x9, lsl 3]
cmp   x11, x6
bne   .L8
```

FMA 자체는 생성됐지만 매 반복에서 scalar 두 개를 읽고, 하나의 누산 체인에
의존하고, 결과를 다시 저장한다. 네 개의 FP/SIMD 파이프라인과 두 개의 FP64
lane을 동시에 채우지 못한다.

`__restrict`, `-O3`, `-funroll-loops`, `-ftree-vectorize`도 적용해 보았다.
분기 횟수는 줄고 `ldp`가 생성됐지만, 연속된 FMA가 여전히 하나의 의존성
사슬을 만들었다. 컴파일러에게 더 강한 힌트를 주는 것만으로는 데이터 재사용
구조가 바뀌지 않았다.

## 핵심은 산술 강도와 재사용이다

Roofline 관점에서 실행 시간은 연산 시간과 데이터 이동 시간 중 더 긴 쪽에
의해 제한된다.

$$
T_{math}=\frac{\text{FLOP}}{\text{peak FLOPS}},\qquad
T_{comm}=\frac{\text{bytes moved}}{\text{bandwidth}}
$$

산술 강도는 이동한 데이터 한 바이트당 수행한 연산량이다.

$$
\text{Arithmetic Intensity}=\frac{\text{FLOP}}{\text{bytes moved}}
$$

직관적인 내부 루프는 C 한 원소를 만들 때 A와 B를 계속 새로 읽는다. 반면
작은 C 타일을 레지스터에 유지한 채 A의 한 열과 B의 한 행으로 외적을
누적하면, 같은 로드로 여러 C 원소를 갱신할 수 있다.

### 10x4 마이크로커널

최종 커널은 한 번에 C의 10행 x 4열, 총 40개 double을 처리한다. NEON
레지스터 하나가 double 두 개를 담으므로 C 누산에 레지스터 20개를 사용한다.
A의 10개 값을 broadcast하는 데 10개, B의 4개 값을 읽는 데 2개를 사용하면
정확히 32개다.

```cpp
for (int k = 0; k < kc; ++k) {
    float64x2_t a0 = vld1q_dup_f64(A + k * 10 + 0);
    // ... a1부터 a9까지 broadcast

    float64x2_t b0 = vld1q_f64(B + k * 4 + 0);
    float64x2_t b1 = vld1q_f64(B + k * 4 + 2);

    c00 = vfmaq_f64(c00, a0, b0);
    c01 = vfmaq_f64(c01, a0, b1);
    // ... 10x4 C 타일 전체에 rank-1 update
}
```

여러 개의 독립적인 누산 체인을 만들었기 때문에 out-of-order 실행기가 FMA를
네 파이프라인에 분산할 여지가 생긴다. C는 `k` 루프가 끝날 때까지 레지스터에
남아 있으므로 반복마다 메모리에 저장하지 않는다.

## L1, L2, SLC에 맞춘 패널화

마이크로커널만으로는 큰 행렬의 캐시 미스를 막을 수 없다. A와 B를 커널이
읽는 순서에 맞춰 연속적인 패널로 복사하고, 각 패널이 목표 캐시 범위를 넘지
않도록 블로킹했다.

참고한 메모리 대역폭 측정에서는 region size 약 4 MB를 전후로 대역폭이
크게 하락했다. 이 값은 캐시 용량 자체와 동일시하기보다 패널 크기를 정하기
위한 경험적인 경계로 사용했다.

![M1 계열 region size별 메모리 대역폭](/assets/images/posts/matrix-multiplication-m1-pro/cache-bandwidth.png)

최종 블로킹 파라미터는 다음과 같다.

| 계층 | 파라미터 | 작업 집합 |
| --- | --- | ---: |
| 마이크로커널 | `NR=10`, `MR=4` | C 10x4 타일 |
| K 방향 블록 | `PC=640` | A/B 커널 입력 약 70 KiB |
| A 패널 | `NC=520`, `PC=640` | 2.66 MB, 약 2.54 MiB |
| B 패널 | `MC=768`, `PC=640` | 3.93 MB, 약 3.75 MiB |

루프는 `j-k-i` 순서로 구성했다.

```text
for each column block j:
    for each depth block k:
        pack B(k, j)
        for each row block i:
            pack A(i, k)
            run every 10x4 micro-kernel in C(i, j)
```

B 패널을 `k` 루프에서 한 번 패킹하고 여러 A 행 블록에서 재사용한다. A는
각 `i` 블록에 들어갈 때 패킹한다. row-major 저장 방식 때문에 처음에는
`i-k-j`가 더 유리할 것으로 예상했지만, 실제 측정에서는 `j-k-i`가 근소하게
앞섰다.

![j-k-i 순서의 최종 단일 코어 성능](/assets/images/posts/matrix-multiplication-m1-pro/neon-single-core-jki.png)

*`j-k-i`, 10x4 커널은 최고 47.5751 GFLOPS를 기록했다.*

![i-k-j 순서의 단일 코어 성능](/assets/images/posts/matrix-multiplication-m1-pro/neon-single-core-ikj.png)

*`i-k-j` 변형은 최고 47.1018 GFLOPS로 아주 조금 낮았다.*

큰 행렬에서는 46-47 GFLOPS가 유지됐다. 직접 작성한 C++ intrinsic 커널이
어셈블리로 세밀하게 튜닝된 OpenBLAS에 근접했다는 점이 가장 의미 있는
결과였다.

## 멀티코어는 왜 미완성으로 남았나

OpenBLAS는 여섯 P-core에서 최고 278.1190 GFLOPS를 기록해 이론치의
95.4240%에 도달했다. E-core까지 함께 쓰면 스케줄링과 패널 분배가 복잡해지는
반면 성능 향상은 뚜렷하지 않아 P-core만 비교했다.

![OpenBLAS 6 P-core 성능](/assets/images/posts/matrix-multiplication-m1-pro/openblas-multicore.png)

*OpenBLAS는 큰 행렬에서 260-278 GFLOPS 구간을 비교적 안정적으로 유지했다.*

직접 작성한 OpenMP 구현은 `i-j-k` 블록 순서와 `omp for collapse(2)`에서
최고 243.4346 GFLOPS, 이론치의 83.5236%를 기록했다.

![OpenMP 멀티코어 실험](/assets/images/posts/matrix-multiplication-m1-pro/neon-multicore.png)

그러나 그래프가 톱니 모양으로 크게 흔들리고, 단일 코어처럼 안정적인
compute-bound 구간을 만들지 못했다. 최고점 하나만 보면 빠르지만, 패널을
코어별로 어떻게 소유하고 공유 캐시와 메모리 대역폭을 어떻게 나눌지 충분히
해결하지 못했다. 그래서 이 결과는 완성된 멀티코어 구현이 아니라 후속
분석이 필요한 실험으로 분류했다.

## 실험에서 얻은 교훈

첫째, 성능 코드는 소스만 보고 판단할 수 없다. 단순 구현에서도 FMA가
생성됐지만, 디스어셈블을 보기 전에는 하나의 의존성 사슬과 scalar load가
파이프라인을 막고 있다는 사실을 분명히 알기 어려웠다.

둘째, SIMD intrinsic만 추가한다고 빨라지지 않는다. 레지스터 타일링과 패널
패킹으로 데이터 재사용 구조를 먼저 만들었기 때문에 NEON FMA가 의미를
가졌다.

셋째, 프로파일러가 없으면 탐색 비용이 급격히 커진다. 이 실험에서는
GFLOPS 곡선만 보고 커널 크기와 패널 크기를 바꾸느라 많은 시행착오를
겪었다. 성능 카운터로 cache miss, stall, pipeline utilization을 확인할 수
있었다면 멀티코어 병목까지 더 빠르게 좁힐 수 있었을 것이다.

## 코드 실행하기

저장소의 현재 구조에서는 최종 구현이 `src`와 `include`에 있고, 과거 커널
탐색 코드는 `legacy`에 보존돼 있다.

```bash
git clone https://github.com/era-s/Matrix-Multiplication-Acceleration-M1-Pro.git
cd Matrix-Multiplication-Acceleration-M1-Pro

# 정확성 검증
make test

# 기본 벤치마크
make benchmark
```

원 보고서와 같은 GCC 계열을 쓰려면 `make CXX=g++-15`로 컴파일러를 지정할
수 있다. 전체 크기 범위를 다시 측정하려면 다음처럼 실행한다.

```bash
./build/benchmark \
  --min 20 \
  --max 4080 \
  --step 20 \
  --trials 5 \
  --output results/benchmark-latest.csv
```

결과는 온도와 전원 상태, 백그라운드 작업, 코어 배치에 영향을 받는다.
재현할 때는 같은 조건에서 여러 번 측정하고 평균과 분산을 함께 보는 편이
좋다.

## 참고 자료

- [DGEMM Tutorial](https://github.com/nakatamaho/dgemm_tutorial/blob/main/01_introduction.md)
- [The JAX Scaling Book - All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)
- [Dougall Johnson - Apple Firestorm](https://dougallj.github.io/applecpu/firestorm.html)
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)
- [M1 cache and memory bandwidth data](https://jsmemtest.chipsandcheese.com/bwdata)
- [Compiler Explorer](https://godbolt.org/)
