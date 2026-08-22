---
title: "캐시부터 NUMA까지: 메모리 지연 시간 직접 측정하기"
categories: [architecture]
tags:
  - CPU
  - Memory Hierarchy
  - Cache
  - NUMA
  - Prefetcher
  - Microbenchmark
lightbox: true
---

메모리 지연 시간은 흔히 하나의 숫자로 표현되지만, 실제 측정값은 데이터가
머무는 계층과 접근 패턴, NUMA 배치, 하드웨어 프리페처의 동작에 따라 크게
달라진다. 이번 글에서는 두 가지 CPU에서 포인터 체이싱 기반 마이크로벤치마크를
실행하고, 작업 집합의 크기와 stride를 바꾸며 메모리 계층의 경계를 확인한다.

<!--more-->

## 실험 환경

측정에는 [Google multichase](https://github.com/google/multichase)를 사용했다.
`multichase`는 배열 안의 포인터를 따라가며 접근 지연 시간을 측정하는
마이크로벤치마크다. 별도 옵션이 없으면 포인터를 랜덤 순서로 연결하며,
`-o`를 사용하면 ordered traversal로 측정할 수 있다.

먼저 `lscpu`로 CPU family와 model, 캐시 인스턴스, NUMA 구성을 확인했다.
두 시스템의 사양은 다음과 같다.

| 항목 | Intel Xeon E-2388G | AMD EPYC 9354 |
| --- | ---: | ---: |
| CPU family / model | 6 / 167 | 25 / 17 |
| 코어 / 스레드 | 8 / 16 | 32 / 64 |
| L1 데이터 캐시 | 코어당 48 KiB, 총 384 KiB | 코어당 32 KiB, 총 1 MiB |
| L2 캐시 | 코어당 512 KiB, 총 4 MiB | 코어당 1 MiB, 총 32 MiB |
| LLC | 프로세서당 16 MiB | CCD당 32 MiB, 총 256 MiB |
| NUMA 구성 | 1개 노드 | NPS1 또는 NPS4 |
| 캐시라인 | 64 B | 64 B |

AMD 시스템은 처음에는 NPS1이었지만 local/remote 지연 시간을 비교하기 위해
BIOS에서 NPS4로 변경했다. 이때 논리 CPU 매핑은 다음과 같았다.

```text
# NPS1
node0: 0-63

# NPS4
node0: 0-7,32-39
node1: 8-15,40-47
node2: 16-23,48-55
node3: 24-31,56-63
```

`dmidecode -t memory`, `dmidecode -t 16`, `dmidecode -t 17`로 메모리
구성을 확인했다. AMD 시스템에는 `DIMMA1`, `DIMMC1`, `DIMMG1`, `DIMMI1`에
각각 64 GiB DIMM이 설치되어 있었다. 두 시스템 모두 아래 경로에서 64 B
캐시라인을 확인했다.

```bash
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
# 64
```

두 시스템의 세대, 코어 수, 클럭, 메모리 구성이 다르기 때문에 이 결과를
CPU 간 우열을 가리는 벤치마크로 해석해서는 안 된다. 관심사는 각 시스템
안에서 작업 집합이 커질 때 나타나는 변곡점과 접근 패턴의 영향이다.

## 실험 목표

실험 목표는 두 가지다.

1. L1, L2, LLC, DRAM의 지연 시간을 측정하고, AMD 시스템에서는 NPS4의
   local 및 remote 메모리까지 비교한다.
2. 하드웨어 프리페처를 활성화한 상태에서 stride를 바꿔가며 각 메모리
   계층의 랜덤 접근과 순차 접근 지연 시간이 어떻게 변하는지 확인한다.

## 실험 방법

### 1. 프리페처 비활성화와 메모리 계층 스윕

메모리 계층 자체의 지연 시간을 확인할 때는 하드웨어 프리페처를
비활성화했다. `msr-tools`를 설치하고 MSR 모듈을 올린 뒤 모든 논리 CPU에
설정을 적용했다.

> 아래 MSR 주소와 값은 이 실험에 사용한 CPU 세대에 종속된다. 지원 여부를
> 확인하지 않고 다른 프로세서에 쓰면 시스템이 멈추거나 데이터가 손상될 수
> 있다. 재현할 때는 해당 CPU의 공식 문서를 먼저 확인해야 한다.

```bash
sudo apt-get install -y msr-tools
sudo modprobe msr

# Intel Rocket Lake 계열: 프리페처 비활성화
for cpu in $(seq 0 $(($(nproc) - 1))); do
  sudo wrmsr -p "$cpu" 0x1a4 0xf
done
sudo rdmsr -p 0 0x1a4

# AMD Genoa 계열: 프리페처 비활성화
for cpu in $(seq 0 $(($(nproc) - 1))); do
  sudo wrmsr -p "$cpu" 0xC0000108 0x2F
done
sudo rdmsr -p 0 0xC0000108
```

전체 계층 스윕에서는 작업 집합을 다음 17개 크기로 늘렸다.

```text
16k 32k 64k 128k 256k 512k 1m 2m 4m 8m 16m 32m 64m 128m 256m 512m 1g
```

기본 stride는 64 B, 내부 샘플 수는 10, 스레드는 1개로 고정했다. 각 크기를
다섯 번 측정하고 최솟값과 최댓값을 제외한 나머지 세 값의 평균을 사용했다.
각 작업 집합 측정 뒤에는 1초의 간격을 뒀다. 랜덤 접근은 기본 실행으로,
순차 접근은 `-o`를 추가해 같은 조건으로 측정했다.

```bash
SIZES=(16k 32k 64k 128k 256k 512k 1m 2m 4m 8m 16m 32m 64m 128m 256m 512m 1g)
STRIDE=64
SAMPLES=10
THREADS=1
ACCESS_ARGS=()          # 랜덤 접근
# ACCESS_ARGS=(-o)      # 순차 접근
OUT="multichase_prefetchOFF.csv"

echo "size,repeat,ns" > "$OUT"
for SZ in "${SIZES[@]}"; do
  for i in {1..5}; do
    LINE=$(./multichase "${ACCESS_ARGS[@]}" -m "$SZ" -s "$STRIDE" \
      -n "$SAMPLES" -t "$THREADS" | tail -n 1)
    echo "$SZ,$i,$LINE" >> "$OUT"
  done
  sleep 1
done
```

전체 스윕과 별도로 명목 캐시 용량 주변을 더 촘촘하게 측정했다.

| 시스템 | 경계 | 측정한 작업 집합 |
| --- | --- | --- |
| Intel | L1 / L2 | `40k, 42k, 44k, 46k, 48k, 50k, 52k, 54k, 56k` |
| Intel | L2 / LLC | `472k, 482k, 492k, 502k, 512k, 522k, 532k, 542k, 552k` |
| Intel | LLC / DRAM | `14m, 14500k, 15m, 15500k, 16m, 16500k, 17m, 17500k, 18m` |
| AMD | L1 / L2 | `26k, 28k, 30k, 31k, 32k, 33k, 34k, 35k, 36k` |
| AMD | L2 / LLC | `984k, 994k, 1004k, 1014k, 1024k, 1034k, 1044k, 1054k, 1064k` |
| AMD | LLC / DRAM | `30000k, 30500k, 31000k, 31500k, 32m, 32500k, 33000k, 33500k, 34000k` |

AMD NPS4의 local/remote 실험에서는 node0을 local로 두고 나머지 세 노드의
결과를 remote1~3으로 기록했다. 작업 집합은 `8m, 16m, 24m, 32m, 48m,
64m, 96m, 128m, 256m`을 사용했다. 원본 실험 기록에는 CPU 및 메모리
바인딩의 정확한 실행 명령이 남아 있지 않으므로, 여기서는 사용하지 않은
명령을 임의로 보충하지 않는다.

### 2. 프리페처 활성화와 stride 스윕

두 번째 실험에서는 위 MSR을 각 플랫폼에서 `0x0`으로 되돌려 프리페처를
활성화했다. 각 계층 내부에 들어가는 대표 작업 집합을 하나씩 선택하고
stride를 바꿨다.

| 시스템 | L1 | L2 | LLC | DRAM |
| --- | ---: | ---: | ---: | ---: |
| Intel | 32 KiB | 384 KiB | 12 MiB | 768 MiB |
| AMD | 24 KiB | 768 KiB | 24 MiB | 768 MiB |

stride는 `8, 16, 32, 64, 128, 256, 512, 1024` B를 사용했다. 각 조합을
다섯 번 반복하고 양 끝값을 제외한 세 값의 평균을 사용했으며, stride 조합
사이에는 1초의 간격을 뒀다.

```bash
SIZES=(32k 384k 12m 768m)  # Intel
# SIZES=(24k 768k 24m 768m)  # AMD
STRIDES=(8 16 32 64 128 256 512 1024)
SAMPLES=10
THREADS=1
ACCESS_ARGS=()             # 랜덤 접근
# ACCESS_ARGS=(-o)         # 순차 접근
OUT="multichase_prefetchON_stride.csv"

echo "size,stride,repeat,ns" > "$OUT"
for SZ in "${SIZES[@]}"; do
  for ST in "${STRIDES[@]}"; do
    for i in {1..5}; do
      LINE=$(./multichase "${ACCESS_ARGS[@]}" -m "$SZ" -s "$ST" \
        -n "$SAMPLES" -t "$THREADS" | tail -n 1)
      echo "$SZ,$ST,$i,$LINE" >> "$OUT"
    done
    sleep 1
  done
done
```

### 3. CCD 간 캐시라인 이동

부록 실험으로 `multichase`의 `pingpong`을 사용했다. 두 코어가 수정된 64 B
캐시라인의 소유권을 번갈아 가져가도록 하고, 같은 CCD 안의 코어 조합과 서로
다른 CCD의 코어 조합을 비교했다. Intra-CCD에는 `0-1, 0-2, 0-3, 0-32,
0-33, 0-34, 0-35`를, Inter-CCD에는 `0-4, 0-8, 0-12, 0-16, 0-20,
0-24, 0-28`을 사용했다.

## 작업 집합 크기로 메모리 계층 찾기

프리페처를 끄고 버퍼 크기를 늘리면 두 시스템 모두 계단 모양의 곡선을
보였다. 작업 집합이 상위 캐시에 더는 들어가지 않는 순간, 다음 계층의
지연 시간이 측정값에 반영되기 시작한다. 전체 범위를 한눈에 보기 위해
가로축은 로그 스케일로 표시했다.

### Intel Xeon E-2388G

<div class="memory-chart-grid" role="group" aria-label="Intel Xeon E-2388G 메모리 계층 측정 결과">
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-xeon-overview.png" alt="Intel 시스템의 버퍼 크기별 전체 메모리 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-xeon-l1-l2.png" alt="Intel 시스템의 L1과 L2 경계 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-xeon-l2-llc.png" alt="Intel 시스템의 L2와 LLC 경계 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-xeon-llc-dram.png" alt="Intel 시스템의 LLC와 DRAM 경계 지연 시간"></figure>
</div>

*Intel 시스템에서는 약 1 ns였던 L1 구간이 L2, LLC, DRAM으로 내려가며
단계적으로 증가한다.*

### AMD EPYC 9354

<div class="memory-chart-grid" role="group" aria-label="AMD EPYC 9354 메모리 계층 측정 결과">
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-epyc-overview.png" alt="AMD 시스템의 버퍼 크기별 전체 메모리 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-epyc-l1-l2.png" alt="AMD 시스템의 L1과 L2 경계 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-epyc-l2-llc.png" alt="AMD 시스템의 L2와 LLC 경계 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/hierarchy-epyc-llc-dram.png" alt="AMD 시스템의 LLC와 DRAM 경계 지연 시간"></figure>
</div>

*AMD 시스템에서도 캐시 용량 경계 부근에서 같은 형태의 변곡점이 나타난다.*

각 계층 안쪽의 대표 작업 집합에서 얻은 값은 다음과 같다.

| 시스템 | 접근 | L1 | L2 | LLC | DRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| Intel | 랜덤 | 1.089 ns | 2.830 ns | 11.363 ns | 75.211 ns |
| Intel | 순차 | 1.093 ns | 2.809 ns | 11.170 ns | 70.720 ns |
| AMD | 랜덤 | 1.056 ns | 3.728 ns | 14.265 ns | 110.800 ns |
| AMD | 순차 | 1.056 ns | 3.713 ns | 14.088 ns | 100.100 ns |

이 표의 값은 각 계층 안쪽에 들어가는 대표 작업 집합에서 얻었다. Intel은
L1 `32 KiB`, L2 `128 KiB`, LLC `4 MiB`, DRAM `512 MiB`를 사용했고,
AMD는 각각 `16 KiB`, `256 KiB`, `4 MiB`, `512 MiB`를 사용했다. 서로
다른 플랫폼의 숫자를 직접 비교하기보다, 한 행 안에서 계층이 내려갈수록
비용이 어떻게 달라지는지를 보는 편이 적절하다.

경계 전후를 더 촘촘하게 측정하면 변화가 분명해진다.

| 시스템 | 경계 | 경계 전 | 경계 후 |
| --- | --- | ---: | ---: |
| Intel | L1 → L2 | 32 KiB: 1.089 ns | 64 KiB: 2.825 ns |
| Intel | L2 → LLC | 512 KiB: 5.989 ns | 1 MiB: 11.063 ns |
| Intel | LLC → DRAM | 16 MiB: 21.371 ns | 32 MiB: 50.509 ns |
| AMD | L1 → L2 | 32 KiB: 1.082 ns | 64 KiB: 3.710 ns |
| AMD | L2 → LLC | 1 MiB: 7.282 ns | 2 MiB: 13.365 ns |
| AMD | LLC → DRAM | 32 MiB: 50.485 ns | 64 MiB: 110.600 ns |

실제 곡선은 캐시의 명목 용량에서 완벽한 수직 절벽을 만들지 않는다. 캐시
연관도와 교체 정책, 공유 LLC의 slice 배치, 벤치마크 자체가 차지하는 공간
등이 함께 작용하기 때문이다. 따라서 이 측정은 캐시 크기를 정확히 역산하는
도구라기보다 계층 전환 구간을 관찰하는 방법에 가깝다.

## NUMA에서는 같은 DRAM도 거리가 다르다

AMD 시스템을 NPS4로 구성하고 local 메모리와 세 remote 메모리의 지연
시간을 비교했다. 작업 집합이 LLC 안에 들어가는 구간에서는 차이가 작지만,
LLC를 벗어나면 local과 remote 사이의 간격이 벌어진다.

<div class="memory-chart-grid" role="group" aria-label="AMD NUMA local 및 remote 메모리 지연 시간">
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/numa-random.png" alt="AMD NUMA local 및 remote 메모리의 랜덤 접근 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/numa-ordered.png" alt="AMD NUMA local 및 remote 메모리의 순차 접근 지연 시간"></figure>
</div>

256 MiB 작업 집합에서 랜덤 접근은 local 101.8 ns, 가장 느린 remote
112.667 ns였다. 순차 접근도 local 102 ns, 가장 느린 remote 113.367 ns로
약 11 ns의 차이를 보였다. 48 MiB 구간에서는 local 56.384 ns와 remote
76.046 ns 사이에 약 20 ns의 차이가 관측됐다.

remote 노드끼리도 완전히 같은 값을 보이지 않았다. 운영체제가 표시하는
NUMA distance가 같더라도 I/O die 안의 메모리 컨트롤러 위치와 실제 라우팅,
메모리 채널 배치가 동일하지 않을 수 있다. 다만 이번 측정만으로 원인을
하나로 확정할 수는 없으므로, 물리적 경로에 따른 차이일 가능성으로 남겨둔다.

이 결과가 주는 실용적인 메시지는 단순하다. 큰 배열을 반복해서 처리하는
코드에서는 스레드 pinning만큼 메모리의 local allocation도 중요하다.

## Stride가 프리페처의 예측을 바꾼다

stride는 연속된 두 접근 사이의 바이트 간격이다. 두 시스템의 캐시라인은
64 B이므로, stride가 커질수록 한 캐시라인에서 실제로 사용하는 데이터가
줄고 프리페처가 다음 주소를 예측하기도 어려워진다.

L1 안에서는 두 시스템 모두 stride에 따른 변화가 거의 없었다. 반면 L2부터
DRAM까지는 뚜렷한 전환점이 나타났다.

### Intel Xeon E-2388G

<div class="memory-chart-grid" role="group" aria-label="Intel Xeon E-2388G stride별 지연 시간">
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-xeon-l1.png" alt="Intel 시스템의 L1 stride별 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-xeon-l2.png" alt="Intel 시스템의 L2 stride별 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-xeon-llc.png" alt="Intel 시스템의 LLC stride별 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-xeon-dram.png" alt="Intel 시스템의 DRAM stride별 지연 시간"></figure>
</div>

Intel 시스템의 L2에서는 stride 8 B일 때 순차 접근 지연 시간이 랜덤 접근의
약 40% 수준이었다. 그러나 stride 256 B에서 순차 접근 지연 시간이 급증해
랜덤 접근보다 느려지는 반전이 나타났다. 512 B 이상에서 두 접근 방식의
지연 시간이 소폭 내려간 것은 cache set 충돌이나 TLB 효과일 가능성이 있지만,
이번 실험만으로 원인을 확정할 수는 없다.

Intel 시스템의 LLC에서는 순차 접근이 stride 128 B 이하일 때 약 1~3 ns로
유지됐지만, 256 B부터 12~14 ns 수준으로 급증했다. DRAM에서도 256 B 부근에
같은 변곡점이 나타났다. 다만 DRAM 구간에서는 순차 접근이 전체적으로 랜덤
접근보다 낮은 지연 시간을 유지했다.

### AMD EPYC 9354

<div class="memory-chart-grid" role="group" aria-label="AMD EPYC 9354 stride별 지연 시간">
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-epyc-l1.png" alt="AMD 시스템의 L1 stride별 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-epyc-l2.png" alt="AMD 시스템의 L2 stride별 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-epyc-llc.png" alt="AMD 시스템의 LLC stride별 지연 시간"></figure>
  <figure><img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/stride-epyc-dram.png" alt="AMD 시스템의 DRAM stride별 지연 시간"></figure>
</div>

AMD 시스템에서는 순차 접근의 전환점이 더 이르게 나타났다. stride 64 B
이하에서는 L2부터 DRAM까지 약 1 ns에 가까운 유효 접근 비용이 측정됐지만,
첫 측정점인 128 B에서 급격히 증가했다. DRAM에서는 순차 접근이 128 B에서
약 60 ns, 256 B 이상에서 약 105~110 ns 수준으로 올라갔다.

여기서 1 ns라는 값은 실제 DRAM의 단일 load-to-use 지연 시간이 아니다.
예측 가능한 스트리밍 접근에서는 프리페처가 다음 캐시라인을 미리 가져오고
여러 요청이 겹쳐 실행되므로, 벤치마크에 보이는 평균 유효 비용이 크게
낮아진 것이다.

<div class="memory-chart-grid" role="group" aria-label="프리페처 활성화 상태의 버퍼 크기별 지연 시간">
  <figure>
    <img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/prefetch-xeon.png" alt="프리페처를 활성화한 Intel 시스템의 버퍼 크기별 랜덤 및 순차 접근 지연 시간">
    <figcaption>Intel Xeon E-2388G</figcaption>
  </figure>
  <figure>
    <img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/prefetch-epyc.png" alt="프리페처를 활성화한 AMD 시스템의 버퍼 크기별 랜덤 및 순차 접근 지연 시간">
    <figcaption>AMD EPYC 9354</figcaption>
  </figure>
</div>

*두 시스템 모두 큰 작업 집합에서 순차 접근이 랜덤 접근보다 훨씬 낮다.
프리페처가 메모리 지연 시간의 상당 부분을 숨긴 결과다.*

1 GiB 순차 접근에서는 Intel이 3.777 ns, AMD가 1.6 ns로 측정됐다. 이 값은
AMD의 순차 접근 경로가 이번 패턴에서 더 적극적으로 지연을 숨겼음을 보여준다.
다만 서로 다른 시스템의 절대 성능 비교가 아니라, 프리페처를 활성화했을 때
각 시스템 안에서 랜덤·순차 접근이 얼마나 달라지는지에 초점을 맞춰야 한다.

따라서 “메모리 지연 시간”을 측정하려면 먼저 무엇을 재고 싶은지 구분해야
한다. 의존적인 랜덤 포인터 체이싱은 캐시 미스의 load-to-use 비용에 가깝고,
순차 접근은 프리페처와 메모리 수준 병렬성이 포함된 실제 스트리밍 성능에
가깝다.

## CCD를 건너는 캐시라인 소유권 이전

마지막으로 `multichase`의 `pingpong`을 사용해 수정된 64 B 캐시라인의
소유권을 두 코어가 번갈아 가져올 때의 평균 지연 시간을 측정했다.

<figure class="memory-chart-single">
  <img decoding="async" width="1000" height="600" src="/assets/images/posts/memory-latency-check/ccd-pingpong.png" alt="AMD EPYC 9354의 Intra-CCD 및 Inter-CCD 캐시라인 이동 지연 시간">
</figure>

- 같은 CCD 안의 코어 조합: 90.633~101.967 ns
- 서로 다른 CCD의 코어 조합: 약 173.7 ns

Inter-CCD 경로는 Intra-CCD보다 약 72~83 ns, 비율로는 약 1.7~1.9배
느렸다. 같은 CCD 안에서는 공유 LLC 영역 안에서 소유권이 이동하지만,
CCD를 건너면 I/O die와 패브릭을 거치는 추가 경로가 필요하기 때문이다.
이번 결과에서 Inter-CCD 조합별 값이 거의 같았던 점도 흥미롭다.

원본 기록의 설명에는 동일 물리 코어의 SMT 조합을 약 10 ns라고 적은 부분이
있지만, 같은 페이지의 그래프에는 `0-32` 조합이 90.633 ns로 표시되어 있다.
이 글에서는 서로 충돌하는 서술 대신 그래프의 수치를 사용했다. 원시 결과가
확보되면 이 조합은 다시 확인할 필요가 있다. Inter-CCD 값도 원본에는
`one-way`라고 표기되어 있지만, 여기서는 `pingpong`이 출력한 교환 지연
값으로만 해석한다.

## 측정에서 얻은 교훈

이번 실험에서 확인한 핵심은 다음과 같다.

1. 작업 집합이 캐시 용량을 넘으면 지연 시간 곡선에 계층 전환이 나타난다.
2. 공유 LLC의 경계는 연관도와 slice 배치 때문에 명목 용량과 정확히 일치하는
   절벽이 아니라 완만한 구간으로 보일 수 있다.
3. NUMA 시스템에서는 LLC를 벗어난 뒤 local과 remote 메모리의 차이가
   실질적인 비용으로 드러난다.
4. 작은 stride의 순차 접근은 하드웨어 프리페처가 지연 시간을 크게 숨긴다.
5. 같은 소켓이라도 CCD 경계를 넘는 캐시라인 이동에는 추가 비용이 든다.

처음에는 AMD 시스템의 LLC가 CCD 사이에서도 공유될 수 있다고 생각했지만,
실험을 통해 LLC는 CCD별로 분리되고 한 CCD 안에서만 공유된다는 점을 확인했다.
이 특성은 행렬 곱셈과 같은 메모리 집약적 연산에도 직접 연결된다. 패널 크기,
블록 크기, leading dimension을 정할 때 캐시 경계와 프리페처가 효과를 잃는
stride 구간을 함께 고려해야 한다.

절대값을 재현하거나 다른 시스템과 비교하려면 더 많은 환경 정보가 필요하다.
CPU governor와 turbo 상태, 코어 affinity, NUMA 메모리 바인딩, 운영체제와
커널, 컴파일러 옵션, 메모리 주파수와 타이밍, huge page 사용 여부, 워밍업
방법을 함께 고정해야 한다. 원시 CSV와 벤치마크 커밋 해시도 기록해 두는 편이
좋다.

메모리 지연 시간은 하나의 고정된 숫자가 아니다. 어떤 데이터를 어떤 순서로,
어느 코어에서, 어느 메모리에 접근하는지까지 포함해야 비로소 의미 있는
측정값이 된다.

## 참고

- [Google multichase](https://github.com/google/multichase)
