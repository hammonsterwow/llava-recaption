# LLaVA 558K Recaptioning 전달 사항과 프롬프트

## 목적

기존 이미지에 대응하는 캡션을 더 자세하고 정확하게 재생성한다. 생성된 설명이 실제 이미지와 잘 맞는지 실행 중간중간 직접 확인한다. 모델을 새로 학습하는 작업이 아니라, 이미 학습된 이미지·언어 모델로 새 캡션을 생성하는 작업이다.

## 카카오톡 전달 내용

### 데이터와 스크립트 위치

- 작업 폴더는 `/mnt3/jisung/recaption`이다.
- 해당 폴더의 LLaVA-558K 데이터와 스크립트를 사용해 실험을 실행한다.

### 모델과 메모리

- 실제 실행 모델은 이미지 입력을 처리하는 `Qwen/Qwen2.5-VL-7B-Instruct`이다.
- TITAN Xp 12 GB GPU에 맞추기 위해 4-bit NF4 quantization을 사용한다.

### GPU 사용과 작업 분배

- GPU 0, 1, 2, 3 네 대를 사용한다.
- 558K 데이터셋을 네 shard로 나누고 GPU 한 대당 하나의 독립 worker/model 인스턴스가 서로 다른 이미지를 처리한다.

### 실행과 원격 연결

- 장시간 실행은 `tmux` 안에서 수행한다.
- VS Code SSH 또는 노트북 연결이 끊겨도 서버와 tmux가 살아 있으면 작업은 계속된다.
- 서버가 종료되거나 재부팅되면 worker는 중단되지만, 같은 output directory로 다시 실행하면 성공한 ID를 건너뛰고 이어서 처리한다.

### 이미지와 캡션 대응 관계

- JSON의 `image` 필드에 적힌 상대 경로를 `/mnt3/jisung/recaption/data/LLaVA-Pretrain` 아래에서 찾는다.
- 예: `00223/002239345.jpg` -> `/mnt3/jisung/recaption/data/LLaVA-Pretrain/00223/002239345.jpg`
- 생성 결과는 GPU shard별 `part_00.jsonl`부터 `part_03.jsonl`에 저장된다.

## 확정 프롬프트

아래 프롬프트를 **그대로** production recaption에 사용한다. `prompt_kakao.txt`와 `recaption.py`의 기본 프롬프트도 아래 문구와 동일하다.

```text
Generate a detailed and visually grounded description of the image.

Describe all visually salient information, including:
- the overall scene and setting,
- objects and people,
- visual attributes such as color, shape, material, appearance, and state,
- actions,
- interactions between objects or people,
- spatial relationships between objects,
- relevant background elements,
- and clearly visible text when applicable.

Pay particular attention to individual objects and their relationships.

Only describe information that is visually supported by the image.
Do not infer hidden intentions, identities, causes, locations, brands,
or events unless they are clearly visible.

Write one coherent, detailed natural-language paragraph.
Avoid unnecessary repetition.
The length should adapt to the visual complexity of the image.
```

## 운영 원칙

- 이전 프롬프트로 만든 테스트/production 결과는 새 production run에 섞지 않는다.
- production을 시작한 뒤에는 프롬프트, model, sharding, visual-token budget, dtype 설정을 임의로 바꾸지 않는다.
- `launch_4gpu.sh`는 output directory에 `run_config.json`을 기록하고 설정이 달라진 재실행을 거부한다.
- 동일한 output directory에 두 launcher가 동시에 쓰지 못하도록 lock을 사용한다.
- crash로 JSONL 마지막 줄이 일부만 기록된 경우 다음 실행에서 그 마지막 partial record만 복구하고 해당 sample을 다시 생성한다.
- `review.py`로 이미지와 생성 캡션을 함께 샘플링하여 중간 검수한다.

자세한 실행 명령과 tmux/resume 절차는 루트 `README.md`를 따른다.
