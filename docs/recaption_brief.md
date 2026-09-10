# LLaVA 558K Recaptioning 전달 사항과 프롬프트

## 목적

기존 이미지에 대응하는 캡션을 더 자세하고 정확하게 재생성한다. 생성된 설명이 실제 이미지와 잘 맞는지 실행 중간중간 직접 확인한다. 모델을 새로 학습하는 작업이 아니라, 이미 학습된 이미지·언어 모델로 새 캡션을 생성하는 작업이다.

## 카카오톡 전달 내용

아래는 전달받은 카카오톡 이미지의 내용을 주제별로 정리한 기록이다. 프롬프트는 다음 절에 별도로 원문을 보존했다.

### 데이터와 스크립트 위치

- 작업 폴더는 `/mnt3/jisung/recaption`이다.
- 해당 폴더에 데이터셋과 참고할 스크립트가 준비되어 있으므로 이를 사용해 실험을 실행한다.

### 모델과 메모리

- 카톡에서는 모델명을 “Qwen2.5-7B-Instruct”로 표기했다.
- 모델 가중치가 대략 14GB여서 12GB GPU에 그대로 올리기 어렵다는 설명과 함께, 4-bit quantization을 적용하는 스크립트를 준비했다고 안내했다.
- **코드 확인 사항:** 실제 `recaption.py`의 기본 모델은 이미지 입력을 처리하는 `Qwen/Qwen2.5-VL-7B-Instruct`이다. 카톡 표기와 구분한다.

### GPU 사용과 작업 분배

- GPU 0, 1, 2, 3번 네 대를 모두 사용할 수 있다고 안내했다.
- 558K 데이터셋을 네 몫으로 나누고 GPU 한 대당 한 몫씩 맡겨 병렬로 실행한다.
- 코드에서는 모델 하나를 GPU 네 대에 나누는 것이 아니라, 각 GPU에서 별도 모델 인스턴스로 서로 다른 이미지를 처리한다.

### 실행과 원격 연결

- 카톡에서는 한 번 실행하면 원격 연결을 계속 유지하지 않아도 되며, 중간중간 정상 진행 여부를 확인하라고 안내했다.
- 실행 시 `python`이 아니라 `python3.10`을 사용하라고 명시했다.
- **실행 시 주의:** 원격 연결 종료 후에도 작업을 유지하려면 README에 안내된 `tmux` 또는 `screen` 안에서 실행한다. 일반 터미널에서 명령을 실행했다는 사실만으로 연결 종료 후 작업 유지가 보장되지는 않는다.

### 이미지와 캡션 대응 관계

- JSON의 `image` 필드에 적힌 경로를 따라가면 해당 캡션과 짝을 이루는 이미지를 확인할 수 있다.
- 스크린샷에서 연두색으로 표시된 `blip_caption`이 기존 캡션이며, 이를 더 자세하고 정확한 설명으로 재생성하는 것이 목표다.
- 스크린샷 예시:
  - JSON 파일: `blip_laion_cc_sbu_558k_meta.json`
  - 이미지 ID: `002239345`
  - 이미지 상대 경로: `00223/002239345.jpg`
  - 기존 캡션: `a grey watch with an army style strap`

### 결과 확인

- 실행 결과로 새 캡션이 담긴 파일이 생성된다.
- 새로 생성된 detailed 캡션이 실제 paired 이미지와 잘 정합하는지 중간중간 직접 확인한다.
- 특히 주요 객체, 시각적 속성, 객체 간 관계가 이미지와 일치하는지 살핀다.
- **코드 확인 사항:** 생성 중간 결과는 GPU별 `part_00.jsonl`부터 `part_03.jsonl`까지 저장된다. 이후 `merge_llava.py`로 LLaVA 형식의 최종 JSON을 만든다.

## 전달받은 프롬프트 원문

아래 프롬프트는 사용자가 전달한 문구를 그대로 기록한 것이다. 이 문서에 기록하는 것만으로 실행 코드의 기본 프롬프트가 변경되지는 않는다.

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

## 프롬프트 적용 전 확인

기록 시점의 `recaption.py` 기본 프롬프트는 위 원문과 취지는 비슷하지만 문구가 다르다. 예를 들어 `salient objects and people`, `count`, `pairwise relationships` 및 일반 지식 추가 금지 문장이 포함되어 있다.

전달받은 원문으로 실행하려면 실제 적용 프롬프트를 확인해야 한다. `recaption.py`는 `--prompt-file` 옵션을 지원하지만, 기록 시점의 `launch_4gpu.sh`에는 해당 옵션을 전달하는 부분이 없다. 이 문서 추가에서는 프롬프트나 실행 스크립트를 수정하지 않았다.

## 참고 파일

- [README](../README.md): 설치, 소량 시험, 전체 실행, 병합, 검증 순서
- [recaption.py](../recaption.py): 이미지와 프롬프트로 새 캡션 생성
- [launch_4gpu.sh](../launch_4gpu.sh): GPU 0~3에서 병렬 실행
- [merge_llava.py](../merge_llava.py): 생성한 캡션을 원본 LLaVA 형식에 반영
- [validate_llava.py](../validate_llava.py): 중복, 이미지 누락, 빈 캡션 등 검사

카톡 전달 내용과 이후 코드 확인에 따른 주의사항은 구분하여 기록했다. 친구와 번갈아 몇만 장씩 실행하는 방식은 별도의 후속 운영 요청이며, 카톡 원래 지시사항에는 포함하지 않았다.
