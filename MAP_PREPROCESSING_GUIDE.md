# 맵 전처리 가이드

## 개요

`preprocess_map.py`는 트랙 맵 이미지를 전처리하여 다음 작업을 수행합니다:

1. **픽셀 분류**: 흰색(주행 가능) vs 회색/검은색(비트랙)
2. **노이즈 제거**: 작은 노이즈 제거
3. **구멍 채우기**: 작은 구멍 자동 채우기
4. **경계 매끄럽게**: 불연속적인 경계를 연속적으로 처리
5. **장애물 처리**: 중간 장애물을 매끄럽게 처리

## 사용 방법

### 기본 사용법

```bash
cd /home/jin/ros2_prj/traj_generator
source bin/activate
cd trajectory_generator

# 기본 실행 (config/params.yaml에서 맵 이름 읽기)
python preprocess_map.py

# 특정 맵 지정
python preprocess_map.py --map Bexco

# 처리 과정 보기 (각 단계를 이미지로 확인)
python preprocess_map.py --show

# 출력 파일 접미사 지정
python preprocess_map.py --suffix _cleaned
```

### 옵션 설명

- `--map MAP`: 처리할 맵 이름 (확장자 제외, 기본값: config에서 읽기)
- `--ext EXT`: 맵 파일 확장자 (기본값: `.pgm`)
- `--show`: 각 처리 단계를 이미지로 표시
- `--suffix SUFFIX`: 출력 파일 접미사 (기본값: `_processed`)

## 처리 단계

### Step 1: 픽셀 분류
- 흰색 (200 이상): 주행 가능한 트랙 → 255 (흰색)
- 회색/검은색 (200 미만): 비트랙 → 0 (검은색)

### Step 2: 노이즈 제거
- Morphological opening으로 작은 노이즈 제거

### Step 3: 구멍 채우기
- Morphological closing으로 작은 구멍 자동 채우기

### Step 4: 경계 매끄럽게
- Gaussian blur로 경계를 부드럽게 처리

### Step 5: 장애물 제거
- 연결된 구성 요소 분석으로 큰 장애물 제거
- 가장 큰 트랙 영역만 유지

### Step 6: 최종 매끄럽게
- Median filter 적용
- 최종 morphological operations
- 경계를 더 부드럽게 만들기

## 출력 파일

처리된 파일은 다음 위치에 저장됩니다:

- **이미지**: `maps/{map_name}_processed.pgm`
- **YAML**: `maps/{map_name}_processed.yaml`

예: `Bexco.pgm` → `Bexco_processed.pgm`

## 예제

### Bexco 맵 전처리

```bash
# 처리 과정 보면서 실행
python preprocess_map.py --map Bexco --show

# 자동으로 처리 (이미지 표시 없음)
python preprocess_map.py --map Bexco
```

### 전처리된 맵 사용

전처리된 맵을 사용하려면:

1. **config/params.yaml 수정**:
   ```yaml
   map_name: 'Bexco_processed'
   ```

2. 또는 **원본 파일 교체**:
   ```bash
   cp maps/Bexco_processed.pgm maps/Bexco.pgm
   cp maps/Bexco_processed.yaml maps/Bexco.yaml
   ```

## 주의사항

1. **GUI 환경 필요**: `--show` 옵션 사용 시 X11 디스플레이 필요
2. **원본 백업**: 전처리 전 원본 파일을 백업하는 것을 권장
3. **파라미터 조정**: 필요시 `preprocess_map.py`의 임계값 조정 가능
   - `classify_pixel()`: 흰색 임계값 (기본: 200)
   - Morphological kernel 크기
   - Blur 강도

## 문제 해결

### 이미지가 너무 어둡거나 밝을 때

`preprocess_map.py`의 `classify_pixel()` 함수에서 임계값 조정:

```python
def classify_pixel(value):
    if value >= 200:  # 이 값을 조정 (예: 180, 220 등)
        return 255
    else:
        return 0
```

### 경계가 너무 거칠 때

`preprocess_map.py`의 blur 파라미터 조정:

```python
# Step 4에서
blurred = cv2.GaussianBlur(filled_img, (5, 5), 1.5)  # (5,5)와 1.5 조정
```

### 작은 장애물이 남아있을 때

`preprocess_map.py`의 morphological operations 반복 횟수 증가:

```python
# Step 2, 3에서
iterations=2  # 이 값을 증가 (예: 3, 4)
```

## 다음 단계

전처리된 맵을 사용하여:

1. **Centerline 생성**:
   ```bash
   python lane_generator.py
   ```

2. **Trajectory 생성**:
   ```bash
   python main_globaltraj.py
   ```
