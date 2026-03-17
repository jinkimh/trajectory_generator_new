# trajectory_generator_new (한국어)

ROS2 `ament_python` 패키지로 구성된 **경로 생성/레이스라인 최적화 파이프라인**입니다.  
기존 `trajectory_generator`의 동작은 유지하면서 폴더 구조를 다음과 같이 정리했습니다.

## 폴더 구조

```
trajectory_generator_new/
  scripts/                # 실행 스크립트
  config/                 # pipeline.yaml 등 설정
  maps/                   # 맵 이미지 + yaml
  inputs/                 # frictionmaps, veh_dyn_info, tracks
  outputs/                # 생성 결과 (centerline/raceline 등)
  helper_funcs_glob/      # 공용 헬퍼
  opt_mintime_traj/       # mintime 최적화 모듈
  frictionmap/            # 마찰 맵 모듈
  params/                 # 파라미터 파일
  run_pipeline_and_plot.sh # 파이프라인 + 플로터
  requirements.txt
  setup_venv.sh
  install_requirements.sh
  setup.py / setup.cfg / package.xml
```

## 빠른 시작

1) venv 생성 + requirements 설치
```bash
cd /home/jin/ros2_prj/trajectory_generator_new
./setup_venv.sh
source .venv/bin/activate
```

2) 파이프라인 실행
```bash
python3 /home/jin/ros2_prj/trajectory_generator_new/scripts/run_pipeline.py
```

3) 파이프라인 + 자동 플로팅
```bash
./run_pipeline_and_plot.sh
```

## 아키텍처(개요)

```
maps/<map>.(png|pgm)
  -> track_boundary_split.py (옵션)
  -> preprocess_map.py
  -> gen_frictionmap_from_map.py (옵션)
  -> lane_generator_multi_centerline.py
  -> main_globaltraj_multi.py (mintime/mincurv)
  -> plot_raceline_on_map.py (overlay)
```

## pipeline.yaml 설정

`config/pipeline.yaml`에서 아래를 주로 설정합니다.

- `map_name`: 확장자 제외 맵 이름
- `map_ext`: `.png` 또는 `.pgm`
- `tracksplit`: 경계선 맵 분리(옵션)
- `preprocess`: 전처리
- `frictionmap`, `centerlines`, `mintime`: 후속 단계

플레이스홀더 사용:
```
map: "{map_name}_processed"
ext: "{map_ext}"
```

`tracksplit`을 켜면 이후 단계의 `map_name`은 자동으로  
`{map_name}{suffix}`로 갱신됩니다.

※ RViz/launch 파일은 제거되었으며, 시각화는 `plot_raceline_on_map.py`를 사용합니다.

## 실행 파일 사용법

모든 실행 파일은 `scripts/`에 있고, **패키지 루트를 기준으로 경로를 참조**합니다.

### 1) track_boundary_split.py
경계선만 있는 도넛형 맵에서 **외곽/내곽 경계 및 트랙 영역 분리**
```bash
python3 scripts/track_boundary_split.py \
  --map icra --ext .png \
  --edge-method canny --canny-low 30 --canny-high 120 \
  --edge-dilate 3 --close-ksize 5 --min-edge-area 0 \
  --suffix _tracksplit --save-yaml
```

### 2) preprocess_map.py
트랙 영역 추출 전처리
```bash
python3 scripts/preprocess_map.py --map levine_2nd_tracksplit --ext .pgm --preset icra_2
```

### 3) gen_frictionmap_from_map.py
맵 기반 마찰맵 생성
```bash
python3 scripts/gen_frictionmap_from_map.py \
  --map-file maps/levine_2nd_tracksplit_processed.pgm \
  --yaml-file maps/levine_2nd_tracksplit_processed.yaml \
  --out-dir inputs/frictionmaps --out-name levine_2nd
```

### 4) lane_generator_multi_centerline.py
센터라인 생성
```bash
python3 scripts/lane_generator_multi_centerline.py --map icra_2_processed --ext .pgm --headless
```

### 5) main_globaltraj_multi.py
레이스라인 최적화(민시간/최소곡률)
```bash
python3 scripts/main_globaltraj_multi.py --map icra_2 --centerline-map icra_2_processed --headless
```

### 6) plot_raceline_on_map.py
레이스라인 CSV를 **원본 맵** 위에 표시 (기본)
```bash
python3 scripts/plot_raceline_on_map.py \
  --csv outputs/icra_2_tracksplit/multi_mintime/traj_race_cl_00.csv \
  --map icra_2.pgm
```
참고:
- `--map`은 파일 경로나 확장자 포함 이름도 가능
- 전처리 맵을 쓰려면 `--use-processed` 옵션 사용

### 7) main_globaltraj.py
단일 경로 최적화 (inputs/tracks 기반)
```bash
python3 scripts/main_globaltraj.py
```

### 8) main_gen_frictionmap.py
레거시 마찰맵 생성
```bash
python3 scripts/main_gen_frictionmap.py
```

### 9) check_image_values.py
이미지 픽셀 값 분포 확인
```bash
python3 scripts/check_image_values.py
```

### 10) raceline_scripts.py
플로팅/분석 보조 스크립트 (필요한 함수만 사용)

### 11) run_pipeline_and_plot.sh
파이프라인 실행 후 mintime 성공 시 최적 레이스라인, 실패 시 센터라인으로 플로팅
```bash
./run_pipeline_and_plot.sh
./run_pipeline_and_plot.sh icra_2
./run_pipeline_and_plot.sh --map ./maps/icra_2.pgm --csv ./outputs/icra_2_tracksplit/multi_mintime/traj_race_cl_00.csv --no-pipeline
```
참고:
- 인자가 없으면 `config/pipeline.yaml`의 `map_name`을 사용합니다.

## ROS2 패키지로 빌드
```bash
cd /home/jin/ros2_prj
colcon build --packages-select trajectory_generator_new
```

직접 실행:
```bash
python3 /home/jin/ros2_prj/trajectory_generator_new/scripts/run_pipeline.py
```

## 자주 발생하는 문제

- `No module named 'casadi'`
  ```bash
  source /home/jin/ros2_prj/trajectory_generator_new/.venv/bin/activate
  ```
- `matplotlib` 오류 (tkinter)
  ```bash
  sudo apt install python3-tk
  ```
- `quadprog` 빌드 오류
  ```bash
  sudo apt install python3-dev gfortran build-essential
  ```
- `install_requirements.sh`는 `quadprog`를 소스 빌드로 재설치하고,
  `trajectory_planning_helpers` 호환 패치를 적용합니다.

## 참고
- 모든 결과는 `outputs/`에 저장됩니다.
- `.png`/`.pgm` 전환은 `map_ext`로 통일됩니다.
- 단계별 입력/출력 경로가 꼬이면 `map_name`, `map_ext`, `tracksplit` 여부를 확인하세요.
