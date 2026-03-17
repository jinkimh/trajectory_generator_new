# main_globaltraj.py 알고리즘 설명

이 문서는 `scripts/main_globaltraj.py`에서 수행되는 **레이스라인 최적화 파이프라인**과
핵심 알고리즘(목적함수/제약/구성)을 요약합니다.  
코드는 `trajectory_planning_helpers`(TPH)와 `opt_mintime_traj` 모듈을 직접 호출하는
구성입니다.

---

## 1) 입력(Inputs)

### 1.1 트랙/맵 관련
- **track csv**: `outputs/<map>/centerline.csv`
  - 형식: `[x, y, w_tr_right, w_tr_left]` 또는 `[x, y, width]`
  - `helper_funcs_glob.src.import_track.import_track`에서 파싱
- **frictionmap** (mintime + var_friction 사용 시)
  - `inputs/frictionmaps/<track>_tpamap.csv`
  - `inputs/frictionmaps/<track>_tpadata.json`

### 1.2 차량 파라미터
`params/racecar.ini`
- `veh_params` (일반)
- `optim_opts_*` (최적화 옵션)
- `vehicle_params_mintime`, `tire_params_mintime`, `pwr_params_mintime` (mintime 전용)
- `ggv.csv`, `ax_max_machines.csv` (가속도/동역학 제약)

### 1.3 설정/옵션
`config/params.yaml` 및 `config/pipeline.yaml`에서 map_name 등 기본값 로드.

---

## 2) 전처리(Preprocessing)

1. **트랙 로딩**  
   `import_track()`에서 centerline과 폭(`w_tr_right`, `w_tr_left`)을 읽음.

2. **Spline 근사 및 보간**
   - `prep_track()` → spline 근사 및 재샘플링
   - 출력:
     - `reftrack_interp` (스무딩된 트랙)
     - `normvec_normalized_interp` (법선 벡터)
     - `a_interp`, `coeffs_x_interp`, `coeffs_y_interp`

3. **법선 교차 검사**
   - 법선이 교차하면 오류 발생 → 트랙 스무딩 필요

---

## 3) 최적화 종류

`opt_type`에 따라 알고리즘이 달라집니다.

## 3.0 공통 수식(센터라인 기반 파라미터화)

센터라인 \(\mathbf{c}(s)\)과 단위 법선 \(\mathbf{n}(s)\)를 이용해 레이스라인을
\(\alpha(s)\)로 파라미터화합니다.

\[
\mathbf{r}(s) = \mathbf{c}(s) + \alpha(s)\,\mathbf{n}(s)
\]

트랙 경계(폭) 제약:

\[
-w_R(s) + \frac{w_{\text{veh}}}{2} \le \alpha(s) \le w_L(s) - \frac{w_{\text{veh}}}{2}
\]

### 3.1 shortest_path
**목적:** 경로 길이 최소화  
**제약:** 트랙 경계(폭) 안에서 이동  
관련 함수: `tph.opt_shortest_path.opt_shortest_path`

### 3.2 mincurv
**목적:** 곡률(κ) 최소화  
**제약:** 트랙 경계 및 곡률 제한(`kappa_bound`)  
관련 함수: `tph.opt_min_curv.opt_min_curv`

### 3.3 mincurv_iqp
**목적:** 곡률 최소화 (Iterative QP)  
**제약:** 위와 동일  
**구성:** 반복적으로 QP를 풀어 곡률 오차를 줄임  
관련 함수: `tph.iqp_handler.iqp_handler`

### 3.4 mintime
**목적:** 랩타임 최소화  
**제약:** 차량 동역학 + 타이어 마찰 + 파워/에너지 제약  
관련 함수: `opt_mintime_traj.src.opt_mintime.opt_mintime`

---

## 4) mintime 최적화 상세

`opt_mintime`는 **비선형 최적화(NLP)**를 IPOPT로 해결합니다.

### 4.1 결정 변수 (Decision Variables)
상태 변수(예):
- `v` (속도), `beta` (사이드슬립), `omega_z` (요 레이트)
- `n` (트랙 법선 방향 위치), `xi` (타이어 슬립 등 내부 상태)
- 전력/열 모델 사용 시: 모터/배터리/인버터/쿨링 상태, SOC 등

제어 변수(예):
- `delta`(조향각), `f_drive`(구동), `f_brake`(제동), `gamma_y`(횡하중 전달)

### 4.2 목적함수 (Objective)
랩타임 최소화 + 제어 입력 변화량 정규화.

- 기본 목적: `∑ dt`
- 정규화:
  - `penalty_F * ||Δ(F_drive + F_brake)||^2`
  - `penalty_delta * ||Δ(delta)||^2`

연속 형태로 쓰면:

\[
\min_{x,u,\alpha}\; \int_0^L \frac{1}{v(s)}\, ds
 + \lambda_F \|\Delta(F_{\text{drive}}+F_{\text{brake}})\|^2
 + \lambda_\delta \|\Delta \delta\|^2
\]

### 4.3 제약조건 (Constraints)

1. **트랙 경계 제약**
   - `n`(법선방향 오프셋)이  
     `[-w_tr_right + width_opt/2, w_tr_left - width_opt/2]` 범위에 존재

   \[
   -w_R(s) + \frac{w_{\text{veh}}}{2} \le \alpha(s) \le w_L(s) - \frac{w_{\text{veh}}}{2}
   \]

2. **동역학 콜로케이션 제약**
   - 차량 상태방정식을 collocation 형태로 추가

   \[
   \dot{x} = f(x,u,\kappa(\mathbf{r}(s)))
   \]

3. **파워 제한**
   - `v * f_drive ≤ power_max`

4. **타이어 마찰 제약 (Kamm’s circle)**
   - 각 타이어의 `(fx, fy)`가 마찰 원 내에 있어야 함

   \[
   \left(\frac{F_x}{\mu F_z}\right)^2 + \left(\frac{F_y}{\mu F_z}\right)^2 \le 1
   \]

5. **횡하중 전달 제약**
   - `gamma_y`로 균형 조건 부여

6. **구동/제동 동시 사용 방지**
   - `f_drive * f_brake ≤ 0`

7. **액추에이터 동역학**
   - 조향/구동/제동의 시간상 제한 (`t_delta`, `t_drive`, `t_brake`)

8. **safe_traj 옵션 (가속도 타원 제약)**
   - `(ax/ax_safe)^2 + (ay/ay_safe)^2 ≤ 1`

   \[
   \left(\frac{a_x}{a_{x,\text{safe}}}\right)^2 +
   \left(\frac{a_y}{a_{y,\text{safe}}}\right)^2 \le 1
   \]

9. **에너지 제한 (optional)**
   - 누적 소비 에너지 ≤ `energy_limit`

10. **주기 경계 조건**
    - 시작 상태 = 끝 상태 (랩 종료 연속성)

---

## 5) 후처리

1. **레이스라인 생성**
   - `create_raceline()`으로 일정 간격으로 보간

2. **곡률/헤딩 계산**
   - `calc_head_curv_an()`으로 스플라인 기반 곡률 계산

3. **속도 프로파일 계산**
   - mintime 미사용 시 `calc_vel_profile()`로 ggv 기반 속도 계산

4. **랩타임 계산**
   - `calc_t_profile()` 통해 lap time 출력

5. **검증**
   - `check_traj()`로 충돌/제약 위반 체크

---

## 6) 요약

- `main_globaltraj.py`는 **트랙 스무딩 → 최적화 → 속도 프로파일 → 검증** 순으로 동작합니다.
- mincurv/shortest_path는 주로 **형상 기반 최적화**,
- mintime은 **동역학/마찰/에너지 제약이 포함된 비선형 최적화**입니다.
- 모든 출력은 `outputs/<map>/` 하위로 저장됩니다.
