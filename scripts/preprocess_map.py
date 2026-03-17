#!/usr/bin/env python3
"""
Map Preprocessing Script
- 흰색: 주행 가능한 트랙
- 회색/검은색: 트랙이 아님 → 검게 칠하기
- 경계의 불연속 부분을 연속적으로 처리 (매끄럽게)
- 중간 장애물도 매끄럽게 처리
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path

from trajectory_generator_new.config_utils import load_map_defaults


def show_image(img, title="Image", wait=True):
    """이미지를 보여주고 사용자 입력을 기다림"""
    if len(img.shape) == 2:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display_img = img.copy()
    
    # 이미지 크기 조정 (너무 크면)
    h, w = display_img.shape[:2]
    max_size = 1200
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    cv2.imshow(title, display_img)
    if wait:
        print("Press any key to continue (or 'q' to quit)...")
        try:
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    raise SystemExit("User requested exit.")
                if key != 255:
                    cv2.destroyAllWindows()
                    return True
                # If the window was closed manually, stop waiting.
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    return True
        except KeyboardInterrupt:
            cv2.destroyAllWindows()
            raise
    return True


def classify_pixel(value, threshold):
    """픽셀 값을 분류: threshold 이상은 트랙(흰색), 미만은 비트랙(검은색)"""
    return 255 if value >= threshold else 0


def find_discontinuities(binary_img, gap_threshold=2):
    """
    흰색과 검은색 경계에서 불연속 부분(갭)을 찾음
    경계를 따라가면서 흰색 영역 사이의 검은색 갭을 찾아서 표시
    
    Args:
        binary_img: 이진 이미지 (255=흰색, 0=검은색)
        gap_threshold: 갭으로 간주할 최소 픽셀 수 (기본: 2)
    
    Returns:
        gap_mask: 갭으로 표시된 마스크 (255=갭, 0=아님)
    """
    h, w = binary_img.shape
    gap_mask = np.zeros_like(binary_img)
    
    # 경계 찾기 (흰색과 검은색이 만나는 부분)
    # 경계 픽셀 찾기: 흰색 픽셀의 8-이웃 중 검은색이 있는 경우
    boundary_pixels = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if binary_img[y, x] == 255:  # 흰색 픽셀
                # 8-이웃 확인
                has_black_neighbor = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary_img[ny, nx] == 0:  # 검은색 이웃
                                has_black_neighbor = True
                                break
                    if has_black_neighbor:
                        break
                if has_black_neighbor:
                    boundary_pixels.append((y, x))
    
    # 각 경계 픽셀에서 8방향으로 갭 찾기
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    for y, x in boundary_pixels:
        for dy, dx in directions:
            # 이 방향으로 검은색이 연속되는 길이 측정
            gap_length = 0
            ny, nx = y + dy, x + dx
            
            # 검은색이 연속되는 동안 진행
            while (0 <= ny < h and 0 <= nx < w and 
                   binary_img[ny, nx] == 0):
                gap_length += 1
                ny += dy
                nx += dx
                
                # 너무 길면 중단 (무한 루프 방지)
                if gap_length > 20:
                    break
            
            # 갭이 threshold 이상이고, 그 끝에 다시 흰색이 있으면
            if gap_threshold <= gap_length <= 20:
                ny_end = ny
                nx_end = nx
                if (0 <= ny_end < h and 0 <= nx_end < w and 
                    binary_img[ny_end, nx_end] == 255):
                    # 갭 영역을 검은색으로 표시 (이미 검은색이지만 마스크에 표시)
                    for i in range(1, gap_length + 1):
                        gy = y + dy * i
                        gx = x + dx * i
                        if 0 <= gy < h and 0 <= gx < w:
                            gap_mask[gy, gx] = 255
    
    return gap_mask


def interpolate_gaps(binary_img, gap_mask):
    """
    갭 영역을 보간하여 검은색으로 채움
    
    Args:
        binary_img: 이진 이미지
        gap_mask: 갭 마스크
    
    Returns:
        filled_img: 갭이 채워진 이미지
    """
    filled_img = binary_img.copy()
    
    # 갭 마스크가 있는 부분을 검은색으로 채우기
    filled_img[gap_mask == 255] = 0
    
    return filled_img


def smooth_contours(binary_img, epsilon_ratio=0.005, min_area=50):
    """
    컨투어 기반 경계 평활화: 윤곽선을 단순화하여 톱니를 제거

    Args:
        binary_img: 이진 이미지 (255=흰색, 0=검은색)
        epsilon_ratio: 컨투어 둘레 대비 근사 정도
        min_area: 작은 컨투어 제거 임계값

    Returns:
        smoothed_img: 평활화된 이진 이미지
    """
    contours, hierarchy = cv2.findContours(
        binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    smoothed = np.zeros_like(binary_img)
    if hierarchy is None:
        return smoothed

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        epsilon = max(1.0, epsilon_ratio * perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # hierarchy: [next, prev, child, parent]
        parent = hierarchy[0][idx][3]
        color = 255 if parent == -1 else 0
        cv2.drawContours(smoothed, [approx], -1, color, thickness=cv2.FILLED)

    return smoothed


def preprocess_track_image(
    input_img,
    show_steps=False,
    gap_threshold=2,
    threshold=None,
    blur_ksize=5,
    morph_ksize=3,
    smooth_blur_ksize=5,
    smooth_mode="blur",
    contour_epsilon=0.005,
    contour_min_area=50,
    nontrack_threshold=60,
    nontrack_dilate=3,
    track_erode=0,
    strict_white=False,
    track_value=254,
    track_tolerance=0,
    post_min_area=0,
    post_keep_largest=False,
    invert_output=False,
    keep_enclosed=False,
):
    """
    트랙 이미지를 전처리: 주행 구간은 흰색, 나머지는 모두 검은색
    
    Args:
        input_img: 입력 이미지 (grayscale)
        show_steps: 각 단계를 보여줄지 여부
        gap_threshold: 사용 안 함 (호환성 유지)
        threshold: 수동 임계값 (None이면 Otsu 자동 계산)
        blur_ksize: 입력 전처리 블러 커널 크기 (홀수 권장)
        morph_ksize: 모폴로지 커널 크기 (홀수 권장)
        smooth_blur_ksize: 경계 평활화용 블러 커널 크기 (홀수 권장)
        smooth_mode: 경계 평활화 방식 ("blur", "contour", "none")
        contour_epsilon: 컨투어 스무딩 비율 (둘레 대비, 기본 0.005)
        contour_min_area: 작은 컨투어 제거용 최소 면적
        nontrack_threshold: 확실한 비트랙(검은 영역) 임계값
        nontrack_dilate: 비트랙 보호 영역 확장 크기
        track_erode: 트랙을 안쪽으로 줄이는 정도 (0이면 비활성)
        strict_white: 특정 밝기 이상만 트랙으로 인정 (회색 혼입 방지)
        track_value: strict_white 기준 밝기 값
        track_tolerance: track_value에서 허용하는 오차
        post_min_area: 최종 결과에서 제거할 최소 면적 기준 (0이면 비활성)
        post_keep_largest: 최종 결과에서 가장 큰 성분만 유지
    
    Returns:
        processed_img: 전처리된 이미지 (흰색=주행 가능, 검은색=주행 불가)
    """
    print("Starting image preprocessing...")
    print("Rule: Bright pixels are drivable (white), others are non-drivable (black)")

    # 1) 약한 블러로 노이즈/미세한 경계 잡음 완화
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(input_img, (blur_ksize, blur_ksize), 0)

    # 1-1) 확실한 비트랙(검은 영역) 보호 마스크
    nontrack_mask = (input_img <= nontrack_threshold).astype(np.uint8) * 255
    if nontrack_dilate > 0:
        if nontrack_dilate % 2 == 0:
            nontrack_dilate += 1
        nontrack_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (nontrack_dilate, nontrack_dilate)
        )
        nontrack_mask = cv2.dilate(nontrack_mask, nontrack_kernel, iterations=1)

    # 2) 임계값으로 트랙/비트랙 분리
    if strict_white:
        track_min = max(0, track_value - track_tolerance)
        processed_img = np.where(input_img >= track_min, 255, 0).astype(np.uint8)
        nontrack_mask = np.where(input_img < track_min, 255, 0).astype(np.uint8)
        print(f"Threshold: strict white (>= {track_min})")
    elif threshold is None:
        _, processed_img = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        print("Threshold: Otsu (auto)")
    else:
        processed_img = np.where(blurred >= threshold, 255, 0).astype(np.uint8)
        print(f"Threshold: manual ({threshold})")

    # 3) 모폴로지로 작은 톱니/잡음 제거
    if morph_ksize % 2 == 0:
        morph_ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_OPEN, kernel)
    processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, kernel)

    # 3-1) 외곽과 연결된 흰색 영역 제거 (벽으로 둘러싸인 영역만 유지)
    if keep_enclosed:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_img)
        h, w = processed_img.shape
        keep = np.zeros_like(processed_img)
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]
            if x == 0 or y == 0 or (x + ww) >= w or (y + hh) >= h:
                continue
            keep[labels == i] = 255
        processed_img = keep

    # 4) 가장 큰 연결 성분만 유지 (밝은 잡음 제거)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_img)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        processed_img = np.where(labels == largest, 255, 0).astype(np.uint8)

    # 5) 경계 평활화
    if smooth_mode == "blur":
        if smooth_blur_ksize % 2 == 0:
            smooth_blur_ksize += 1
        smooth = cv2.GaussianBlur(processed_img, (smooth_blur_ksize, smooth_blur_ksize), 0)
        _, processed_img = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY)
    elif smooth_mode == "contour":
        processed_img = smooth_contours(
            processed_img,
            epsilon_ratio=contour_epsilon,
            min_area=contour_min_area,
        )
    elif smooth_mode == "none":
        pass
    else:
        print(f"Unknown smooth_mode '{smooth_mode}', skipping smoothing.")

    # 6) 비트랙 보호 마스크 적용 (경계 침범 방지)
    processed_img[nontrack_mask == 255] = 0

    # 7) 필요시 트랙을 안쪽으로 살짝 줄이기
    if track_erode > 0:
        if track_erode % 2 == 0:
            track_erode += 1
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (track_erode, track_erode)
        )
        processed_img = cv2.erode(processed_img, erode_kernel, iterations=1)

    # 8) 최종 노이즈 제거 (blur 재이진화 후 생긴 작은 섬 제거)
    if post_keep_largest or post_min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed_img)
        if num_labels > 1:
            if post_keep_largest:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                processed_img = np.where(labels == largest, 255, 0).astype(np.uint8)
            elif post_min_area > 0:
                keep = np.zeros_like(processed_img)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= post_min_area:
                        keep[labels == i] = 255
                processed_img = keep

    if invert_output:
        processed_img = 255 - processed_img

    if show_steps:
        show_image(input_img, "Original Image")
        show_image(processed_img, "Processed Image (White=Drivable, Black=Non-drivable)")
    
    print("Preprocessing completed!")
    return processed_img


def process_map(
    map_name,
    map_ext=".pgm",
    show_steps=False,
    output_suffix="_processed",
    gap_threshold=2,
    threshold=None,
    blur_ksize=5,
    morph_ksize=3,
    smooth_blur_ksize=5,
    smooth_mode="blur",
    contour_epsilon=0.005,
    contour_min_area=50,
    save_compare=False,
    nontrack_threshold=60,
    nontrack_dilate=3,
    track_erode=0,
    strict_white=False,
    track_value=254,
    track_tolerance=0,
    preset=None,
    post_min_area=0,
    post_keep_largest=False,
    invert_output=False,
    keep_enclosed=False,
):
    """
    맵 파일을 처리하여 전처리된 버전을 저장
    
    Args:
        map_name: 맵 이름 (확장자 제외)
        map_ext: 맵 파일 확장자
        show_steps: 각 단계를 보여줄지 여부
        output_suffix: 출력 파일에 추가할 접미사
    """
    module = ROOT
    
    # 입력 파일 경로
    input_img_path = os.path.join(module, "maps", map_name + map_ext)
    yaml_path = os.path.join(module, "maps", map_name + ".yaml")
    
    # 파일 존재 확인
    if not os.path.exists(input_img_path):
        print(f"Error: Input image not found: {input_img_path}")
        return False
    
    if not os.path.exists(yaml_path):
        print(f"Warning: YAML file not found: {yaml_path}")
    
    # 이미지 읽기
    print(f"Reading image: {input_img_path}")
    input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    
    if input_img is None:
        print(f"Error: Failed to read image: {input_img_path}")
        return False
    
    print(f"Image size: {input_img.shape}")
    
    # 원본 이미지 표시
    if show_steps:
        show_image(input_img, "Original Image")
    
    # 전처리
    if preset == "bexco":
        strict_white = True
        track_value = 254
        track_tolerance = 0
        smooth_mode = "blur"
        smooth_blur_ksize = 3
        nontrack_threshold = 60
        nontrack_dilate = 3
        track_erode = 1
        post_keep_largest = True
    elif preset == "bexco_smooth":
        strict_white = True
        track_value = 254
        track_tolerance = 0
        smooth_mode = "blur"
        smooth_blur_ksize = 5
        nontrack_threshold = 60
        nontrack_dilate = 3
        track_erode = 1
        post_keep_largest = True
    elif preset == "icra_2":
        # icra_2 map uses broad bright ranges (not strictly white)
        strict_white = False
        if threshold is None:
            threshold = 180
        smooth_mode = "blur"
        smooth_blur_ksize = 5
        nontrack_threshold = 60
        nontrack_dilate = 3
        track_erode = 0
        post_keep_largest = True
        keep_enclosed = True
    processed_img = preprocess_track_image(
        input_img,
        show_steps=show_steps,
        gap_threshold=gap_threshold,
        threshold=threshold,
        blur_ksize=blur_ksize,
        morph_ksize=morph_ksize,
        smooth_blur_ksize=smooth_blur_ksize,
        smooth_mode=smooth_mode,
        contour_epsilon=contour_epsilon,
        contour_min_area=contour_min_area,
        nontrack_threshold=nontrack_threshold,
        nontrack_dilate=nontrack_dilate,
        track_erode=track_erode,
        strict_white=strict_white,
        track_value=track_value,
        track_tolerance=track_tolerance,
        post_min_area=post_min_area,
        post_keep_largest=post_keep_largest,
        invert_output=invert_output,
        keep_enclosed=keep_enclosed,
    )
    
    # 결과 비교
    if show_steps:
        comparison = np.hstack([input_img, processed_img])
        show_image(comparison, "Before vs After")
    
    # 출력 파일 경로
    output_img_path = os.path.join(module, "maps", map_name + output_suffix + map_ext)

    # 저장
    print(f"Saving processed image: {output_img_path}")
    cv2.imwrite(output_img_path, processed_img)

    # 비교 저장: blur/contour 모드를 동시에 저장
    if save_compare:
        blur_img = preprocess_track_image(
            input_img,
            show_steps=False,
            gap_threshold=gap_threshold,
            threshold=threshold,
        blur_ksize=blur_ksize,
        morph_ksize=morph_ksize,
        smooth_blur_ksize=smooth_blur_ksize,
        smooth_mode="blur",
        contour_epsilon=contour_epsilon,
        contour_min_area=contour_min_area,
        nontrack_threshold=nontrack_threshold,
        nontrack_dilate=nontrack_dilate,
        track_erode=track_erode,
        strict_white=strict_white,
        track_value=track_value,
        track_tolerance=track_tolerance,
        post_min_area=post_min_area,
        post_keep_largest=post_keep_largest,
    )
        contour_img = preprocess_track_image(
            input_img,
            show_steps=False,
            gap_threshold=gap_threshold,
            threshold=threshold,
            blur_ksize=blur_ksize,
            morph_ksize=morph_ksize,
            smooth_blur_ksize=smooth_blur_ksize,
            smooth_mode="contour",
            contour_epsilon=contour_epsilon,
            contour_min_area=contour_min_area,
            nontrack_threshold=nontrack_threshold,
            nontrack_dilate=nontrack_dilate,
            track_erode=track_erode,
            strict_white=strict_white,
            track_value=track_value,
            track_tolerance=track_tolerance,
            post_min_area=post_min_area,
            post_keep_largest=post_keep_largest,
        )

        blur_path = os.path.join(module, "maps", map_name + "_blur" + map_ext)
        contour_path = os.path.join(module, "maps", map_name + "_contour" + map_ext)
        compare_path = os.path.join(module, "maps", map_name + "_compare" + map_ext)

        print(f"Saving blur image: {blur_path}")
        cv2.imwrite(blur_path, blur_img)
        print(f"Saving contour image: {contour_path}")
        cv2.imwrite(contour_path, contour_img)

        comparison = np.hstack([blur_img, contour_img])
        print(f"Saving comparison image: {compare_path}")
        cv2.imwrite(compare_path, comparison)
    
    # YAML 파일도 복사 (필요시 수정)
    if os.path.exists(yaml_path):
        output_yaml_path = os.path.join(module, "maps", map_name + output_suffix + ".yaml")
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # 이미지 파일명 업데이트
        if 'image' in yaml_data:
            yaml_data['image'] = map_name + output_suffix + os.path.splitext(yaml_data.get('image', ''))[1]
        
        with open(output_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        print(f"Saved YAML file: {output_yaml_path}")
    
    print(f"\nProcessing complete!")
    print(f"Original: {input_img_path}")
    print(f"Processed: {output_img_path}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess map image for trajectory generation')
    parser.add_argument('--map', type=str, default=None, help='Map name (without extension)')
    parser.add_argument('--ext', type=str, default='.pgm', help='Map file extension')
    parser.add_argument('--show', action='store_true', help='Show processing steps')
    parser.add_argument('--suffix', type=str, default='_processed', help='Output file suffix')
    parser.add_argument('--gap', type=int, default=2, help='Gap threshold in pixels (default: 2)')
    parser.add_argument('--threshold', type=int, default=None, help='Manual threshold (0-255). If omitted, use Otsu.')
    parser.add_argument('--blur', type=int, default=5, help='Pre-blur kernel size (odd, default: 5)')
    parser.add_argument('--morph', type=int, default=3, help='Morphology kernel size (odd, default: 3)')
    parser.add_argument('--smooth-blur', type=int, default=5, help='Boundary smoothing blur size (odd, default: 5)')
    parser.add_argument('--smooth-mode', type=str, default='blur', choices=['blur', 'contour', 'none'],
                        help='Boundary smoothing mode: blur | contour | none')
    parser.add_argument('--contour-epsilon', type=float, default=0.005,
                        help='Contour smoothing epsilon ratio (default: 0.005)')
    parser.add_argument('--contour-min-area', type=int, default=50,
                        help='Minimum contour area to keep (default: 50)')
    parser.add_argument('--nontrack-threshold', type=int, default=60,
                        help='Non-track protection threshold (default: 60)')
    parser.add_argument('--nontrack-dilate', type=int, default=3,
                        help='Non-track protection dilation size (odd, default: 3)')
    parser.add_argument('--track-erode', type=int, default=0,
                        help='Erode track inward (odd, default: 0 = off)')
    parser.add_argument('--strict-white', action='store_true',
                        help='Use only near-white pixels as track (bypass gray)')
    parser.add_argument('--track-value', type=int, default=254,
                        help='Track brightness value for strict-white (default: 254)')
    parser.add_argument('--track-tolerance', type=int, default=0,
                        help='Tolerance for strict-white threshold (default: 0)')
    parser.add_argument('--preset', type=str, default='bexco_smooth',
                        choices=['bexco', 'bexco_smooth', 'icra_2'],
                        help='Use a preset configuration (e.g., bexco)')
    parser.add_argument('--post-min-area', type=int, default=0,
                        help='Remove components smaller than this area after smoothing (default: 0)')
    parser.add_argument('--post-keep-largest', action='store_true',
                        help='Keep only the largest component after smoothing')
    parser.add_argument('--invert', action='store_true',
                        help='Invert output (swap drivable/non-drivable)')
    parser.add_argument('--keep-enclosed', action='store_true',
                        help='Keep only white components not touching the border')
    parser.add_argument('--save-compare', action='store_true',
                        help='Save blur/contour outputs and a side-by-side comparison')
    
    args = parser.parse_args()
    
    # config에서 맵 이름 읽기
    module = ROOT
    if args.map is None:
        map_name, _ = load_map_defaults(module)
    else:
        map_name = args.map
    
    print(f"Processing map: {map_name}")
    print("=" * 50)
    
    try:
        success = process_map(
            map_name=map_name,
            map_ext=args.ext,
            show_steps=args.show,
            output_suffix=args.suffix,
            gap_threshold=args.gap,
            threshold=args.threshold,
            blur_ksize=args.blur,
            morph_ksize=args.morph,
            smooth_blur_ksize=args.smooth_blur,
            smooth_mode=args.smooth_mode,
            contour_epsilon=args.contour_epsilon,
            contour_min_area=args.contour_min_area,
            save_compare=args.save_compare,
            nontrack_threshold=args.nontrack_threshold,
            nontrack_dilate=args.nontrack_dilate,
            track_erode=args.track_erode,
        strict_white=args.strict_white,
        track_value=args.track_value,
        track_tolerance=args.track_tolerance,
        preset=args.preset,
        post_min_area=args.post_min_area,
        post_keep_largest=args.post_keep_largest,
        invert_output=args.invert,
        keep_enclosed=args.keep_enclosed,
    )
    except SystemExit as exc:
        print(f"\n⚠️ Exiting: {exc}")
        raise
    
    if success:
        print("\n✅ Successfully processed map!")
    else:
        print("\n❌ Failed to process map!")
