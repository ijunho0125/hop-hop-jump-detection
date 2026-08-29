import os
import time
import threading
import urllib.request
from math import acos, degrees

import cv2
import numpy as np
import mediapipe as mp
import pyautogui


# ============================================================
# 기본 설정
# ============================================================

CAMERA_ID = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

MODEL_PATH = "pose_landmarker_lite.task"


# ============================================================
# 점프 설정
# ============================================================

# 작을수록 점프를 더 민감하게 감지
JUMP_SPEED_THRESHOLD = 0.16

# 몸이 기준점보다 최소 얼마나 올라가야 하는지
MIN_JUMP_RISE = 0.007

# 착지 판정 범위
LANDING_MARGIN = 0.012

# 같은 점프가 여러 번 입력되는 것 방지
JUMP_COOLDOWN = 0.16


# ============================================================
# 앉기 설정
# ============================================================

# 이 각도보다 무릎이 많이 접히면 앉기 후보
DUCK_KNEE_ANGLE = 145

# 다시 이 각도 이상 펴지면 일어서기 후보
DUCK_RELEASE_ANGLE = 160

# 몸 중심이 평상시보다 아래로 내려간 정도
DUCK_DROP_THRESHOLD = 0.035

# 다시 일어났다고 판단하는 범위
DUCK_RELEASE_DROP = 0.020

# 점프 직후 앉기로 잘못 인식하는 것 방지
AFTER_JUMP_DUCK_DELAY = 0.15


# ============================================================
# 기타
# ============================================================

CALIBRATION_TIME = 1.2

MIN_VISIBILITY = 0.45


# ============================================================
# MediaPipe 모델 다운로드
# ============================================================

def download_model():

    if os.path.exists(MODEL_PATH):
        return

    print("MediaPipe Pose Lite 모델 다운로드 중...")

    urls = [

        # 최신 모델
        (
            "https://storage.googleapis.com/"
            "mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/"
            "pose_landmarker_lite.task"
        ),

        # fallback
        (
            "https://storage.googleapis.com/"
            "mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/"
            "pose_landmarker_lite.task"
        )
    ]

    for url in urls:

        try:

            urllib.request.urlretrieve(
                url,
                MODEL_PATH
            )

            print("모델 다운로드 완료!")
            return

        except Exception:
            pass

    print()
    print("모델 자동 다운로드 실패")
    print(
        "pose_landmarker_lite.task 파일을 "
        "main.py와 같은 폴더에 넣어주세요."
    )

    raise RuntimeError(
        "Pose Landmarker 모델을 찾을 수 없습니다."
    )


download_model()


# ============================================================
# 키보드 입력 설정
# ============================================================

pyautogui.PAUSE = 0


# ============================================================
# 각도 계산
# ============================================================

def calculate_angle(a, b, c):
    """
    a = 골반
    b = 무릎
    c = 발목

    b를 기준으로 각도를 계산
    """

    ax = a.x * CAMERA_WIDTH
    ay = a.y * CAMERA_HEIGHT

    bx = b.x * CAMERA_WIDTH
    by = b.y * CAMERA_HEIGHT

    cx = c.x * CAMERA_WIDTH
    cy = c.y * CAMERA_HEIGHT


    vector1 = np.array([
        ax - bx,
        ay - by
    ])

    vector2 = np.array([
        cx - bx,
        cy - by
    ])


    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)

    if length1 == 0 or length2 == 0:
        return 180


    cosine = np.dot(
        vector1,
        vector2
    ) / (
        length1 * length2
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    angle = degrees(
        acos(cosine)
    )

    return angle


# ============================================================
# 동작 감지 클래스
# ============================================================

class MotionDetector:

    def __init__(self):

        self.lock = threading.Lock()

        self.latest_landmarks = None

        # 기준 몸 높이
        self.baseline_y = None

        # 이전 프레임
        self.previous_y = None
        self.previous_time = None

        # 현재 정보
        self.velocity = 0
        self.rise = 0
        self.drop = 0

        self.knee_angle = 180

        # 상태
        self.jump_active = False
        self.duck_active = False

        self.last_jump_time = 0

        self.jump_count = 0
        self.duck_count = 0

        self.status = "CALIBRATING"

        # 초기 보정
        self.calibration_samples = []
        self.calibration_start = time.perf_counter()


    # ========================================================
    # 기준 위치 다시 측정
    # ========================================================

    def reset_calibration(self):

        # 혹시 ↓가 눌린 상태라면 해제
        if self.duck_active:

            pyautogui.keyUp("down")

        with self.lock:

            self.baseline_y = None

            self.previous_y = None
            self.previous_time = None

            self.velocity = 0
            self.rise = 0
            self.drop = 0

            self.knee_angle = 180

            self.jump_active = False
            self.duck_active = False

            self.calibration_samples = []

            self.calibration_start = time.perf_counter()

            self.status = "CALIBRATING"

        print()
        print("재보정 중...")
        print("편하게 서 있으세요.")


    # ========================================================
    # 몸 중심 Y
    # ========================================================

    def get_body_y(self, landmarks):

        # 어깨 + 골반
        indexes = [
            11,
            12,
            23,
            24
        ]

        values = []

        for index in indexes:

            lm = landmarks[index]

            visibility = getattr(
                lm,
                "visibility",
                1
            )

            if visibility >= MIN_VISIBILITY:
                values.append(lm.y)

        if len(values) < 3:
            return None

        return sum(values) / len(values)


    # ========================================================
    # 무릎 각도
    # ========================================================

    def get_knee_angle(self, landmarks):

        angles = []


        # ----------------------------------------------------
        # 왼쪽 다리
        #
        # 23 골반
        # 25 무릎
        # 27 발목
        # ----------------------------------------------------

        left_indexes = [
            23,
            25,
            27
        ]

        if all(
            getattr(
                landmarks[i],
                "visibility",
                1
            ) >= MIN_VISIBILITY
            for i in left_indexes
        ):

            left_angle = calculate_angle(

                landmarks[23],
                landmarks[25],
                landmarks[27]

            )

            angles.append(left_angle)


        # ----------------------------------------------------
        # 오른쪽 다리
        #
        # 24 골반
        # 26 무릎
        # 28 발목
        # ----------------------------------------------------

        right_indexes = [
            24,
            26,
            28
        ]

        if all(
            getattr(
                landmarks[i],
                "visibility",
                1
            ) >= MIN_VISIBILITY
            for i in right_indexes
        ):

            right_angle = calculate_angle(

                landmarks[24],
                landmarks[26],
                landmarks[28]

            )

            angles.append(right_angle)


        if not angles:
            return 180


        return sum(angles) / len(angles)


    # ========================================================
    # MediaPipe 결과 처리
    # ========================================================

    def process(self, result):

        now = time.perf_counter()


        # 사람이 없음
        if not result.pose_landmarks:

            with self.lock:
                self.status = "NO PERSON"

            return


        landmarks = result.pose_landmarks[0]


        body_y = self.get_body_y(
            landmarks
        )


        if body_y is None:
            return


        knee_angle = self.get_knee_angle(
            landmarks
        )


        with self.lock:

            self.latest_landmarks = landmarks

            self.knee_angle = knee_angle


        # ====================================================
        # 초기 위치 보정
        # ====================================================

        if self.baseline_y is None:

            self.calibration_samples.append(
                body_y
            )

            elapsed = (
                now
                - self.calibration_start
            )


            if elapsed >= CALIBRATION_TIME:

                if len(
                    self.calibration_samples
                ) > 5:

                    self.baseline_y = float(
                        np.median(
                            self.calibration_samples
                        )
                    )

                    self.previous_y = body_y
                    self.previous_time = now

                    with self.lock:
                        self.status = "READY"

                    print()
                    print("==============================")
                    print("  준비 완료!")
                    print("==============================")
                    print()
                    print("점프        -> SPACE")
                    print("쪼그려 앉기 -> DOWN")
                    print()


            return


        # ====================================================
        # 속도 계산
        # ====================================================

        if self.previous_y is None:

            self.previous_y = body_y
            self.previous_time = now

            return


        dt = (
            now
            - self.previous_time
        )


        if dt <= 0:
            return


        # MediaPipe 화면 좌표는
        # 위로 갈수록 y 감소
        velocity = (
            self.previous_y
            - body_y
        ) / dt


        # 기준점보다 위로 올라간 양
        rise = (
            self.baseline_y
            - body_y
        )


        # 기준점보다 아래로 내려간 양
        drop = (
            body_y
            - self.baseline_y
        )


        self.velocity = velocity
        self.rise = rise
        self.drop = drop


        # ====================================================
        # 1. 점프 감지
        # ====================================================

        jump_ready = (

            not self.jump_active

            and not self.duck_active

            and (
                now
                - self.last_jump_time
            ) > JUMP_COOLDOWN

        )


        if (

            jump_ready

            and velocity
            > JUMP_SPEED_THRESHOLD

            and rise
            > MIN_JUMP_RISE

        ):

            pyautogui.press(
                "space"
            )


            self.jump_active = True

            self.last_jump_time = now

            self.jump_count += 1


            with self.lock:
                self.status = "JUMP!"


            print(
                f"JUMP #{self.jump_count} | "
                f"speed={velocity:.3f}"
            )


        # ====================================================
        # 2. 착지 감지
        # ====================================================

        if self.jump_active:

            if (

                body_y
                >= self.baseline_y
                - LANDING_MARGIN

                and (
                    now
                    - self.last_jump_time
                ) > 0.15

            ):

                self.jump_active = False

                with self.lock:
                    self.status = "READY"


        # ====================================================
        # 3. 앉기 감지
        # ====================================================

        can_duck = (

            not self.jump_active

            and (
                now
                - self.last_jump_time
            ) > AFTER_JUMP_DUCK_DELAY

        )


        # ----------------------------------------------------
        # 앉기 시작
        # ----------------------------------------------------

        if (
            can_duck
            and not self.duck_active
        ):

            # 무릎이 접히고
            # 몸도 아래로 내려갔을 때
            crouching = (

                knee_angle
                < DUCK_KNEE_ANGLE

                and drop
                > DUCK_DROP_THRESHOLD

            )


            # 무릎이 잠깐 안 잡히더라도
            # 몸이 아주 크게 내려갔다면 앉기로 판단
            deep_drop = (

                drop
                > DUCK_DROP_THRESHOLD * 1.8

            )


            if crouching or deep_drop:

                # ↓ 키를 누른 채 유지
                pyautogui.keyDown(
                    "down"
                )

                self.duck_active = True

                self.duck_count += 1


                with self.lock:
                    self.status = "DUCK!"


                print(
                    f"DUCK #{self.duck_count} | "
                    f"knee={knee_angle:.1f} | "
                    f"drop={drop:.3f}"
                )


        # ----------------------------------------------------
        # 다시 일어서기
        # ----------------------------------------------------

        elif self.duck_active:

            standing_again = (

                knee_angle
                > DUCK_RELEASE_ANGLE

                and drop
                < DUCK_RELEASE_DROP

            )


            # 높이가 거의 완전히 원래대로 돌아왔다면
            # 무릎 검출이 조금 불안정해도 해제
            height_recovered = (

                drop < 0.008

            )


            if (
                standing_again
                or height_recovered
            ):

                pyautogui.keyUp(
                    "down"
                )

                self.duck_active = False


                with self.lock:
                    self.status = "READY"


                print("STAND")


        # ====================================================
        # 4. 기준 위치 자동 보정
        # ====================================================

        if (

            not self.jump_active
            and not self.duck_active

            and abs(velocity) < 0.08

            and knee_angle > 165

        ):

            alpha = 0.003

            self.baseline_y = (

                self.baseline_y
                * (1 - alpha)

                +

                body_y
                * alpha

            )


        # ====================================================
        # 다음 프레임
        # ====================================================

        self.previous_y = body_y
        self.previous_time = now


# ============================================================
# Detector
# ============================================================

detector = MotionDetector()


# ============================================================
# MediaPipe callback
# ============================================================

def result_callback(
    result,
    output_image,
    timestamp_ms
):

    detector.process(
        result
    )


# ============================================================
# MediaPipe 설정
# ============================================================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=(
        RunningMode.LIVE_STREAM
    ),

    num_poses=1,

    min_pose_detection_confidence=0.4,

    min_pose_presence_confidence=0.4,

    min_tracking_confidence=0.4,

    output_segmentation_masks=False,

    result_callback=result_callback
)


landmarker = (
    PoseLandmarker.create_from_options(
        options
    )
)


# ============================================================
# 카메라
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID,
    cv2.CAP_DSHOW
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)


try:

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

except Exception:
    pass


if not cap.isOpened():

    print("카메라를 열 수 없습니다.")
    raise SystemExit


# ============================================================
# 스켈레톤
# ============================================================

POSE_CONNECTIONS = [

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32)

]


def draw_pose(
    frame,
    landmarks
):

    if landmarks is None:
        return


    height, width = (
        frame.shape[:2]
    )


    points = {}


    indexes = [

        11, 12,

        13, 14,

        15, 16,

        23, 24,

        25, 26,

        27, 28,

        29, 30,

        31, 32

    ]


    for index in indexes:

        landmark = landmarks[index]

        visibility = getattr(
            landmark,
            "visibility",
            1
        )


        if visibility < MIN_VISIBILITY:
            continue


        x = int(
            landmark.x
            * width
        )

        y = int(
            landmark.y
            * height
        )


        points[index] = (
            x,
            y
        )


    # 연결선
    for start, end in POSE_CONNECTIONS:

        if (
            start in points
            and end in points
        ):

            cv2.line(

                frame,

                points[start],
                points[end],

                (0, 255, 0),

                2

            )


    # 관절
    for point in points.values():

        cv2.circle(

            frame,

            point,

            5,

            (0, 255, 255),

            -1

        )


# ============================================================
# FPS
# ============================================================

fps = 0

frame_counter = 0

fps_timer = time.perf_counter()

start_timestamp = time.perf_counter()


# ============================================================
# 시작
# ============================================================

print()
print("==========================================")
print(" Chrome Dino Motion Controller")
print("==========================================")
print()
print("1. 카메라에 전신이 나오도록 서기")
print("2. 약 1초간 움직이지 않기")
print("3. Chrome Dino 창 클릭")
print()
print("점프          = SPACE")
print("쪼그려 앉기   = DOWN")
print()
print("Q = 종료")
print("R = 자세 재보정")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            continue


        # 거울 화면
        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # BGR -> RGB
        # ====================================================

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB

        )


        # cv2.ascontiguousarray가 아니라
        # numpy의 ascontiguousarray
        rgb = np.ascontiguousarray(
            rgb
        )


        # ====================================================
        # MediaPipe Image
        # ====================================================

        mp_image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb

        )


        # ====================================================
        # timestamp
        # ====================================================

        timestamp_ms = int(

            (
                time.perf_counter()
                - start_timestamp
            )

            * 1000

        )


        # ====================================================
        # 비동기 분석
        # ====================================================

        landmarker.detect_async(

            mp_image,

            timestamp_ms

        )


        # ====================================================
        # 최신 상태
        # ====================================================

        with detector.lock:

            landmarks = (
                detector.latest_landmarks
            )

            status = detector.status

            velocity = detector.velocity

            knee_angle = (
                detector.knee_angle
            )

            drop = detector.drop

            jump_count = (
                detector.jump_count
            )

            duck_count = (
                detector.duck_count
            )


        # ====================================================
        # 스켈레톤
        # ====================================================

        draw_pose(
            frame,
            landmarks
        )


        # ====================================================
        # FPS
        # ====================================================

        frame_counter += 1

        now = time.perf_counter()


        if (
            now
            - fps_timer
            >= 1
        ):

            fps = (

                frame_counter

                / (
                    now
                    - fps_timer
                )

            )

            frame_counter = 0
            fps_timer = now


        # ====================================================
        # 상태 색상
        # ====================================================

        if status == "JUMP!":

            color = (
                0,
                255,
                255
            )

        elif status == "DUCK!":

            color = (
                255,
                100,
                0
            )

        elif status == "READY":

            color = (
                0,
                255,
                0
            )

        else:

            color = (
                0,
                165,
                255
            )


        # ====================================================
        # UI
        # ====================================================

        cv2.putText(

            frame,

            f"STATUS: {status}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            color,

            2

        )


        cv2.putText(

            frame,

            f"FPS: {fps:.1f}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Jump: {jump_count}",

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Duck: {duck_count}",

            (20, 135),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Knee: {knee_angle:.1f}",

            (20, 165),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Up speed: {velocity:.3f}",

            (20, 195),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Drop: {drop:.3f}",

            (20, 225),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "JUMP = SPACE",

            (
                20,
                CAMERA_HEIGHT - 55
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            1

        )


        cv2.putText(

            frame,

            "DUCK = DOWN   Q: Quit   R: Reset",

            (
                20,
                CAMERA_HEIGHT - 25
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (255, 255, 255),

            1

        )


        # ====================================================
        # 화면
        # ====================================================

        cv2.imshow(

            "Chrome Dino Motion Controller",

            frame

        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("q"):
            break


        elif key == ord("r"):

            detector.reset_calibration()


# ============================================================
# 안전 종료
# ============================================================

finally:

    # 프로그램 종료 때
    # ↓ 키가 계속 눌려있지 않도록 반드시 해제
    pyautogui.keyUp(
        "down"
    )

    cap.release()

    landmarker.close()

    cv2.destroyAllWindows()


print("종료되었습니다.")