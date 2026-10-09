# คู่มือทำความเข้าใจโปรเจกต์นี้ทั้งหมด

เขียนสำหรับคนที่**เขียนโค้ดเป็น แต่ไม่เคยทำระบบ/deploy มาก่อน** จะปูพื้นเรื่องโครงสร้างระบบจากศูนย์ แต่ไม่ปูพื้นเรื่องไวยากรณ์ Python

อ่านเรียงตามลำดับได้เลย ส่วนที่ 1-2 คือแนวคิด ส่วนที่ 3-8 คือโค้ด ส่วนที่ 9-11 คือการ deploy กับบทเรียน

---

# 1. โปรเจกต์นี้ทำอะไร และทำไมต้องซับซ้อนขนาดนี้

## สิ่งที่อยากได้

มีโมเดล AI ชื่อ **LatentSync** รับวิดีโอหน้าคน + ไฟล์เสียง แล้วขยับปากในวิดีโอให้ตรงกับเสียง (lip-sync) อยากเรียกใช้ผ่าน API

## ทำไมไม่รันบนเครื่องตัวเองเฉยๆ

โมเดลนี้ต้องใช้ **GPU ที่มี VRAM ~18 GB** ซึ่งการ์ดจอทั่วไปไม่มี (การ์ดเกมระดับกลางมี 8-12 GB) ต้องไปเช่า GPU บนคลาวด์

## ทำไมไม่เช่าเซิร์ฟเวอร์ GPU ไว้ตลอด

GPU 24 GB เช่าชั่วโมงละ ~$0.69 ถ้าเปิดทิ้งไว้ 24 ชม. = **$497/เดือน** ทั้งที่อาจใช้จริงวันละไม่กี่นาที

## คำตอบคือ Serverless

จ่ายเฉพาะ**วินาทีที่รันจริง** ไม่มีงาน = ไม่จ่าย เครื่องดับไปเลย งานนี้เลยไปอยู่บน **RunPod Serverless**

## แต่เป้าหมายจริงของโปรเจกต์นี้ไม่ใช่ API

เป้าหมายคือตอบคำถามเดียว: **"ลิปซิงก์ 1 คลิปราคาเท่าไหร่จริงๆ"** เพื่อตัดสินใจว่าควรทำเองหรือจ่ายเจ้าอื่นทำ

เพราะงั้นโค้ดชุดนี้จึงออกแบบให้ **วัดราคา** เป็นหลัก ไม่ใช่ให้สมบูรณ์แบบในแง่ API ทุก response จะพ่วงเวลาที่ใช้ VRAM ที่กิน และราคาที่คำนวณแล้วกลับมาด้วย

> **ผลสรุปที่วัดได้จริง:** $0.0315 ต่อคลิป 9.6 วินาที (A5000, steps 10) ถูกกว่า WaveSpeedAI 1.59 เท่า รายละเอียดอยู่ใน `README.md`

---

# 2. แนวคิดพื้นฐาน 7 อย่างที่ต้องรู้ก่อนอ่านโค้ด

## 2.1 GPU กับ VRAM

GPU มีหน่วยความจำของตัวเองแยกจาก RAM เรียกว่า **VRAM** โมเดล AI ต้องถูกโหลดเข้า VRAM ทั้งก้อนถึงจะรันได้

- โมเดลนี้ไฟล์ `latentsync_unet.pt` ขนาด **4.8 GB**
- ตอนรันจริงกิน VRAM **~19 GB** (เพราะต้องเก็บผลลัพธ์ระหว่างทางด้วย)
- ถ้า VRAM ไม่พอ → error `CUDA out of memory` (เรียกสั้นๆ ว่า **OOM**)

## 2.2 fp16 กับ fp32

ตัวเลขทศนิยมในโมเดลเก็บได้ 2 ขนาด:

| | บิต | VRAM ที่ใช้ | ความเร็ว |
|---|---|---|---|
| fp32 | 32 | 2 เท่า | ช้ากว่า |
| **fp16** | 16 | 1 เท่า | เร็วกว่า |

โมเดลนี้ต้องใช้ fp16 ถึงจะพอใน 24 GB **แต่ GPU รุ่นเก่า (compute capability ≤ 7 เช่น T4, V100) ทำ fp16 ได้ไม่ดี** โค้ดต้นฉบับเลยสั่งให้ถอยไปใช้ fp32 อัตโนมัติ — ซึ่งแปลว่าจะ OOM ทันที

**นี่คือเหตุผลที่เอกสารย้ำว่าห้ามใช้ T4** ไม่ใช่เพราะ VRAM อย่างเดียว แต่เพราะมันจะถอยไป fp32 แล้วยิ่งไม่พอ

## 2.3 Container กับ Docker image

ปัญหาคลาสสิก: "โค้ดรันได้บนเครื่องฉัน แต่พังบนเซิร์ฟเวอร์" เพราะเวอร์ชัน Python ต่างกัน ไลบรารีขาด ฯลฯ

**Container** แก้ปัญหานี้ด้วยการแพ็ก **ทุกอย่าง** ไว้ด้วยกัน — ระบบปฏิบัติการ, Python, ไลบรารีทุกตัว, โค้ดเรา, แม้แต่ไฟล์โมเดล 4.8 GB

- **Dockerfile** = สูตรอาหาร (ไฟล์ข้อความบอกว่าต้องทำอะไรบ้าง)
- **Image** = อาหารที่ทำเสร็จแล้ว แช่แข็งไว้ (ของเราขนาด 27 GB)
- **Container** = เอาอาหารแช่แข็งมาอุ่นแล้วกิน (image ที่กำลังรันอยู่)

image เดียวกันเอาไปรันที่ไหนก็ได้ผลเหมือนกันเป๊ะ

## 2.4 Layer caching — ทำไมลำดับบรรทัดใน Dockerfile สำคัญ

Docker build ทีละบรรทัด แต่ละบรรทัดได้ **layer** หนึ่งชั้น ซ้อนกันขึ้นไป

ของสำคัญคือ: **ถ้าบรรทัดไหนไม่เปลี่ยน Docker จะใช้ layer เก่าซ้ำ (cache) ไม่ต้องทำใหม่** แต่พอมีบรรทัดไหนเปลี่ยน **บรรทัดนั้นและทุกบรรทัดถัดไปต้องทำใหม่หมด**

เพราะงั้น Dockerfile ที่ดีจะเรียงแบบนี้:

```
ของที่ไม่ค่อยเปลี่ยน  →  อยู่บน  (ติดตั้งไลบรารี, โหลดโมเดล)
ของที่เปลี่ยนบ่อย     →  อยู่ล่าง (โค้ดเรา)
```

ในโปรเจกต์นี้ `COPY handler.py` อยู่เกือบบรรทัดสุดท้าย **เพราะเราแก้ handler บ่อย** ถ้าเอาไว้ข้างบน ทุกครั้งที่แก้โค้ด 1 บรรทัดจะต้องโหลดโมเดล 4.8 GB ใหม่ทุกครั้ง

## 2.5 Serverless, Worker, Cold start

**Worker** = เครื่อง 1 เครื่องที่รัน container ของเรา

RunPod ทำงานแบบนี้:

```
ไม่มีงาน        → ไม่มี worker เลย → ไม่เสียเงิน
มีงานเข้ามา     → RunPod เปิด worker ใหม่ → เริ่มเสียเงิน
                  ├─ ดึง image 27 GB ลงเครื่อง (ครั้งแรกเท่านั้น, ช้ามาก)
                  ├─ เปิด container
                  ├─ โหลดโมเดลเข้า VRAM  ← "cold start"
                  └─ รันงาน
งานถัดมาทันที   → ใช้ worker เดิม โมเดลยังอยู่ใน VRAM ← "warm"
ว่างนานเกินกำหนด → RunPod ปิด worker → หยุดเสียเงิน
```

**cold start** = ค่าใช้จ่ายของการเริ่ม worker ใหม่ ซึ่งคุณต้องจ่ายด้วย

ในโปรเจกต์นี้ cold start = **17-24 วินาที** (แค่ 10% ของงานทั้งหมด) เพราะเรา **ฝังไฟล์โมเดลไว้ใน image แล้ว** ถ้าไม่ฝังต้องโหลดจากอินเทอร์เน็ตทุกครั้งที่เปิด worker = ช้าและแพงกว่านี้มาก

## 2.6 Queue vs Load balancer

RunPod ให้เลือก 2 แบบ:

| | ใช้เมื่อ | เรียกยังไง |
|---|---|---|
| **Queue** ← เราใช้อันนี้ | งานนานเป็นนาที | ส่งงาน → ได้ job id → คอยถามว่าเสร็จยัง |
| Load balancer | งานเร็วเป็นวินาที | ยิง HTTP แล้วรอคำตอบเลย |

งานเรานาน 160-330 วินาที ถ้าใช้แบบรอคำตอบเลย connection จะหลุดก่อน เลยต้องใช้ Queue

## 2.7 Environment variable (env var)

ค่าที่ตั้งจากข้างนอกโดยไม่ต้องแก้โค้ด เช่น `GPU_USD_PER_HOUR=1.10`

ใช้เพราะ**ค่าเดียวกันอาจต่างกันในแต่ละที่ที่รัน** และบางค่าเป็นความลับ (API key) ซึ่งห้ามเขียนลงโค้ดเด็ดขาด เพราะโค้ดขึ้น GitHub แล้วใครก็เห็น

---

# 3. ภาพรวมระบบ

```
  เครื่องคุณ                      RunPod Cloud
  ─────────                      ────────────

  bench.py
     │
     │  POST /run  {"input": {...}}
     ├──────────────────────────────►  คิวงาน
     │                                    │
     │  ◄── {"id": "abc123"}              │ เปิด worker (ถ้ายังไม่มี)
     │                                    ▼
     │                              ┌──────────────────────┐
     │                              │ Container ของเรา     │
     │                              │                      │
     │  GET /status/abc123          │  handler.py          │
     ├──────────────────────────────►    1. โหลด input     │
     │  ◄── {"status":"IN_PROGRESS"} │    2. แปลงไฟล์ ffmpeg│
     │                              │    3. รันโมเดล ← GPU │
     │       (ถามซ้ำทุก 3 วิ)        │    4. วัดเวลา/VRAM   │
     │                              │    5. ลบไฟล์ชั่วคราว │
     │  ◄── {"status":"COMPLETED",   │                      │
     │        "output": {...}}      └──────────────────────┘
     │
     ▼
  ตารางราคา + outputs/run-01.mp4
```

---

# 4. ไฟล์ทั้งหมดทำอะไรบ้าง

| ไฟล์ | บรรทัด | หน้าที่ | รันที่ไหน |
|---|---|---|---|
| `Dockerfile` | 101 | สูตรสร้าง image | ตอน build |
| `scripts/prefetch.py` | 317 | โหลดไฟล์โมเดลฝังใน image + ตรวจว่าโค้ดต้นฉบับยังเข้ากันได้ | ตอน build |
| `handler.py` | 681 | ตัวรับงานจริง | ใน container บน RunPod |
| `bench.py` | 424 | ยิงงานแล้วสรุปราคา | เครื่องคุณ |
| `requirements.txt` | - | ไลบรารีที่**เรา**เพิ่ม (runpod, requests) | ตอน build |
| `test_input.json` | - | input ตัวอย่างสำหรับทดสอบในเครื่อง | ตอนทดสอบ |
| `.dockerignore` | - | บอกว่าไฟล์ไหนไม่ต้องส่งเข้า build | ตอน build |
| `.gitattributes` | - | บังคับ line ending เป็น LF | ตอน git |
| `README.md` | 257 | วิธี deploy + ผลที่วัดได้ | - |

**ข้อสังเกต:** ไลบรารีของ LatentSync เอง เราไม่ได้เขียนไว้ใน `requirements.txt` ของเรา แต่ใช้ไฟล์ของ repo ต้นฉบับโดยตรง เพราะถ้าเขียนซ้ำแล้วเวอร์ชันไม่ตรงกันจะพังแบบหายาก

---

# 5. `Dockerfile` — ทีละส่วน

## 5.1 เลือก base image

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
```

เริ่มจาก Ubuntu ที่ติดตั้ง CUDA 12.1 มาให้แล้ว (CUDA = ชุดเครื่องมือที่ทำให้โปรแกรมคุยกับ GPU ได้)

**ทำไมไม่เลือกตัวที่มี cuDNN มาให้ด้วย** (มี tag `-cudnn8` ให้เลือก): เพราะไลบรารี `onnxruntime` ที่เราใช้ต้องการ **cuDNN 9** แต่ tag นั้นให้ cuDNN 8 มา ถ้าติดตั้งซ้อนกันเครื่องจะหยิบตัวเก่าไปใช้แล้ว**พังแบบเงียบๆ** — ไม่ error แต่ไปรันบน CPU ซึ่งช้ากว่ามาก

## 5.2 ปักหมุดเวอร์ชันโค้ดต้นฉบับ

```dockerfile
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
```

ไม่ใช้ `main` เพราะ `main` เปลี่ยนได้ตลอด ถ้า build วันนี้กับพรุ่งนี้ได้โค้ดคนละเวอร์ชัน **ตัวเลขที่วัดได้จะเทียบกันไม่ได้** การปักหมุดเป็นรหัส commit ทำให้ build กี่ครั้งก็ได้ของเดิมเป๊ะ

## 5.3 ติดตั้งของจากระบบปฏิบัติการ

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git ffmpeg libgl1 libglib2.0-0 \
        python3.10 python3.10-dev python3-pip build-essential \
    && rm -rf /var/lib/apt/lists/*
```

| ของ | ทำไมต้องมี |
|---|---|
| `ffmpeg` | ตัดต่อ/แปลงวิดีโอกับเสียง — ขาดไม่ได้เลย |
| `libgl1`, `libglib2.0-0` | ไลบรารีที่ OpenCV ต้องใช้ ถ้าขาดจะ error ตอน `import cv2` |
| `build-essential` | คอมไพเลอร์ C++ — ต้องมีเพราะ `insightface` ไม่มีไฟล์สำเร็จรูป ต้องคอมไพล์เอง |
| `git` | ไว้ clone โค้ดต้นฉบับ |

สองเทคนิคในบรรทัดนี้:

- **`&&` รวมทุกอย่างเป็น `RUN` เดียว** — ถ้าแยกเป็นหลาย `RUN` จะได้หลาย layer และขนาด image ใหญ่ขึ้น
- **`rm -rf /var/lib/apt/lists/*` ในบรรทัดเดียวกัน** — ลบแคชของ apt ทิ้ง ถ้าไปลบใน `RUN` ถัดไปจะไม่ช่วยเลย เพราะไฟล์ยังอยู่ใน layer ก่อนหน้า (layer ลบย้อนหลังไม่ได้)

## 5.4 ทำไมต้อง Python 3.10 เป๊ะๆ

Ubuntu 22.04 มี Python 3.10 มาให้อยู่แล้ว ซึ่งตรงกับที่ LatentSync ระบุ (3.10.13) พอดี

**ห้ามอัปเป็น 3.12** เพราะ `mediapipe==0.10.11` มีไฟล์สำเร็จรูปถึง Python 3.11 เท่านั้น ถ้าใช้ 3.12 จะต้องคอมไพล์เองซึ่งใช้เวลานานมากและพังง่าย

## 5.5 ติดตั้ง torch แยกก่อน

```dockerfile
RUN python3 -m pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu121
```

**ทำไมต้องแยกออกมาก่อน** ทั้งที่ `requirements.txt` ของ LatentSync ก็มี torch อยู่แล้ว:

torch มีหลายรุ่นตาม CUDA เวอร์ชัน (cu121, cu124, ...) ไฟล์ของ LatentSync ใช้คำสั่งที่**เปิดโอกาสให้ pip เลือกเองได้** ซึ่งอาจไปหยิบรุ่น cu124 มาทั้งที่ base image เราเป็น CUDA 12.1

ติดตั้งเองก่อนแบบล็อก index ชัดเจน พอถึงคิว `requirements.txt` pip จะเห็นว่ามีแล้วและข้ามไป

## 5.6 เตรียมของก่อนติดตั้ง insightface

```dockerfile
RUN python3 -m pip install "cython<3.0" "numpy==1.26.4"
RUN python3 -m pip install -r "${LATENTSYNC_ROOT}/requirements.txt"
```

`insightface` บน PyPI มีแต่ซอร์สโค้ด ไม่มีไฟล์สำเร็จรูป ตอนติดตั้งมันต้องคอมไพล์ และตัวคอมไพล์ต้องการ `cython` กับ `numpy` **มีอยู่ก่อนแล้ว** ถ้าไม่เตรียมไว้จะพังตอนนาทีที่ 10 ของ build

## 5.7 ทำให้ onnxruntime หา cuDNN เจอ

```dockerfile
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:...
```

torch ขนไลบรารี NVIDIA มาเองตอนติดตั้ง แต่มันซ่อนไว้ในที่ที่โปรแกรมอื่นหาไม่เจอ `LD_LIBRARY_PATH` คือรายชื่อโฟลเดอร์ที่ระบบจะไปค้นหาไลบรารี

ถ้าไม่ตั้ง `onnxruntime` จะหา cuDNN ไม่เจอ แล้ว **ถอยไปรันบน CPU โดยแค่เตือนเบาๆ ไม่ error** ผลคือการตรวจจับใบหน้าช้ามาก และตัวเลขราคาที่วัดได้จะผิด

(ใน `prefetch.py` ยังมีการตั้งค่าอีกวิธีซ้อนด้วย เผื่อ path ไม่ตรง)

## 5.8 ลำดับท้าย — จุดที่สำคัญมาก

```dockerfile
WORKDIR ${LATENTSYNC_ROOT}          # = /opt/latentsync
ENV PYTHONPATH=${LATENTSYNC_ROOT}

COPY scripts/prefetch.py ${APP_ROOT}/scripts/prefetch.py
RUN python3 "${APP_ROOT}/scripts/prefetch.py"     # โหลดโมเดล 5.2 GB

ENV HF_HUB_OFFLINE=1

COPY handler.py ${LATENTSYNC_ROOT}/handler.py     # โค้ดเราอยู่ล่างสุด
COPY test_input.json ${LATENTSYNC_ROOT}/test_input.json

CMD ["python3", "-u", "handler.py"]
```

**`WORKDIR` คือโฟลเดอร์ปัจจุบันตอนรัน และต้องเป็น `/opt/latentsync` เท่านั้น**

เพราะโค้ดต้นฉบับเขียน path แบบ**สัมพัทธ์** ไว้หลายที่:

```python
DDIMScheduler.from_pretrained("configs")          # ← ไม่มี / นำหน้า
FaceAnalysis(root="checkpoints/auxiliary")        # ← เหมือนกัน
mask_image_path: latentsync/utils/mask.png        # ← เหมือนกัน
```

path แบบนี้แปลว่า "หาจากโฟลเดอร์ปัจจุบัน" ถ้ารันจากที่อื่นจะหาไฟล์ไม่เจอทันที

**`ENV HF_HUB_OFFLINE=1` ต้องอยู่หลัง prefetch** — บรรทัดนี้สั่งห้ามต่ออินเทอร์เน็ตไปโหลดโมเดล ถ้าวางไว้ก่อน prefetch ก็จะโหลดอะไรไม่ได้เลย วางไว้หลังเพื่อการันตีว่าตอนรันจริงจะไม่มีการโหลดแอบแฝงมาทำให้เวลาที่วัดเพี้ยน

**`-u` ใน CMD** = ไม่ต้องพักข้อความไว้ใน buffer พิมพ์ออกมาเลย ถ้าไม่ใส่ log จะโผล่ช้าหรือหายตอน container ถูกปิด ซึ่งทำให้ debug ไม่ได้

---

# 6. `scripts/prefetch.py` — ตัวที่ทำให้ของพร้อมตั้งแต่ build

ไฟล์นี้รัน**ตอน build เท่านั้น** มี 2 หน้าที่

## 6.1 หน้าที่แรก: ฝังไฟล์โมเดลทุกตัวลง image

```python
def fetch_latentsync_weights():   # โมเดลหลัก 4.8 GB + whisper 73 MB
def fetch_buffalo_l():            # โมเดลตรวจจับใบหน้า 2 ไฟล์ 21 MB
def fetch_vae():                  # ตัวแปลงภาพ ~335 MB
```

**ทำไมต้องโหลดตอน build ไม่โหลดตอนใช้งาน**

ถ้าโหลดตอนมี request เข้ามา เวลาที่ใช้โหลด**ถูกคิดเป็นวินาที GPU** ซึ่งคุณจ่าย และจะทำให้ตัวเลขราคาที่เราพยายามวัดเพี้ยนไปหมด

กับดักที่เจอจริงในโปรเจกต์นี้: โค้ดต้นฉบับเรียกใช้ตัวตรวจจับใบหน้าแบบนี้

```python
FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"],
             root="checkpoints/auxiliary", ...)
```

ถ้าไม่มีไฟล์อยู่ มันจะ**แอบไปโหลดไฟล์ zip ขนาด 280 MB จากเซิร์ฟเวอร์ที่ล่มบ่อย** ตอน request แรก เราเลยโหลดมาฝังไว้ก่อน และฉลาดกว่านั้นคือ — ดูจาก `allowed_modules` จะเห็นว่าใช้จริงแค่ 2 โมเดล เลยโหลดแค่ 2 ไฟล์ (21 MB) แทนที่จะเอา zip ทั้งก้อน

```python
BUFFALO_MIRRORS = (
    ("public-data/insightface", "models/buffalo_l/{name}"),
    ("lithiumice/insightface", "models/buffalo_l/{name}"),
    ("KwaiVGI/LivePortrait", "insightface/models/buffalo_l/{name}"),
)
```

เตรียมแหล่งโหลดสำรอง 3 ที่ ถ้าที่แรกล่ม build จะไม่พัง

## 6.2 หน้าที่สอง: ตรวจว่าโค้ดต้นฉบับยังเข้ากันได้

นี่คือส่วนที่คนทำระบบมือใหม่มักไม่คิดถึง แต่สำคัญมาก

```python
expected = {
    LipsyncPipeline.__init__: {"vae", "audio_encoder", "unet", "scheduler"},
    LipsyncPipeline.__call__: {"video_path", "audio_path", ...},
    ...
}
for func, required in expected.items():
    missing = required - _named_params(func)
    if missing:
        fail(f"{func.__qualname__} no longer accepts {sorted(missing)}")
```

**ปัญหาที่ป้องกัน:** ฟังก์ชันหลักของ LatentSync ลงท้ายด้วย `**kwargs` ซึ่งแปลว่า**ถ้าเราส่งชื่อพารามิเตอร์ผิด มันจะเงียบ ไม่ error** แค่เมินค่านั้นทิ้ง

ถ้าวันหนึ่งต้นฉบับเปลี่ยนชื่อพารามิเตอร์ เราจะได้วิดีโอที่ดูเหมือนปกติแต่ความจริงรันด้วยค่า default ผิดๆ โดยไม่มีใครรู้ — และรู้ตัวตอนจ่ายเงินค่า GPU ไปแล้ว

โค้ดนี้เลยเช็คชื่อพารามิเตอร์ตอน build ถ้าไม่ตรง **build พังทันที** ดีกว่าไปพังตอนเสียเงิน

ยังเช็คอีกว่าโหลดโมเดลแบบตัดเน็ตได้จริง:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
AutoencoderKL.from_pretrained(VAE_REPO)   # ถ้าแคชไม่ครบ ตรงนี้พัง
```

---

# 7. `handler.py` — หัวใจของระบบ

681 บรรทัด แบ่งเป็น 5 ส่วน

## 7.1 ส่วนตั้งค่า — ทุกอย่างปรับได้โดยไม่ต้องแก้โค้ด

```python
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default

GPU_USD_PER_HOUR = _env_float("GPU_USD_PER_HOUR", 0.69)
MAX_AUDIO_SECONDS = _env_float("MAX_AUDIO_SECONDS", 60.0)
```

อ่านค่าจาก env var ถ้าไม่มีหรือค่าเสียก็ใช้ default **ไม่พังเพราะตั้งค่าผิด** — สำคัญมากกับงานที่รันอยู่ไกลๆ ที่คุณเข้าไปแก้ไม่ได้ทันที

`MAX_AUDIO_SECONDS` เป็นตัวกันเงินไหล — ถ้ามีคนส่งไฟล์เสียง 1 ชั่วโมงมา จะโดนปฏิเสธก่อนเริ่มรัน แทนที่จะรัน 5 ชั่วโมงแล้วเก็บเงินคุณ

## 7.2 โหลดโมเดลครั้งเดียว — เทคนิคสำคัญที่สุดของ serverless

```python
_pipeline = None
_pipeline_lock = threading.Lock()

def get_pipeline():
    global _pipeline, _pipeline_meta, _build_seconds, _ready_offset
    if _pipeline is None:                      # เช็คครั้งที่ 1 (เร็ว)
        with _pipeline_lock:                   # ล็อก
            if _pipeline is None:              # เช็คครั้งที่ 2 (กันซ้อน)
                _pipeline, _pipeline_meta = _build_pipeline()
                ...
    return _pipeline
```

**ทำไมต้องทำแบบนี้:** โหลดโมเดล 4.8 GB เข้า VRAM ใช้เวลา ~17 วินาที ถ้าโหลดใหม่ทุก request จะเสียเวลา (และเงิน) 17 วินาทีต่อครั้งฟรีๆ

เก็บไว้ในตัวแปร global แปลว่า **โหลดครั้งเดียวตอน worker เปิด แล้วใช้ซ้ำได้เรื่อยๆ** จนกว่า worker จะถูกปิด

**รูปแบบเช็คสองครั้ง (double-checked locking)** มีไว้กันกรณีมี 2 request เข้ามาพร้อมกันพอดีแล้วต่างคนต่างโหลดโมเดล — จะกิน VRAM 2 เท่าและ OOM

การ `import threading` แล้วใช้ `Lock()` คือเครื่องมือมาตรฐานสำหรับ "ให้ทำได้ทีละคน"

## 7.3 สร้าง pipeline — ลอกจากต้นฉบับ

```python
def _build_pipeline():
    has_cuda = torch.cuda.is_available()
    is_fp16_supported = has_cuda and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    if not has_cuda:
        log.error("no CUDA device visible to this process; ...")
    elif not is_fp16_supported:
        log.warning("%s has compute capability %s, so upstream falls back to fp32...")
```

นี่คือเรื่อง fp16 ในข้อ 2.2 เราไม่ได้แก้พฤติกรรมของต้นฉบับ แต่**เพิ่มคำเตือนดังๆ** เพราะถ้าไม่เตือนคุณจะงงว่าทำไมมันช้า/พัง

```python
    scheduler = DDIMScheduler.from_pretrained("configs")
    audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", ...)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    unet, _ = UNet3DConditionModel.from_pretrained(..., device="cpu")
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder,
                               unet=unet, scheduler=scheduler).to("cuda")
```

ส่วนนี้**ลอกมาจาก `scripts/inference.py` ของต้นฉบับแทบทุกบรรทัด** ตั้งใจให้เหมือนที่สุด เพราะถ้าเขียนเองอาจพลาดรายละเอียดเล็กๆ ที่ทำให้ผลลัพธ์ต่างไป

`.to("cuda")` = ย้ายโมเดลจาก RAM เข้า VRAM

```python
    if ENABLE_DEEPCACHE:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
```

**DeepCache** = เทคนิคเร่งความเร็ว โมเดลแบบนี้คำนวณซ้ำๆ หลายรอบ (20 รอบ) และผลบางส่วนแทบไม่เปลี่ยน DeepCache เก็บผลเก่ามาใช้ซ้ำทุกๆ 3 รอบ ค่า `3` กับ `0` ลอกจากต้นฉบับ

## 7.4 เตรียมไฟล์ด้วย ffmpeg

```python
def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-12:]
        raise MediaError(f"{cmd[0]} failed (exit {proc.returncode}):\n" + "\n".join(tail))
    return proc
```

ตัวช่วยเรียกโปรแกรมภายนอก (ffmpeg) จุดที่ตั้งใจ:

- **`timeout=`** — ถ้า ffmpeg ค้าง จะถูกฆ่าทิ้ง ไม่ปล่อยให้กิน GPU ไปเรื่อยๆ
- **เอา stderr 12 บรรทัดท้ายใส่ error** — ffmpeg พ่นข้อความเยอะมาก แต่สาเหตุจริงมักอยู่ท้ายสุด ถ้าไม่เก็บไว้จะได้แค่ "exit 1" ซึ่ง debug ไม่ได้
- **ส่งคำสั่งเป็น list ไม่ใช่ string** — ป้องกัน shell injection และไม่ต้องกังวลเรื่องช่องว่างในชื่อไฟล์

### แปลงไฟล์ให้โมเดลกินได้

```python
def _normalize_video(src, dest, audio_sec):
    video_sec = _probe_duration(src)
    cmd = ["ffmpeg", "-y"]
    if video_sec < audio_sec - 0.05:
        cmd += ["-stream_loop", "-1"]        # วิดีโอสั้นกว่าเสียง → วนซ้ำ
    cmd += ["-i", str(src),
            "-t", f"{audio_sec:.3f}",        # ตัดให้ยาวเท่าเสียง
            "-r", str(VIDEO_FPS),            # บังคับ 25 fps
            "-an",                           # ตัดเสียงเดิมทิ้ง
            "-vf", r"scale=w=trunc(min(1280\,iw)/2)*2:h=-2",
            ...]
```

**ทำไมต้องแปลงก่อน** ทั้งที่โมเดลอาจรับได้อยู่แล้ว: เพราะ**งานที่พังก็เสียเงินเหมือนกัน** ถ้าวิดีโอเป็น 30 fps หรือสั้นกว่าเสียง หรือขนาดภาพเป็นเลขคี่ แล้วโมเดลพังกลางทาง คุณจ่ายค่า GPU ไปแล้วโดยไม่ได้อะไร จัดการให้เรียบร้อยก่อนจึงคุ้มกว่า

- `trunc(.../2)*2` = ปัดให้เป็นเลขคู่ เพราะ codec วิดีโอส่วนใหญ่ต้องการขนาดหารด้วย 2 ลงตัว
- `h=-2` = ให้คำนวณความสูงเองตามสัดส่วน และปัดเป็นเลขคู่

มีฟังก์ชันพี่น้องอีก 2 ตัว: `_normalize_audio` (แปลงเป็น 16kHz mono ตามที่โมเดลต้องการ) และ `_image_to_video` (ถ้าส่งรูปนิ่งมา จะวนเป็นวิดีโอยาวเท่าเสียง)

## 7.5 ฟังก์ชัน `handler` — ตัวหลัก

RunPod จะเรียกฟังก์ชันนี้ทุกครั้งที่มีงาน รับ dict คืน dict

```python
def handler(job: dict) -> dict:
    request_start = time.monotonic()
    job_id = str(job.get("id", "local"))
    payload = job.get("input") or {}
    is_cold = _requests_served == 0
    work = WORK_DIR / f"{job_id}-{uuid.uuid4().hex[:8]}"
    try:
        ...
```

**`time.monotonic()` ไม่ใช่ `time.time()`** — `time.time()` คือเวลานาฬิกาซึ่งกระโดดได้ถ้าเครื่องซิงค์เวลา ทำให้วัดระยะเวลาได้ค่าติดลบ `monotonic()` เดินหน้าอย่างเดียว เหมาะกับการจับเวลา

**โฟลเดอร์แยกต่อ request** (`uuid` ต่อท้าย) — ถ้าใช้โฟลเดอร์เดียวกัน งาน 2 งานจะเขียนทับไฟล์กัน

### ลำดับงาน

```python
        pipeline = get_pipeline()                      # 1. โมเดล (โหลดแล้ว = เร็ว)
        _materialize(payload, ("audio_url", ...), raw_audio)   # 2. เอาไฟล์มา
        _normalize_audio(raw_audio, audio_path)        # 3. แปลง
        audio_sec = _probe_duration(audio_path)
        if audio_sec > MAX_AUDIO_SECONDS: raise MediaError(...)  # 4. กันเงินไหล
        _normalize_video(raw_visual, video_path, audio_sec)
        torch.cuda.reset_peak_memory_stats()           # 5. เริ่มนับ VRAM
        pipeline(video_path=..., audio_path=..., ...)  # 6. รันโมเดล ← ช้าสุด
```

แต่ละขั้นจับเวลาแยกเก็บใน `timings` เพื่อให้รู้ว่าเวลาหายไปไหน

### คืนค่าพร้อมข้อมูลวัดผล

```python
        return {
            "timings_sec": timings,
            "worker": {"cold_start": is_cold, "model_build_sec": ..., ...},
            "vram": _vram_report(),
            "cost": _cost_report(billable),
            "media": {"realtime_factor": round(timings["inference"] / audio_sec, 2), ...},
        }
```

`realtime_factor` = เวลารัน ÷ ความยาวเสียง **นี่คือตัวเลขที่มีค่าที่สุด** เพราะเอาไปคูณความยาวคลิปไหนก็ได้ ไม่ต้องทดสอบใหม่ทุกความยาว

### คิดเงินยังไงให้ถูก

```python
        uptime = time.monotonic() - _PROCESS_START
        billable = uptime if is_cold else total
```

- **request แรก (cold)** — RunPod คิดเงินตั้งแต่ worker เปิดเครื่อง ซึ่งรวมเวลา import Python และโหลดโมเดล เลยต้องใช้ `uptime` (เวลาตั้งแต่โปรแกรมเริ่ม)
- **request ถัดไป (warm)** — worker เปิดอยู่แล้ว คิดแค่เวลาของงานนี้

เคยเขียนผิดเป็น `init + total` ซึ่งนับซ้ำและเพี้ยนถ้าโมเดลโหลดช้า แก้เป็นแบบนี้แล้วตรงกับที่ RunPod คิดจริง

### จัดการ error แยกประเภท

```python
    except MediaError as exc:
        return {"error": str(exc), "error_type": "bad_input"}

    except torch.cuda.OutOfMemoryError as exc:
        return {
            "refresh_worker": True,
            "error": "CUDA out of memory. LatentSync 1.6 needs ~18 GB in fp16...",
            "error_type": "cuda_oom",
            "vram": _vram_report(),
        }

    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}",
                "error_type": "inference_error",
                "traceback": traceback.format_exc()[-2000:]}
```

แยก 3 ประเภทเพราะ**แต่ละแบบต้องแก้คนละทาง**:

| ประเภท | สาเหตุ | ทำยังไง |
|---|---|---|
| `bad_input` | ผู้ใช้ส่งไฟล์เสีย/ยาวเกิน | แก้ที่ input |
| `cuda_oom` | VRAM ไม่พอ | เปลี่ยน GPU |
| `inference_error` | อย่างอื่น | ดู traceback |

**`"refresh_worker": True`** เป็นคำสั่งพิเศษของ RunPod = "ทิ้ง worker ตัวนี้แล้วเปิดใหม่" ใช้ตอน OOM เพราะหน่วยความจำกระจัดกระจายแล้ว งานถัดไปบน worker เดิมก็จะพังอีก **และแต่ละครั้งที่พังคุณก็ยังจ่ายเงิน**

### `finally` — ส่วนที่ขาดไม่ได้

```python
    finally:
        _requests_served += 1
        shutil.rmtree(work, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

`finally` ทำงาน**เสมอ** ไม่ว่าจะสำเร็จหรือพัง

- **ลบโฟลเดอร์ชั่วคราว** — พื้นที่ดิสก์ของ container มีจำกัด (เราตั้ง 10 GB) ถ้าไม่ลบ ทุก request จะทิ้งไฟล์ไว้หลายร้อย MB แล้วอีกไม่กี่สิบงานดิสก์เต็ม → **ทุกงานหลังจากนั้นพังหมด** และเป็นบั๊กที่หาสาเหตุยากมากเพราะตอนแรกมันทำงานได้ปกติ
- **`empty_cache()`** — คืน VRAM ที่ PyTorch จองไว้ ลดโอกาส OOM ในงานถัดไป

## 7.6 ตอนเปิด worker

```python
def _startup() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for leftover in WORK_DIR.iterdir():
        shutil.rmtree(leftover, ignore_errors=True)   # เก็บขยะจาก worker ที่ถูกฆ่า
    _check_onnxruntime_providers()
    if EAGER_LOAD:
        try:
            get_pipeline()
        except Exception:
            log.error("eager model load failed:\n%s", traceback.format_exc())

_startup()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "concurrency_modifier": _concurrency_modifier})
```

**โหลดโมเดลทันทีตอนเปิด (`EAGER_LOAD`)** แทนที่จะรอ request แรก — ได้เวลา cold start ที่วัดได้ชัดเจน

**`try/except` ครอบไว้** — ถ้าโหลดไม่ได้ ไม่ให้ container ตายทันที ปล่อยให้ request แรกเข้ามาแล้วรายงาน error ที่อ่านรู้เรื่องกลับไป ดีกว่า container ดับเฉยๆ โดยไม่มีใครรู้สาเหตุ

**`concurrency_modifier` คืน 1 เสมอ** = worker 1 ตัวรับงานทีละ 1 งาน ถ้ารับพร้อมกัน 2 งานจะแย่ง VRAM กัน (19 GB × 2 = OOM แน่นอน) และเวลาที่วัดได้จะเพี้ยน

**`runpod.serverless.start(...)` เขียนบรรทัดเดียว** — ไม่ใช่เรื่องความสวยงาม แต่เพราะระบบของ RunPod สแกนหาข้อความ `runpod.serverless.start({` ตรงๆ ถ้าเขียนแยกบรรทัดมันจะหาไม่เจอแล้วขึ้นคำเตือนว่า "ไม่พบ handler"

---

# 8. `bench.py` — ตัววัดราคา

รันบนเครื่องคุณ ไม่ได้อยู่ใน container

## 8.1 ส่งงานแล้วคอยถาม

```python
def submit(session, endpoint, payload) -> str:
    resp = session.post(f"{API_BASE}/{endpoint}/run", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["id"]

def wait(session, endpoint, job_id, timeout) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        body = session.get(f"{API_BASE}/{endpoint}/status/{job_id}", timeout=60).json()
        if body.get("status") in TERMINAL:
            return body
        if time.monotonic() > deadline:
            raise TimeoutError(...)
        time.sleep(POLL_INTERVAL_SEC)
```

นี่คือรูปแบบ **polling** — ส่งงานได้ job id กลับมา แล้ววนถามทุก 3 วินาทีว่าเสร็จยัง เป็นวิธีมาตรฐานของงานที่ใช้เวลานาน

## 8.2 บทเรียนราคาแพง: อย่าเชื่อว่าได้การ์ดที่สั่ง

```python
GPU_HOURLY = (
    ("h100", 4.79), ("a100", 2.72),
    ("l40s", 1.75), ("l40", 1.75),      # ← ต้องมาก่อน "l4"
    ("a6000", 1.22), ("4090", 1.10),
    ("a5000", 0.69), ("3090", 0.69), ("l4", 0.69),
)

def resolve_rate(gpu_name, explicit):
    ...
    if explicit is not None and abs(explicit - detected) > 0.001:
        print("  !! You passed --usd-per-hour ... but RunPod ran this on a ...")
        return detected, "..."
```

**เรื่องจริงที่เกิดขึ้น:** ตั้ง endpoint ให้ใช้ RTX 4090 ($1.10/ชม) แต่ RunPod จับ **A5000** ($0.69/ชม) มาให้ เพราะมันให้การ์ดตัวไหนก็ได้ที่ว่างในกลุ่มนั้น ถ้าคำนวณด้วยเรตที่เราสั่ง ราคาจะ**ผิดสูงไป 59%**

แก้โดยให้อ่านชื่อ GPU ที่ worker รายงานกลับมาจริงๆ แล้วเปิดตารางเรตเอง

**`("l40", ...)` ต้องอยู่ก่อน `("l4", ...)`** เพราะเช็คด้วยการหาข้อความย่อย ถ้าสลับกัน "NVIDIA L40S" จะไปแมตช์ "l4" แล้วคิดราคาผิด

## 8.3 เปิดเผยเวลาที่ถูกคิดเงินแต่มองไม่เห็น

```python
gaps = [billed_of(r) - r["total_sec"] for r in basis if ...]
gap = statistics.median(gaps)
print(f"  Platform overhead   {gap:.1f}s billed outside the handler")
```

`timings_sec.total` วัดจากใน handler จบตอน `return` แต่ RunPod คิดเงินจาก `executionTime` ซึ่งยาวกว่า เพราะหลัง handler คืนค่าแล้ว worker ยังต้องแพ็กผลลัพธ์แล้วอัปโหลด — **เวลานั้นก็ถูกคิดเงินแต่เรามองไม่เห็น**

ส่วนต่างของสองค่านี้เปิดเผยมันออกมา (วัดได้จริง 0.2-0.8 วินาที = ~$0.0002 น้อยมาก)

---

# 9. การ Deploy — ทำอะไรบ้างและทำไม

## เส้นทางที่เลือก: ให้ RunPod build เอง

```
โค้ดในเครื่อง  →  GitHub  →  RunPod build image เอง  →  Endpoint
   (45 KB)       (45 KB)         (27 GB อยู่ฝั่งเขา)
```

**ทำไมไม่ build ในเครื่องแล้ว push ขึ้น registry:** image ขนาด 27 GB ถ้าเน็ตบ้าน upload ช้า จะใช้เวลาหลายชั่วโมง ส่ง**โค้ด 45 KB** ขึ้น GitHub แล้วให้ RunPod build บนเครื่องเขาเร็วกว่ามาก

> แต่เรายัง build ในเครื่องก่อนอยู่ดี — เพื่อ**ตรวจว่า Dockerfile ใช้ได้จริง** ก่อนเสียเวลา build ฝั่ง RunPod การ build ครั้งแรกเจอบั๊กจริง 1 ตัว (`PYTHONPATH` ตั้งผิดตำแหน่ง) ถ้าไม่ลอง build ก่อนก็จะไปเจอตอน deploy

## ขั้นตอน

1. **เชื่อม GitHub กับ RunPod** — Settings → GitHub → Connect
2. **สร้าง Endpoint** — Serverless → New Endpoint → Import Git Repository → เลือก repo + branch
3. **ตั้งค่า** (ดูตารางข้างล่าง)
4. **RunPod build ให้** ~30-40 นาที

## ค่าที่ต้องตั้ง และเหตุผล

| ช่อง | ค่า | ถ้าตั้งผิดเกิดอะไร |
|---|---|---|
| Type | **Queue** | เลือก Load balancer แล้ว endpoint จะไม่ตอบ `/run` |
| **Active Workers** | **0** | ถ้า > 0 คือเปิดเครื่องทิ้งไว้ 24 ชม. **จ่ายตลอดแม้ไม่มีงาน** |
| **Max Workers** | **1** | ถ้ามากกว่านี้ อาจเปิดหลายเครื่องพร้อมกัน จ่าย cold start ซ้ำ |
| **Idle Timeout** | **5 วินาที** | worker **ยังคิดเงินตอนนอนรอ** ตั้ง 60 วิ = จ่ายเพิ่ม 60 วิ ทุกงาน |
| GPU | 24 GB | ต่ำกว่านี้ OOM / T4 จะถอยไป fp32 แล้วยิ่งพัง |
| Execution Timeout | 1800 | ต่ำเกินงานจะถูกตัดกลางคัน **แต่ยังจ่ายเงินส่วนที่รันไปแล้ว** |
| Container Disk | 10 GB | พื้นที่ `/tmp` สำหรับไฟล์ชั่วคราว |

**สามช่องตัวหนาคือช่องที่ทำเงินไหลเงียบๆ** มือใหม่พลาดกันบ่อยที่สุด

## ทดสอบ

```powershell
$env:RUNPOD_API_KEY="<key>"; $env:RUNPOD_ENDPOINT_ID="<id>"
python bench.py --n 1 --save-video --timeout 2400
```

request แรกช้า 5-15 นาที เพราะ worker ต้องดึง image 27 GB **ให้ตัดรอบแรกทิ้งจากการวัด**

## ปิดเมื่อเสร็จ

ตั้ง Max Workers = 0 หรือลบ endpoint ทิ้ง

---

# 10. บทเรียนจริงจากโปรเจกต์นี้

เรื่องพวกนี้สำคัญกว่าโค้ด เพราะเป็นสิ่งที่อ่านจากหนังสือไม่ได้

## 10.1 การประมาณการมีค่าเท่ากับศูนย์

ประมาณไว้ว่า inference ใช้ 60-90 วินาที **ของจริง 324 วินาที — ผิด 4.3 เท่า** และราคาที่ประมาณไว้ $0.015 ของจริง $0.064

ที่สำคัญกว่าคือ**มันพลิกข้อสรุป** จาก "ทำเองคุ้มกว่ามาก" เป็น "ทำเองแพงกว่าซื้อ" ถ้าตัดสินใจบนตัวเลขประมาณการจะตัดสินใจผิด

> **บทเรียน:** ถ้าตัวเลขไหนเป็นตัวตัดสินใจ ต้องวัด ไม่ใช่ประมาณ และต้องรู้ว่าตัวเลขไหนที่ทั้งเรื่องแขวนอยู่กับมัน

## 10.2 เลือก GPU ผิดเพราะ optimize ผิดตัวแปร

เลือก L4 เพราะ **ถูกสุดต่อชั่วโมง** ในกลุ่ม 24 GB แต่สิ่งที่ต้อง optimize คือ:

```
ราคาต่อคลิป = เวลา × เรตต่อชั่วโมง
```

L4 เป็นการ์ดประหยัดไฟที่ช้า **A5000 เร็วกว่า 1.4 เท่าที่เรตเท่ากันเป๊ะ** → ถูกลง 31% ฟรีๆ

> **บทเรียน:** ดูว่าสิ่งที่คุณ optimize คือสิ่งที่คุณจ่ายจริงหรือเปล่า

## 10.3 ของที่พังเงียบอันตรายกว่าของที่พังดัง

สามอย่างในโปรเจกต์นี้พังแบบไม่มี error:

| สิ่งที่พังเงียบ | ผลที่ตามมา | ที่ป้องกันไว้ |
|---|---|---|
| `onnxruntime` หา cuDNN ไม่เจอ → รันบน CPU | ช้ามาก ราคาเพี้ยน | ตั้ง `LD_LIBRARY_PATH` + เช็คตอน startup |
| `**kwargs` กลืนชื่อพารามิเตอร์ที่พิมพ์ผิด | ได้ผลลัพธ์ผิดโดยดูปกติ | เช็คลายเซ็นฟังก์ชันตอน build |
| GPU เก่าถอยไป fp32 | OOM หรือช้ามาก | เตือนดังๆ ตอน startup |

> **บทเรียน:** เวลาทำระบบ ให้ถามว่า "ถ้าอันนี้พัง ฉันจะรู้ไหม" ถ้าคำตอบคือไม่ ต้องเพิ่มการตรวจ

## 10.4 เครื่องมือวัดเองก็มีบั๊กได้

`bench.py` มีบั๊ก 3 ตัวที่ทำให้อ่านผลผิด:

- `"return_video": not args.download_video` — ตรรกะกลับด้าน
- คิดราคาด้วยเรตการ์ดที่สั่ง ไม่ใช่การ์ดที่ได้จริง (ผิด 59%)
- พิมพ์ `"0.8x more"` ทั้งที่ความจริง**ถูกกว่า**

> **บทเรียน:** ถ้าตัวเลขออกมาแปลก ให้สงสัยเครื่องมือวัดก่อนสรุป

## 10.5 build ในเครื่องก่อนเสมอ

static check ผ่านหมด แต่ build จริงเจอบั๊กทันที (`PYTHONPATH` ตั้งผิดตำแหน่ง ทำให้ `import latentsync` ไม่เจอ)

> **บทเรียน:** โค้ดที่ยังไม่เคยรัน คือโค้ดที่ยังไม่รู้ว่าใช้ได้

---

# 11. ถ้าจะเอาไปใช้งานจริง ต้องเพิ่มอะไร

โปรเจกต์นี้ตัดของสำหรับ production ออกตั้งใจ เพราะเป้าหมายคือวัดราคา เรียงตามความสำคัญ:

## 11.1 S3 — จำเป็นที่สุด

ตอนนี้ส่งวิดีโอกลับเป็น **base64** (แปลงไฟล์เป็นข้อความ) ซึ่งมีปัญหา 3 อย่าง:

1. ขนาดโตขึ้น 33% ฟรีๆ
2. RunPod จำกัด response ที่ 10 MB → **คลิปเกิน ~20 วินาทีส่งไม่ได้เลย**
3. เวลาอัปโหลดถูกคิดเป็นวินาที GPU — ใช้การ์ด $0.69/ชม. ไปทำหน้าที่เซิร์ฟเวอร์ไฟล์

**ทางแก้:** ให้ worker อัปขึ้น S3/Cloudflare R2 เองแล้วคืน URL กลับมา ฝั่งผู้ใช้ดาวน์โหลดจาก S3 ซึ่ง RunPod ไม่คิดเงิน (~60 บรรทัด)

## 11.2 ตรวจสอบสิทธิ์ (auth)

ตอนนี้ใครมี endpoint ID + API key ก็เรียกได้ และ API key ตัวเดียวทำได้ทุกอย่าง ควรมีระบบ key แยกต่อผู้ใช้

## 11.3 กัน SSRF

`_download()` ยอมโหลดจาก URL ไหนก็ได้ ถ้าเปิดให้คนนอกใช้ เขาอาจใส่ URL ที่ชี้เข้าเครือข่ายภายใน ควรบล็อก IP ส่วนตัว (`127.0.0.1`, `10.x`, `192.168.x`, `169.254.169.254`)

## 11.4 Retry + webhook

งานใช้เวลา 3-6 นาที การให้ client นั่ง poll ไม่เหมาะกับงานจริง RunPod รองรับ webhook อยู่แล้ว

## 11.5 ลดขนาด image

27 GB ใหญ่เกินจำเป็น ของที่ไม่ได้ใช้: `gradio` (162 MB), `jaxlib` (317 MB ลากมาโดย mediapipe ที่ไม่ได้ใช้เลย), opencv ติดตั้งซ้อน 3 ตัว (336 MB) ลดเหลือ ~13-15 GB ได้ → build เร็วขึ้น เปิด worker ใหม่เร็วขึ้น

## 11.6 กันการ์ดช้า

กลุ่ม $0.69/ชม มีทั้ง L4, A5000, 3090 และ **L4 ช้ากว่า A5000 1.4 เท่าที่ราคาเท่ากัน** ราคาจริงเลยแกว่ง $0.0315-0.045 ควรวัดว่าได้การ์ดอะไรบ้างจริง แล้วตัดสินใจว่าจะยอมรับความแกว่งหรือจ่ายแพงขึ้นเพื่อล็อกรุ่น

---

# 12. คำศัพท์

| คำ | ความหมาย |
|---|---|
| **VRAM** | หน่วยความจำของ GPU โมเดลต้องอยู่ในนี้ถึงรันได้ |
| **OOM** | Out of memory — VRAM ไม่พอ |
| **fp16 / fp32** | ความละเอียดตัวเลขทศนิยม fp16 กินครึ่งเดียวและเร็วกว่า |
| **CUDA** | ชุดเครื่องมือของ NVIDIA ให้โปรแกรมคุยกับ GPU |
| **cuDNN** | ไลบรารีเร่งความเร็ว deep learning ต่อยอดจาก CUDA |
| **Container** | กล่องที่บรรจุ OS + ไลบรารี + โค้ด ไว้ด้วยกัน |
| **Image** | container ที่แพ็กเสร็จแล้ว พร้อมเอาไปรัน |
| **Layer** | ชั้นของ image แต่ละคำสั่งใน Dockerfile ได้ 1 ชั้น |
| **Registry** | ที่เก็บ image (เช่น Docker Hub) |
| **Worker** | เครื่อง 1 เครื่องที่รัน container ของเรา |
| **Cold start** | เวลา (และเงิน) ที่เสียไปกับการเปิด worker ใหม่ |
| **Warm** | worker ที่เปิดอยู่แล้ว โมเดลพร้อมใน VRAM |
| **Polling** | วนถามซ้ำๆ ว่าเสร็จหรือยัง |
| **Idle timeout** | เวลาที่ worker นอนรอก่อนถูกปิด — **ยังคิดเงินอยู่** |
| **env var** | ค่าที่ตั้งจากข้างนอกโดยไม่ต้องแก้โค้ด |
| **RTF** | realtime factor = เวลารัน ÷ ความยาวคลิป |
| **Inference** | การเอาโมเดลที่เทรนแล้วมาใช้งาน (ตรงข้ามกับ training) |
| **Checkpoint / weights** | ไฟล์โมเดลที่เทรนเสร็จแล้ว |
| **SSRF** | ช่องโหว่ที่หลอกให้เซิร์ฟเวอร์ยิง request เข้าเครือข่ายภายใน |

---

# อ่านต่อ

- `README.md` — วิธี deploy และผลที่วัดได้ทั้งหมด
- โค้ดทุกไฟล์มีคอมเมนต์อธิบาย**เหตุผล** ไม่ใช่แค่บอกว่าทำอะไร — ส่วนที่งงให้อ่านคอมเมนต์เหนือบรรทัดนั้น
- ต้นฉบับ LatentSync: https://github.com/bytedance/LatentSync (ปักหมุดที่ `a229c394`)
