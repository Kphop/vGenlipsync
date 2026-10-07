# LatentSync 1.6 บน RunPod Serverless — ชุดวัดราคา

เป้าหมายของ repo นี้คือตอบคำถามเดียว: **ลิปซิงก์ 1 คลิปบน RunPod ราคาเท่าไหร่จริงๆ**

ไม่ใช่ production API — ตัด S3, auth, webhook, retry ออกทั้งหมด เหลือแค่สิ่งที่จำเป็นต่อการวัดราคาให้แม่น ถ้าราคาผ่านแล้วค่อยกลับมาทำรอบ production

```
ราคา/คลิป = (worker init + เวลาประมวลผล) วินาที x $0.000192/วินาที
```

ตัวแปรเดียวที่ยังไม่รู้คือ **เวลาประมวลผลจริง** ซึ่ง endpoint นี้จะวัดและคืนมาให้ในทุก response

---

## 1. Build และ push

ต้องใส่ `--platform linux/amd64` เพราะ build จาก Windows (ถ้าเครื่องเป็น ARM จะได้ image ที่ RunPod รันไม่ได้)

```powershell
# ตั้งชื่อ image ของคุณ
$IMAGE = "docker.io/<your-dockerhub-user>/latentsync-runpod:1.6"

docker buildx build --platform linux/amd64 -t $IMAGE --load .
docker push $IMAGE
```

Build ใช้เวลา ~20-40 นาที และได้ image ขนาด **~17 GB** (torch+CUDA ~6GB, weights ~5.6GB, deps ~2GB, base ~2.5GB)

**ถ้า build ล้มเหลว** `scripts/prefetch.py` จะ fail ทันทีพร้อมบอกว่าอะไรพัง — ตั้งใจให้ fail ที่ build แทนที่จะไปพังตอนรันบน GPU ที่เสียเงิน

---

## 2. สร้าง Serverless Endpoint

RunPod Console → Serverless → New Endpoint → Custom Source → Docker image

ตั้งค่าตามนี้ ช่องที่ **ตัวหนา** คือช่องที่ทำเงินไหลทิ้งถ้าตั้งผิด:

| ช่อง | ค่า | เหตุผล |
|---|---|---|
| Container Image | `<your image>` | |
| GPU | **24 GB** (L4 / A5000 / RTX 3090) | ถูกสุดที่ VRAM พอ — $0.69/ชม |
| **Active Workers** | **0** | active = จ่าย 24 ชม. ต่อให้ไม่มี request |
| **Max Workers** | **1** | กัน cold start หลายตัวพร้อมกันตอนทดสอบ |
| **Idle Timeout** | **5 วินาที** | worker ยัง bill ตอน idle — ช่องนี้กินเงินเงียบที่สุด |
| FlashBoot | เปิด | ลด cold start ซึ่งเป็นก้อนราคาใหญ่สุดตอน volume ต่ำ |
| Execution Timeout | 600 วินาที | กัน job ค้างกินเงิน |
| Container Disk | 10 GB | `/tmp` + temp frames ของ pipeline |

### ห้ามเลือก GPU 16 GB หรือ T4

สองเหตุผล: 1.6 ต้องการ VRAM ~18 GB และ upstream เปิด fp16 เฉพาะเมื่อ compute capability > 7 — T4 คือ 7.5 เลยตกไปใช้ fp32 ซึ่งยิ่งไม่พอ L4 / A5000 / RTX 3090 (8.6-8.9) ใช้ได้หมด ถ้าเลือกผิด handler จะ log warning ตัวโตให้เห็น

---

## 3. วัดราคา

```powershell
$env:RUNPOD_API_KEY = "<your-api-key>"
$env:RUNPOD_ENDPOINT_ID = "<your-endpoint-id>"

python -m pip install requests
python bench.py --n 5
```

ค่า default ใช้ไฟล์ demo ที่ bake อยู่ใน image แล้ว (`assets/demo1_video.mp4`) เลย**ไม่ต้องไป host ไฟล์ที่ไหน** และไม่มี noise จากการดาวน์โหลดมาปนในตัวเลขเวลา

ผลที่ได้จะเป็นแบบนี้:

```
Run  Cold  Inference     Total   RunPod exec       RTF    PeakVRAM      $/clip
  1   yes      68.40s    72.60s       113.90s     6.84x     17.80GB   $0.0218
  2    no      66.10s    69.00s        69.40s     6.61x     17.90GB   $0.0133

Summary  (n=5 ok, 0 failed)
  GPU                 NVIDIA L4  @ $0.69/hr
  Billed per clip     median 69.4s  (warm)
  Cost per clip       $0.0133
  Realtime factor     6.61x  ->  1 min of audio ~ 397s ~ $0.0761
  Peak VRAM           17.9 GB of 23.0 GB  (this tier is right-sized)

Monthly projection  (warm, 10.0s clips)
     100 clips/mo    $      1.33
   1,000 clips/mo    $     13.30
  10,000 clips/mo    $    133.00
```

### ตัวเลขที่ต้องดู

| ตัวเลข | ใช้ตอบว่า |
|---|---|
| `Cost per clip` (warm) | ราคาจริงต่อคลิป |
| `Realtime factor` | คูณความยาวคลิปได้เลย — ใช้ project คลิปความยาวอื่นโดยไม่ต้องรันใหม่ |
| `Peak VRAM` | ถ้าเหลือ headroom > 6 GB แปลว่าอาจลง GPU tier ถูกกว่าได้ ลองเลย |
| `Cold start` | ส่วนต่างราคา cold vs warm — ตัวนี้กำหนดว่า volume ต่ำจะคุ้มไหม |

### ทดสอบ cold start แยก

```powershell
python bench.py --n 3 --sleep 120   # รอเกิน idle timeout ให้ worker ตายก่อนยิงใหม่
```

### ลองความยาว / steps อื่น

```powershell
python bench.py --n 3 --steps 30                                  # steps เยอะขึ้น = ช้าลง = แพงขึ้น
python bench.py --n 3 --video-url https://... --audio-url https://...   # ไฟล์ของคุณเอง
python bench.py --n 3 --out results.json                          # เก็บ raw ไว้ดูเอง
```

---

## 4. ใช้งานตรงๆ (ถ้าไม่ผ่าน bench.py)

```bash
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"video_url":"https://.../face.mp4","audio_url":"https://.../speech.wav"}}'
```

**input ที่รับได้** — เลือกอย่างใดอย่างหนึ่งต่อประเภท:

| key | หมายเหตุ |
|---|---|
| `video_url` / `video_base64` / `video_path` | `_path` คือไฟล์ใน image ใช้ตอน benchmark |
| `image_url` / `image_base64` / `image_path` | ภาพนิ่ง จะถูก loop เป็นวิดีโอ 25fps ยาวเท่าเสียงอัตโนมัติ |
| `audio_url` / `audio_base64` / `audio_path` | **จำเป็น** |
| `inference_steps` | default 20 |
| `guidance_scale` | default 1.5 |
| `return_video` | default `true` ตั้ง `false` เพื่อวัดราคาโดยไม่เอา payload |

ใช้ `/run` (async) แล้ว poll `/status/<id>` — **อย่าใช้ `/runsync`** เพราะงานกินเวลาเป็นนาที

`return_video: true` จะคืน mp4 เป็น base64 ซึ่งชนเพดาน response ของ RunPod (10 MB บน `/run`, 20 MB บน `/runsync`) ถ้าคลิปยาว — handler จะใส่ `warning` มาเตือนเมื่อใกล้เพดาน สำหรับคลิปทดสอบสั้นๆ พอใช้ ถ้าจะทำจริงต้องต่อ S3 (อยู่นอก scope รอบนี้)

---

## 5. ทดสอบในเครื่องก่อน (ถ้ามี NVIDIA GPU 24 GB)

```powershell
docker run --rm --gpus all $IMAGE python3 -u /opt/app/handler.py
```

runpod SDK จะอ่าน `test_input.json` อัตโนมัติ (ชี้ไปที่ไฟล์ demo ใน image) แล้วรัน 1 รอบพร้อมพิมพ์ telemetry ออกมา ไม่ต้องมี API key

ไม่มี GPU ก็เช็ค build ได้:

```powershell
docker run --rm $IMAGE python3 -c "import torch, onnxruntime; print(torch.__version__, onnxruntime.get_available_providers())"
```

---

## 6. ปิดบิลหลังวัดเสร็จ

1. **ลบ endpoint ทิ้ง** หรือเซ็ต Max Workers = 0
2. เช็ค Billing ว่าไม่มี active worker ค้าง

**หมายเหตุเรื่อง cold start แรก:** image ~17 GB ทำให้ RunPod pull รอบแรกนานหลายนาที เป็น one-time และไม่ถูกคิดเป็น compute แต่ทำให้ cold start แรกดูนานผิดปกติ — **ตัดรอบแรกทิ้งจากการวัด** รอบที่ 2 เป็นต้นไปคือตัวเลขจริง

---

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `Dockerfile` | CUDA 12.1 + py3.10 + LatentSync pinned `a229c394` + weights ทั้งหมด |
| `scripts/prefetch.py` | ดาวน์โหลด weights ตอน build + **ตรวจ API contract ของ upstream** |
| `handler.py` | RunPod handler โหลดโมเดลครั้งเดียว + วัด timing/VRAM/ราคา |
| `bench.py` | client ยิงซ้ำแล้วสรุปเป็นตารางราคา (รันบนเครื่องคุณ) |
| `test_input.json` | input สำหรับ smoke test ในเครื่อง |
| `requirements.txt` | deps ของเราเท่านั้น — LatentSync ใช้ `requirements.txt` ของ repo |

## สิ่งที่โค้ดนี้จัดการไว้แล้ว (เจอจากการอ่าน source จริง)

1. **Path แบบ relative** — upstream เรียก `DDIMScheduler.from_pretrained("configs")` และ `FaceAnalysis(root="checkpoints/auxiliary")` เป็น path relative ทั้งคู่ handler จึง `chdir` ไป repo root ตอน import
2. **InsightFace ดาวน์โหลดโมเดลตอน request แรก** — ปกติจะดึง `buffalo_l.zip` ~280 MB จาก host ที่ล่มบ่อย และเวลานั้นถูกคิดเงินรวมใน request ทำให้ราคาเพี้ยน → bake เฉพาะ 2 ไฟล์ที่ `allowed_modules` ใช้จริง (22 MB) พร้อม mirror สำรอง 3 แหล่ง
3. **onnxruntime-gpu ต้องเจอ cuDNN 9** — ถ้าหาไม่เจอมันตก CPU provider เงียบๆ face detection จะช้ามากและราคาที่วัดได้จะผิด → ลงทะเบียน NVIDIA libs ของ torch กับ `ldconfig` + เช็คตอน startup
4. **`mediapipe==0.10.11` มี wheel ถึง cp311 เท่านั้น** → ล็อกที่ python3.10 ตามที่ upstream ระบุ (3.10.13) ซึ่งเป็น system python ของ Ubuntu 22.04 พอดี ถ้าไปใช้ 3.12+ จะต้อง build mediapipe จาก source
5. **`insightface==0.7.3` บน PyPI มีแต่ sdist** (ตรวจแล้ว: `insightface-0.7.3.tar.gz` ไม่มี wheel เลย) → ต้องมี `cython` + `numpy` + `build-essential` ก่อน install ไม่งั้น build ล้มกลางทาง
6. **วิดีโอสั้นกว่าเสียง** → loop ให้ยาวพอ และ normalize เป็น 25fps + ขนาดคู่ ก่อนส่งเข้า pipeline (run ที่ล่มก็เสียเงินเหมือนกัน)
7. **DeepCache** — `cache_interval=3, cache_branch_id=0` ตรงตาม `inference.sh` ของ upstream เพราะกระทบความเร็ว = กระทบราคาโดยตรง

## ที่มาของราคา

- RunPod serverless 24 GB flex (L4 / A5000 / RTX 3090) = **$0.69/ชม = $0.000192/วินาที** · [runpod.io/pricing](https://www.runpod.io/pricing)
- LatentSync 1.6 ต้องการ VRAM **18 GB** (1.5 ต้องการ 8 GB) · [upstream README](https://github.com/bytedance/LatentSync)
- ทางเลือกจ่ายต่อ run: WaveSpeedAI ~$0.05, Replicate ~$0.10

**ถ้า volume ต่ำ (< 100 คลิป/เดือน) ให้ดูตัวเลขที่ `bench.py` คำนวณให้ดีๆ** — RunPod ถูกกว่าต่อคลิป แต่ต้องแลกกับเวลา build, การ pull image 17 GB และ cold start ที่ volume ต่ำแล้วอาจไม่คุ้มกับค่าแรงตัวเอง
