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
| GPU | **24 GB** — อ่าน "ผลที่วัดได้จริง" ก่อนเลือก | L4 วัดแล้วได้ $0.064/คลิป แพงกว่า WaveSpeed — อย่าเลือกตามเรตต่อชั่วโมง |
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

## 4. หน้าเว็บสำหรับใช้งานจริง

ถ้าไม่อยากพิมพ์คำสั่ง มีหน้าเว็บให้เลือกไฟล์ กดปุ่ม แล้วดาวน์โหลดคลิปที่เสร็จแล้ว

```powershell
cd C:\Users\Phop\Documents\ClaudeCode\GenVideoLipsync
$env:RUNPOD_API_KEY="<key>"; $env:RUNPOD_ENDPOINT_ID="<endpoint-id>"
python webapp\server.py
```

เบราว์เซอร์จะเปิดที่ `http://127.0.0.1:8000` ให้เอง ลากไฟล์วิดีโอ (หรือรูปนิ่ง) กับไฟล์เสียงมาวาง เลือกคุณภาพ แล้วกดเริ่ม พอเสร็จจะเล่นให้ดูพร้อมปุ่มดาวน์โหลด และแสดงเวลา/VRAM/ค่าใช้จ่ายของงานนั้น

ไฟล์ที่ได้ถูกเก็บไว้ที่ `outputs\web\<job-id>.mp4` ด้วย

### ทำไมต้องมีเซิร์ฟเวอร์ ไม่ใช่ไฟล์ HTML เฉยๆ

**API key ห้ามไปอยู่ในเบราว์เซอร์** หน้าเว็บที่ยิง `api.runpod.ai` ตรงๆ จะมี key ฝังอยู่ ซึ่งใครเปิด view-source ก็เอาไปจ่ายเงินและคุม RunPod account ทั้งบัญชีได้ `server.py` ถือ key ไว้ฝั่งเครื่องคุณ เบราว์เซอร์คุยกับ `127.0.0.1` เท่านั้น

```
เบราว์เซอร์ ──► server.py (ถือ API key) ──► RunPod
  localhost          เครื่องคุณ
```

เซิร์ฟเวอร์ผูกกับ `127.0.0.1` อย่างเดียว **อย่าเปิดออกเน็ต**ถ้ายังไม่ได้ใส่ระบบล็อกอิน เพราะทุก request ที่มันส่งต่อคือเงินจริง

### ข้อจำกัด

| | |
|---|---|
| ขนาดไฟล์ | 6 MB ต่อไฟล์ — RunPod จำกัด request ที่ ~10 MB และ base64 ทำให้โตขึ้น 33% |
| ความยาวเสียง | 60 วินาที (ปรับได้ด้วย env `MAX_AUDIO_SECONDS` บน endpoint) |
| ตัวแปรที่ตั้งได้ | `PORT` (ค่าเริ่ม 8000), `EXEC_TIMEOUT_SEC` (1800), `NO_BROWSER=1` ไม่ต้องเปิดเบราว์เซอร์เอง |

ถ้าคลิปยาวกว่านี้ต้องต่อ S3 ก่อน (ดู `EXPLAIN.md` หัวข้อ 11.1)

## 5. ใช้งานตรงๆ (ถ้าไม่ผ่าน bench.py)

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

## 6. ทดสอบในเครื่องก่อน (ถ้ามี NVIDIA GPU 24 GB)

```powershell
docker run --rm --gpus all $IMAGE python3 -u /opt/app/handler.py
```

runpod SDK จะอ่าน `test_input.json` อัตโนมัติ (ชี้ไปที่ไฟล์ demo ใน image) แล้วรัน 1 รอบพร้อมพิมพ์ telemetry ออกมา ไม่ต้องมี API key

ไม่มี GPU ก็เช็ค build ได้:

```powershell
docker run --rm $IMAGE python3 -c "import torch, onnxruntime; print(torch.__version__, onnxruntime.get_available_providers())"
```

---

## 7. ปิดบิลหลังวัดเสร็จ

1. **ลบ endpoint ทิ้ง** หรือเซ็ต Max Workers = 0
2. เช็ค Billing ว่าไม่มี active worker ค้าง

**หมายเหตุเรื่อง cold start แรก:** image ~17 GB ทำให้ RunPod pull รอบแรกนานหลายนาที เป็น one-time และไม่ถูกคิดเป็น compute แต่ทำให้ cold start แรกดูนานผิดปกติ — **ตัดรอบแรกทิ้งจากการวัด** รอบที่ 2 เป็นต้นไปคือตัวเลขจริง

---

## Build ต้อง reproducible (เรียนรู้จากการล่มจริง 2026-10-16)

endpoint ที่ใช้งานได้อยู่ดีๆ ก็พังด้วย `All workers are unhealthy` ทั้งที่ **ไฟล์ใน image ไม่เปลี่ยนเลยแม้แต่ไบต์เดียว** สิ่งเดียวที่ต่างคือวันที่ build

สาเหตุลึกกว่าการที่ไลบรารีขยับเวอร์ชัน — `pip check` บน image ที่เคยใช้งานได้รายงานว่า:

```
gradio 5.24.0     ต้องการ tomlkit<0.14.0  แต่มี tomlkit 0.15.1
mediapipe 0.10.11 ต้องการ protobuf<4      แต่มี protobuf 7.36.2
```

**ชุดไลบรารีนั้นขัดแย้งกันเองมาตลอด** เกิดจาก pip ติดตั้งทีละขั้น: สองตัวนี้ลงก่อน แล้วขั้นหลังอัป dependency ของมันทับโดยไม่ย้อนกลับไปตรวจ มันไม่เคยถูกต้อง แค่บังเอิญไม่พังเพราะไม่มีใครเรียกใช้

### ทางแก้

1. **`requirements.lock.txt`** — ปักหมุด 205 แพ็กเกจ ใช้กับ `pip install` ทุกขั้นผ่าน `-c` รวมถึง `pip`/`setuptools`/`wheel` เอง
2. **ตัด `mediapipe` กับ `gradio` ออก** — ไม่มีตัวไหนอยู่ในเส้นทาง inference (`mediapipe` ใช้แค่ใน `preprocess/` + `eval/`, `gradio` ใช้แค่ใน `gradio_app.py`) และเป็นต้นเหตุที่ทำให้ล็อกเวอร์ชันไม่ได้

ผลพลอยได้: ได้ `opencv 4.9.0` ตามที่ upstream ปักหมุดจริง (เดิมได้ 4.11.0 เพราะ opencv ติดตั้งซ้อน 3 ตัวแล้วตัวหลังทับ)

### ⚠️ RunPod build ใหม่ทุกครั้งที่ push

commit ที่แก้แค่ README ก็ทำให้ production ล่มได้ เพราะมันไป build ใหม่ด้วยไลบรารีของวันนั้น — ซึ่งคือสิ่งที่เกิดขึ้นจริง `requirements.lock.txt` คือสิ่งที่กันไม่ให้เกิดซ้ำ

### ตรวจก่อน deploy เสมอ

```powershell
docker buildx build --platform linux/amd64 -t latentsync-runpod:test --load .
docker run --rm latentsync-runpod:test python3 -m pip check   # ต้องสะอาด
docker run --rm latentsync-runpod:test                         # SDK ต้องเริ่มทำงาน
```

### อัปเดต lock เมื่อตั้งใจจะอัปเกรด

```powershell
docker run --rm <image> python3 -m pip freeze | Select-String "^[A-Za-z0-9._-]+==" | Sort-Object
```

เอาผลลัพธ์แทนที่ส่วนรายการใน `requirements.lock.txt` แล้วทดสอบใหม่ทั้งชุด

---

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `requirements.lock.txt` | ปักหมุดเวอร์ชันไลบรารีทั้งหมด ทำให้ build ซ้ำได้ผลเดิม |
| `Dockerfile` | CUDA 12.1 + py3.10 + LatentSync pinned `a229c394` + weights ทั้งหมด |
| `scripts/prefetch.py` | ดาวน์โหลด weights ตอน build + **ตรวจ API contract ของ upstream** |
| `handler.py` | RunPod handler โหลดโมเดลครั้งเดียว + วัด timing/VRAM/ราคา |
| `bench.py` | client ยิงซ้ำแล้วสรุปเป็นตารางราคา (รันบนเครื่องคุณ) |
| `webapp/server.py` | เซิร์ฟเวอร์เล็กๆ ถือ API key แล้วส่งต่อให้ RunPod |
| `webapp/index.html` | หน้าเว็บอัปโหลดไฟล์ + ดาวน์โหลดผลลัพธ์ |
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

## ผลสรุป (วัดจริง 2026-10-07 ถึง 2026-10-09)

คลิป demo1 (เสียง 9.6 วินาที, วิดีโอ 1080×1920), DeepCache เปิด, ราคาเป็นแบบ warm:

| GPU | เรต | steps | Inference | RTF | **$/คลิป** |
|---|---|---|---|---|---|
| L4 | $0.69/ชม | 20 | 324s | 33.7x | $0.0642 |
| A5000 | $0.69/ชม | 20 | 226s | 23.6x | $0.0445 |
| **A5000** | **$0.69/ชม** | **10** | **159s** | **16.6x** | **$0.0315** |

| ทางเลือก | $/คลิป | |
|---|---|---|
| **RunPod A5000 steps 10** | **$0.0315** | **ถูกสุด** |
| WaveSpeedAI | $0.0500 | แพงกว่า 1.59x |
| Replicate | $0.1000 | แพงกว่า 3.18x |

### โครงสร้างต้นทุนเวลา

จาก 2 จุดวัด (steps 20 = 226s, steps 10 = 159s) แยกได้เป็น:

```
เวลา ≈ 92 วินาที (คงที่) + 6.7 วินาที × จำนวน steps
```

ส่วนคงที่ 92 วินาทีคือ face detection + VAE + I/O ซึ่งไม่ลดตาม steps — แปลว่าลด steps ต่ำกว่า 10 ได้ผลน้อยลงเรื่อยๆ (steps 5 ≈ 126s ≈ $0.024)

### steps 10 เสียคุณภาพไหม — วัดแล้ว ไม่เสีย

- ภาพนิ่งบริเวณปาก: ต่างกันเล็กน้อย ขอบปาก/ฟันนุ่มกว่านิดหน่อย ไม่มี artifact ไม่มีรอยต่อ mask
- **ความนิ่งระหว่างเฟรม** (จุดที่ steps ต่ำมักพัง): วัดด้วย `tblend=difference` + `signalstats` บนบริเวณปากตลอดคลิป
  - steps 20: mean 4.975, peak 14.007
  - steps 10: mean 4.971, peak 13.994
  - ต่างกัน 0.08% — แยกไม่ออก

ยังไม่ได้ตรวจ: ความแม่นของ lip-sync เทียบกับเสียง (ต้องดูด้วยหูคน) และผลกับคลิป/ใบหน้าแบบอื่น

### ⚠️ ความเสี่ยงที่เหลือ: การ์ดในกลุ่มเดียวกันเร็วไม่เท่ากัน

กลุ่ม 24GB $0.69/ชม มีทั้ง L4, A5000, RTX 3090 และ RunPod จับตัวไหนก็ได้ที่ว่าง **L4 ช้ากว่า A5000 1.4 เท่าที่ราคาเท่ากัน** ราคาจริงจึงแกว่งระหว่าง $0.0315 (A5000) ถึง $0.045 (L4) ที่ steps 10

ตอนทดสอบบน L4 ยังเจอ request ล้มด้วย `executionTimeout exceeded` ซึ่งบน A5000 ไม่เจอเลย — น่าจะเพราะ L4 เหลือ VRAM แค่ 2.8 GB (A5000 เหลือ 5.4 GB) แต่ยังไม่ได้ยืนยันจาก log

## รายละเอียดการวัดรอบแรก (2026-10-07)

วัดบน **NVIDIA L4** ($0.69/ชม) ด้วยคลิป demo1 (เสียง 9.6 วินาที, วิดีโอ 1080×1920), steps 20, DeepCache เปิด:

| | ค่าที่วัดได้ |
|---|---|
| Inference | **323.78 วินาที** |
| RunPod คิดเงิน | 335.02 วินาที |
| **Realtime factor** | **33.73x** |
| **ราคา/คลิป** | **$0.0642** |
| โหลดโมเดล (cold) | 13.72 วินาที — แค่ 4% ของเวลาทั้งหมด |
| Peak VRAM | **19.21 GB** จาก 22.0 GB (ใกล้เคียง 18 GB ที่ upstream ระบุ) |
| Platform overhead | 0.8 วินาที ($0.0002) — การส่ง base64 แทบไม่มีผล |

### ข้อสรุปที่ขัดกับที่คาดไว้ตอนแรก

1. **ประมาณการแรกผิดไป 4.3 เท่า** — เดาว่า inference 60-90 วินาที ของจริง 324 วินาที ราคาจึงเป็น $0.064 ไม่ใช่ $0.015
2. **RunPod บน L4 แพงกว่า WaveSpeedAI** ($0.05/run) ซึ่งไม่ต้อง build ไม่ต้องดูแลเลย
3. **การเลือก GPU ที่ถูกสุดต่อชั่วโมงเป็นเหตุผลที่ผิด** — ราคาต่อคลิป = เวลา × เรต L4 เป็นการ์ดประหยัดไฟที่ช้า ความช้ากลืนส่วนที่ประหยัดต่อชั่วโมงไปหมด GPU ที่แรงกว่าและแพงกว่าต่อชั่วโมงอาจถูกกว่าต่อคลิป
4. **cold start ไม่ใช่ปัญหา** — โหลดโมเดลแค่ 13.7 วินาที เพราะ weights ถูก bake ไว้ในอิมเมจ ราคา warm กับ cold ต่างกันไม่ถึง 5%
5. **VRAM เหลือแค่ 2.8 GB บน L4** และ request ถัดมาล้มด้วย `executionTimeout exceeded` — ยังไม่ยืนยันสาเหตุ แต่สงสัยว่า memory fragmentation ทำให้ช้าลงมาก

ถ้าจะวัด GPU อื่น **ต้องส่ง `--usd-per-hour` ให้ตรงเรตด้วย** ไม่งั้นราคาที่ได้จะผิด (default คือ 0.69 = เรต L4)

## ที่มาของราคา

- RunPod serverless 24 GB flex (L4 / A5000 / RTX 3090) = **$0.69/ชม = $0.000192/วินาที** · [runpod.io/pricing](https://www.runpod.io/pricing)
- LatentSync 1.6 ต้องการ VRAM **18 GB** (1.5 ต้องการ 8 GB) · [upstream README](https://github.com/bytedance/LatentSync)
- ทางเลือกจ่ายต่อ run: WaveSpeedAI ~$0.05, Replicate ~$0.10

**ถ้า volume ต่ำ (< 100 คลิป/เดือน) ให้ดูตัวเลขที่ `bench.py` คำนวณให้ดีๆ** — RunPod ถูกกว่าต่อคลิป แต่ต้องแลกกับเวลา build, การ pull image 17 GB และ cold start ที่ volume ต่ำแล้วอาจไม่คุ้มกับค่าแรงตัวเอง
