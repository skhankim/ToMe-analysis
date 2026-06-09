# ImageNet-1k Validation Set 다운로드 가이드

ToMe 논문의 모든 핵심 실험에 필요한 데이터셋. **validation set만** 사용.

## 기본 정보

| 항목 | 값 |
|------|-----|
| 이미지 수 | 50,000장 |
| 클래스 수 | 1,000 |
| 압축 파일 크기 | ~6.3 GB (`ILSVRC2012_img_val.tar`) |
| 압축 해제 후 | ~6.7 GB |
| 평균 해상도 | 가변 (≥256x256 권장 resize) |

---

## 다운로드 방법

### 공식 ImageNet 사이트


1. https://image-net.org 가입 후 `ILSVRC2012_img_val.tar` 다운로드


### 방법 4: Kaggle

```bash
pip install kaggle
# ~/.kaggle/kaggle.json 에 API 키 설정
kaggle competitions download -c imagenet-object-localization-challenge
# val 폴더만 추출
```

---

## 디렉토리 구조 

```
data/
  ├── n01440764/
  │   ├── ILSVRC2012_val_00000293.JPEG
  │   ├── ILSVRC2012_val_00002138.JPEG
  │   └── ...
  ├── n01443537/
  │   └── ...
  ├── n01484850/
  └── ... (총 1000개 클래스 폴더)
```

### 분류 스크립트

**방법 B: PyTorch 공식 스크립트**

```bash
# 압축 해제
cd /path/to/data
mkdir -p imagenet/val
tar -xf ILSVRC2012_img_val.tar -C imagenet/val

# 분류 스크립트 다운로드 및 실행
cd imagenet/val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

---

## 서버 전송

ImageNet val은 ~6.3GB이므로 sync.sh가 아닌 별도 rsync로 전송한다.

```bash
# 로컬에서 서버로 직접 전송
# server-55 (114.71.51.55)
rsync -avz --progress -e "ssh -p 22000" \
  /mnt/f/kimauve/ToMe/data/imagenet/ \
  shkim@114.71.51.55:/home/shkim/ToMe/data/imagenet/

# server-53 (114.71.51.53)
rsync -avz --progress -e "ssh -p 22000" \
  /mnt/f/kimauve/ToMe/data/imagenet/ \
  shkim@114.71.51.53:/home/shkim/ToMe/data/imagenet/
```

**또는 서버에서 직접 다운로드 (권장 — 로컬→서버 트래픽 절약)**:

```bash
ssh server-55
cd /home/shkim/ToMe
mkdir -p data
cd data

# 위의 방법 1~4 중 하나로 서버에서 직접 받기
pip install huggingface_hub
huggingface-cli login
huggingface-cli download ILSVRC/imagenet-1k \
  --repo-type dataset --include "data/val_images.tar.gz" \
  --local-dir .

# 압축 해제 및 분류
tar -xzf data/val_images.tar.gz
# ... (분류 스크립트 실행)
```

