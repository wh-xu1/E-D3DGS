# 📸 Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction. **ICCV 2025**

<div align="center">
  ▶️ <a href="assets/Video.mp4">Click here to watch the demo video</a>
</div>

<br>

Official implementation of our ICCV 2025 paper:
[**Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction**](https://arxiv.org/pdf/2411.16180)
by **Wenhao Xu**, **Wenming Weng**, **Yueyi Zhang**, and **Zhiwei Xiong**.

---

## 📂 Dataset

We provide both **synthetic** and **real-world dynamic scene datasets**, available at:
👉 [**Hugging Face: Heisenberg-xu / E-D3DGS**](https://huggingface.co/datasets/Heisenberg-xu/E-D3DGS)

---

## 🚀 Get Started

### 🔧 Environment Setup

* CUDA 11.8
* PyTorch 2.1.2

```bash
pip install submodules/depth-diff-gaussian-rasterization/
pip install submodules/simple-knn
```

### 🏃 Training Example

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config 'config/lego.ini' --override 'model_path=output/Lego'
```
---

## 🙏 Acknowledgements

This project builds upon the excellent works of:

* [**D-3DGS**](https://github.com/ingra14m/Deformable-3D-Gaussians)
* [**4DGS**](https://github.com/hustvl/4DGaussians)

We sincerely thank the authors for open-sourcing their codes.

---

## 📄 Citation

If you find our work useful, please consider citing:

```bibtex
@article{xu2024event,
  title={Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction},
  author={Xu, Wenhao and Weng, Wenming and Zhang, Yueyi and Xu, Ruikang and Xiong, Zhiwei},
  journal={arXiv preprint arXiv:2411.16180},
  year={2024}
}
```

---

## 📬 Contact

For any questions or feedback, feel free to reach out:
📧 **[wh-xu@mail.ustc.edu.cn](mailto:wh-xu@mail.ustc.edu.cn)**
