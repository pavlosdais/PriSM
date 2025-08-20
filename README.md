<h1 align="center">
  <span style="font-family: Consolas;">PriSM: Prior-Guided Search Methods for Query Efficient Adversarial Black-Box Attacks</span>
</h1>

<div align="center">
  <a href="https://www.linkedin.com/in/pavlosdais/" target="_blank">Pavlos Ntais</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=Ro0CzSMAAAAJ&hl=en" target="_blank">Thanassis Avgerinos</a><sup>1</sup> (Supervisor)
  <br>
  <sup>1</sup> National and Kapodistrian University of Athens, Department of Informatics and Telecommunications
</div>
<br>

![logo.png](./assets/logo.png)

**TL;DR**:
Adversarial attacks against black-box models face a fundamental trade-off between attack success rate and query efficiency. This work introduces **PriSM**, a framework containing two novel hybrid attack methods that leverage transfer-based priors from surrogate models to significantly improve query efficiency while maintaining high success rates. We first propose **TGEA**, which uses transfer-based attacks to provide a high-quality initial population for an evolutionary search. We then develop **SGSA**, an advanced attack that uses the surrogate's gradients to create a saliency map, intelligently guiding the location and size of perturbations at every step of the search process. Both methods bridge the gap between transfer learning and query-based optimization, establishing new benchmarks for query-efficient adversarial attacks.

---

### Contributions
* **A Novel Hybrid Framework**: Our work bridges the gap between transfer learning and query-based optimization, leveraging the strengths of both methods with two unique approaches that use prior information from surrogate models to guide the search process.
* **Transfer-Guided Evolutionary Attack (TGEA)**: A global optimization attack that effectively "warm-starts" a powerful evolutionary algorithm (CMA-ES), ensuring the search begins with a high-quality population that is already close to the target model's decision boundary.
* **Saliency-Guided Square Attack (SGSA)**: A sophisticated local search attack that replaces the random search of the state-of-the-art [**Square Attack**](https://arxiv.org/abs/1912.00049) with intelligent, multi-faceted guidance. It uses a surrogate's saliency map to focus on vulnerable regions, adapt the perturbation size, and dynamically update its strategy.
* **State-of-the-Art Performance**: Our methods demonstrate a new state of the art in the trade-off between query efficiency and attack success rate. SGSA emerges as the more query efficient attack, while TGEA-SEGI achieves the highest success rates on complex models.

---

### Methodology

#### Transfer-Guided Evolutionary Attack (TGEA)
TGEA is a **global search optimization method** designed to find complex, non-local perturbations. It operates by initializing the population of a powerful evolutionary algorithm, [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES), with adversarial examples generated on a surrogate model. This informed initialization ensures the search begins in promising regions of the input space, dramatically reducing the queries needed for convergence compared to a random start. We propose two distinct initialization strategies:
* **TASI (Transfer-Attack Seeded Initialization)**: Seeds the population using a diverse portfolio of off-the-shelf attacks (e.g., PGD) run on the surrogate model. This provides a robust and diverse set of starting points for the evolutionary search.
* **SEGI (Surrogate-Evolved Genetic Initialization)**: An advanced meta-optimization strategy. It uses a Genetic Algorithm that runs *entirely on the surrogate model* to evolve a bespoke population of highly-fit and diverse candidates. This provides a powerful initial population, leading to higher success rates on complex target models.

#### Saliency-Guided Square Attack (SGSA)
SGSA is an enhanced **local search algorithm** that extends the state-of-the-art [Square Attack](https://arxiv.org/abs/1912.00049) and makes it significantly more query-efficient. Instead of relying on a purely random search, SGSA uses a surrogate model to create a "vulnerability map" that intelligently guides every step of the attack. Its key components include:
* **Hybrid Guidance Map**: SGSA generates a dynamic heatmap by combining multi-scale saliency, an optional attention mechanism, and temporal smoothing. This map identifies the most promising regions of an image to perturb.
* **Probabilistic Location Sampling**: The attack treats the guidance map as a probability distribution, focusing its perturbations on the most vulnerable areas of the image instead of searching randomly.
* **Adaptive Sizing**: The size of the square perturbation is made inversely proportional to the local saliency. This allows the attack to make small, precise changes in highly sensitive areas and larger, exploratory changes elsewhere.
* **Robust Fallback Mechanism**: If the saliency guidance is not effective (due to poor transferability), the attack automatically reverts to a random search, ensuring robustness.

---

### Experimental Evaluation
Our experiments on [MNIST](https://en.wikipedia.org/wiki/MNIST_database) and [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) reveal a clear trade-off between the query efficiency of SGSA and the high success rate of TGEA-SEGI.

#### MNIST Results
On the simpler MNIST dataset, the guided local search of **SGSA** is dominant, achieving the highest Attack Success Rate (ASR) with the lowest Average Queries (AQ) in nearly all scenarios.

| Target Model | &epsilon; | TGEA-TASI (ASR/AQ) | TGEA-SEGI (ASR/AQ) | Random (ASR/AQ) | Square (ASR/AQ) | SimBA (ASR/AQ) | **SGSA (ASR/AQ)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | 0.2 | 65.76% / 250.05 | 47.15% / 255.86 | 37.77% / 519.96 | 87.86% / 239.38 | 59.32% / 306.05 | **89.89% / 177.48** |
| | 0.3 | 91.76% / 130.24 | 83.69% / 164.57 | 76.85% / 398.30 | **99.93%** / 91.45 | 94.14% / 199.98 | 99.81% / **54.37** |
| **Model B** | 0.2 | 82.11% / 197.74 | 60.34% / 272.78 | 51.37% / 556.02 | 91.79% / 264.05 | 69.07% / 265.60 | **93.98% / 181.80** |
| | 0.3 | 97.13% / 74.94 | 90.78% / 125.23 | 88.39% / 357.62 | 100.00% / 97.73 | 81.50% / 151.06 | **100.00% / 56.68** |

#### CIFAR-10 Results
On the more complex CIFAR-10 dataset, **TGEA-SEGI** consistently achieves the highest ASR. At the same time, **SGSA** remains the most query-efficient method by a significant margin.

| Target Model | &epsilon; | TGEA-TASI (ASR/AQ) | **TGEA-SEGI (ASR/AQ)** | Random (ASR/AQ) | Square (ASR/AQ) | SimBA (ASR/AQ) | **SGSA (ASR/AQ)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **InceptionV3**| 0.1 | 89.69% / 120.75 | **91.43%** / 105.26 | 85.27% / 148.62 | 88.20% / 76.64 | 75.89% / 105.63 | 85.64% / **53.47** |
| | 0.2 | 97.10% / 53.82 | **98.36%** / **27.95** | 95.56% / 78.30 | 94.39% / 38.15 | 77.92% / 81.80 | 93.04% / 30.21 |
| **VGG16** | 0.1 | 88.03% / 180.28 | **89.80%** / 127.82 | 83.29% / 198.42 | 81.56% / 91.70 | 76.33% / 163.42 | 82.85% / **77.62** |
| | 0.2 | 97.20% / 59.73 | **98.19%** / 34.80 | 92.61% / 80.85 | 91.33% / 41.95 | 77.94% / 104.60 | 91.04% / **34.05** |

---

### How to Use

The full source code is available in this repository. Here is how you can run the attack:

**1. Clone the repository and install dependencies:**
```bash
git clone https://github.com/pavlosdais/PriSM.git
cd src
pip install -r requirements.txt
```

**2. (Optionally) Download CIFAR-10 weights**
```bash
python3 ./models/weights.py --download_weights 1
cp -r ./cifar10_models/state_dicts/ .
```

**3. Run the attack:**
```
python3 attack.py \
  --model 'inception' \
  --dataset 'cifar10' \
  --attack 'TASI'
```

Available Options:
- Models: `model_a`, `model_b`, `vgg`, `inception`
- Datasets: `mnist`, `cifar10`
- Attacks: `TASI`, `SEGI`, `SGSA`

---

This work is conducted as part of my ongoing thesis at the National and Kapodistrian University of Athens, Department of Informatics and Telecommunications, under the supervision of prof. Thanassis Avgerinos.